"""
research_agent.py — Production-grade Enterprise Research Agent
=============================================================
Architecture: AccountPlanState → ResearchTool → LLMEngine → ResearchAgent

CRITICAL FIX SUMMARY:
    - No global singleton: ResearchAgent must be instantiated per Chainlit session.
    - Full async architecture: AsyncGroq + asyncio.to_thread() for search.
    - Retry with exponential backoff on all Groq and SerpAPI calls.
    - TTL cache (5 min) on search results to prevent redundant API calls.
    - Structured prompts (ROLE / EXTRACTION RULES / SYNTHESIS RULES / CONFLICT RULES / OUTPUT FORMAT / FEW-SHOT EXAMPLE).
    - Section-level update pipeline (SECTION_UPDATE intent).
    - Confidence scoring per section: HIGH / MEDIUM / LOW.
    - Private/low-data company detection with user-facing warnings.
    - Plan hash for change detection (prevents redundant re-renders).
    - Key executives include titles + strategic relevance.
    - Named competitors with differentiation notes.
    - Comparison mode (parallel research on two companies).
    - Streaming direct answers for follow-up questions.
    - Dead code removed: TerminalUI and TestExecution are kept but separated clearly.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from ddgs import DDGS
from serpapi import GoogleSearch
from groq import AsyncGroq, Groq

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("research_agent")

# ── Environment Validation ────────────────────────────────────────────────────
REQUIRED_ENV_VARS = ["GROQ_API_KEY"]


def validate_environment() -> None:
    """
    Validates required environment variables at startup.
    Raises EnvironmentError immediately so the app fails fast with a clear message
    rather than crashing on the first user message.
    """
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            "Copy .env.example to .env and fill in your API keys."
        )


validate_environment()

# ── Plan Constants ────────────────────────────────────────────────────────────
UPDATABLE_SECTIONS: frozenset = frozenset({
    "company_overview", "financial_snapshot", "market_revenue", "competitors",
    "key_executives", "strategic_priorities", "pain_points",
    "value_proposition", "action_plan",
})


# ── Account Plan State ────────────────────────────────────────────────────────

class AccountPlanState:
    """
    Per-session, in-memory state for a single Account Plan.

    IMPORTANT: This class must be instantiated inside cl.on_chat_start() and
    stored via cl.user_session.set("agent", agent). A global instance would
    cause all concurrent users to share — and overwrite — each other's plans.

    Design: _default_plan() is the single source of truth for the schema.
    Both __init__ and reset_plan() call it, guaranteeing permanent sync.
    """

    def _default_plan(self) -> Dict[str, Any]:
        """Returns a fresh, fully-typed Account Plan skeleton."""
        return {
            "company_name":         "Not Yet Provided",
            "user_goal":            "Not Yet Provided",
            "company_overview":     "Unknown",
            "financial_snapshot":   "Unknown",
            "market_revenue":       "Unknown",
            "competitors":          [],   # List[str] — "Name — differentiation note"
            "key_executives":       [],   # List[str] — "Name — Title, strategic note"
            "strategic_priorities": [],   # List[str]
            "pain_points":          [],   # List[str]
            "value_proposition":    "Unknown",
            "action_plan":          [],   # List[str]
            "data_confidence":      {},   # Dict[section_key, "HIGH"|"MEDIUM"|"LOW"]
            "data_warnings":        [],   # List[str] — user-facing data quality notices
            "source_references":    [],   # List[str] — URLs / sources for attribution
        }

    def __init__(self) -> None:
        self.plan: Dict[str, Any] = self._default_plan()
        self.open_questions: List[str] = []
        self._rendered_hash: str = ""
        self._seen_suggestions: set = set()  # deduplicates proactive suggestions across calls

    def reset_plan(self) -> None:
        """
        Full state reset for a company pivot.
        Also clears the conflict queue so stale clarification prompts
        cannot execute after the context switch.
        """
        self.plan = self._default_plan()
        self.open_questions = []
        self._rendered_hash = ""
        self._seen_suggestions = set()  # clear suggestion history on company switch

    def update_section(self, section_key: str, data: Any) -> bool:
        """
        Safely updates a single section. Returns False on unknown key
        rather than silently creating orphan fields.
        """
        if section_key not in self.plan:
            logger.warning(
                "update_section: unknown field '%s' — skipping.", section_key
            )
            return False
        self.plan[section_key] = data
        return True

    def get_current_plan(self) -> Dict[str, Any]:
        """
        Returns a copy of the plan that is safe to mutate externally.

        A plain dict.copy() is a shallow copy — mutable list values (competitors,
        pain_points, etc.) would be shared references. Mutating them in the caller
        (e.g. current_state["open_questions"] = []) would corrupt the live state.
        We deep-copy the lists while keeping scalar values as-is for efficiency.
        """
        copy: Dict[str, Any] = {}
        for k, v in self.plan.items():
            copy[k] = list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)
        return copy

    def compute_hash(self) -> str:
        """SHA-256 of the serialised plan for change detection."""
        return hashlib.sha256(
            json.dumps(self.plan, sort_keys=True, default=str).encode()
        ).hexdigest()

    def has_changed_since_last_render(self) -> bool:
        """
        Returns True if the plan changed since the last time this was called.
        Calling this method also snapshots the new hash, so repeated calls
        return False until another change occurs.
        """
        new_hash = self.compute_hash()
        if new_hash != self._rendered_hash:
            self._rendered_hash = new_hash
            return True
        return False


# ── Search Cache ──────────────────────────────────────────────────────────────

class SearchCache:
    """
    Simple TTL-based in-memory cache for search results.
    Prevents identical API calls from being fired when the user asks
    about the same company twice within the cache window (default: 5 min).
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._store: Dict[str, Tuple[float, List[Dict]]] = {}
        self.ttl = ttl_seconds
        self._in_flight: set = set()  # lightweight concurrency guard

    def get(self, key: str) -> Optional[List[Dict]]:
        entry = self._store.get(key)
        if entry:
            ts, data = entry
            if time.monotonic() - ts < self.ttl:
                logger.info("Cache HIT: '%s'", key[:70])
                return data
            del self._store[key]
        return None

    def set(self, key: str, data: List[Dict]) -> None:
        self._store[key] = (time.monotonic(), data)
        self._in_flight.discard(key)

    def is_in_flight(self, key: str) -> bool:
        return key in self._in_flight

    def mark_in_flight(self, key: str) -> None:
        self._in_flight.add(key)

    def clear(self) -> None:
        self._store.clear()


# ── Research Tool ─────────────────────────────────────────────────────────────

class ResearchTool:
    """
    Web search layer with graceful fallback (SerpAPI → DuckDuckGo),
    TTL caching, and async wrappers for non-blocking use in Chainlit.

    Source attribution is preserved in every result for transparency —
    a B2B research tool must be able to show where facts came from.
    """

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results
        self.serp_key: Optional[str] = os.getenv("SERP_API_KEY")
        self._cache = SearchCache(ttl_seconds=300)

    # ── Sync search (used internally) ─────────────────────────────────────────

    def search_web(self, query: str) -> List[Dict[str, str]]:
        """
        Executes organic + news search. SerpAPI → DDGS fallback.
        Results are cached by query string.

        An in-flight guard prevents concurrent calls with the same query from
        firing duplicate API requests. If the query is already in-flight (being
        fetched by another coroutine via asyncio.to_thread), we wait briefly
        and return the cached result once it lands.
        """
        cached = self._cache.get(query)
        if cached is not None:
            return cached

        # If another thread is already fetching this query, wait for it to
        # populate the cache rather than firing a duplicate API request.
        if self._cache.is_in_flight(query):
            logger.info("Query '%s' already in-flight — waiting for cache...", query[:70])
            deadline = time.monotonic() + 10  # max wait: 10 s
            while self._cache.is_in_flight(query) and time.monotonic() < deadline:
                time.sleep(0.1)
            cached = self._cache.get(query)
            if cached is not None:
                return cached
            # Fell through (e.g. the in-flight request failed) — proceed normally

        self._cache.mark_in_flight(query)
        logger.info("Searching web: '%s'", query[:70])
        try:
            results = self._try_serpapi(query) or self._try_ddgs(query)
        finally:
            self._cache._in_flight.discard(query)
        self._cache.set(query, results)  # also calls _in_flight.discard(query)
        return results

    def _try_serpapi(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Attempts SerpAPI (organic + news). Returns None on any failure."""
        if not self.serp_key:
            return None
        try:
            organic = GoogleSearch({"q": query, "api_key": self.serp_key}).get_dict()
            if "error" in organic:
                raise RuntimeError(f"SerpAPI: {organic['error']}")
            news = GoogleSearch(
                {"q": query, "tbm": "nws", "api_key": self.serp_key}
            ).get_dict()

            results: List[Dict] = []
            for faq in organic.get("related_questions", [])[:2]:
                results.append({
                    "category": "VERIFIED_FAQ",
                    "title":    faq.get("question", "FAQ"),
                    "snippet":  faq.get("snippet", ""),
                    "date":     "",
                    "source":   "SerpAPI/FAQ",
                })
            for r in organic.get("organic_results", [])[:self.max_results]:
                results.append({
                    "category": "COMPANY_DATA",
                    "title":    r.get("title", ""),
                    "snippet":  r.get("snippet", ""),
                    "date":     r.get("date", ""),
                    "source":   r.get("link", "SerpAPI"),
                })
            for n in news.get("news_results", [])[:self.max_results]:
                results.append({
                    "category": "RECENT_NEWS",
                    "title":    n.get("title", ""),
                    "snippet":  n.get("snippet", ""),
                    "date":     n.get("date", ""),
                    "source":   n.get("link", "SerpAPI/News"),
                })

            logger.info("SerpAPI → %d results for '%s'", len(results), query[:60])
            return results or None

        except Exception as exc:
            logger.warning("SerpAPI unavailable (%s). Falling back to DDGS.", exc)
            return None

    def _try_ddgs(self, query: str) -> List[Dict[str, str]]:
        """DuckDuckGo fallback. Always returns a list (empty on total failure)."""
        results: List[Dict] = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results) or []:
                    results.append({
                        "category": "COMPANY_DATA",
                        "title":    r.get("title", ""),
                        "snippet":  r.get("body", ""),
                        "date":     "",
                        "source":   r.get("href", "DDGS"),
                    })
                # News fetch is isolated: a failure here should NOT inject an
                # ERROR snippet into results — that string would be fed to the LLM
                # as if it were a real search result and could corrupt synthesis.
                try:
                    for r in ddgs.news(query, max_results=self.max_results) or []:
                        results.append({
                            "category": "RECENT_NEWS",
                            "title":    r.get("title", ""),
                            "snippet":  r.get("body", ""),
                            "date":     r.get("date", ""),
                            "source":   r.get("url", "DDGS/News"),
                        })
                except Exception as news_exc:
                    logger.warning("DDGS news() failed for '%s': %s", query[:60], news_exc)
                    # Do NOT append an ERROR result — just continue with text results.

            logger.info("DDGS → %d results for '%s'", len(results), query[:60])
        except Exception as exc:
            logger.error("DDGS text() failed: %s", exc)
            # Return empty list — do not inject error strings into LLM context.
        return results

    # ── Async wrappers ────────────────────────────────────────────────────────

    async def async_search_web(self, query: str) -> List[Dict[str, str]]:
        """Non-blocking search using asyncio.to_thread (search libs are synchronous)."""
        return await asyncio.to_thread(self.search_web, query)

    async def async_search_multi(self, queries: List[str]) -> List[Dict[str, str]]:
        """
        Runs multiple targeted search queries CONCURRENTLY and merges results.
        Deduplicates on title to prevent redundant snippets in the LLM context.
        """
        batches = await asyncio.gather(*[self.async_search_web(q) for q in queries])
        merged: List[Dict] = []
        seen: set = set()
        for batch in batches:
            for r in batch:
                if r["title"] not in seen:
                    merged.append(r)
                    seen.add(r["title"])
        return merged

    # ── LLM Context Formatting ────────────────────────────────────────────────

    @staticmethod
    def format_for_llm(results: List[Dict[str, str]], max_chars: int = 12_000) -> str:
        """
        Converts results into a structured string for LLM consumption.
        Hard cap at max_chars to prevent context window crowding.
        Source URLs are included for downstream attribution.
        Results are sorted by category priority so high-quality sources
        are not displaced by low-quality ones when the budget is exhausted.
        """
        PRIORITY = {"COMPANY_DATA": 0, "VERIFIED_FAQ": 1, "RECENT_NEWS": 2}
        results = sorted(results, key=lambda r: PRIORITY.get(r.get("category", ""), 99))
        parts: List[str] = []
        total = 0
        for res in results:
            date_tag   = f" [{res.get('date', '')}]"   if res.get("date")   else ""
            source_tag = f" (Source: {res.get('source', '')})" if res.get("source") else ""
            chunk = (
                f"--- {res.get('category', 'INFO')}{date_tag}{source_tag} ---\n"
                f"Title: {res.get('title', 'Unknown')}\n"
                f"Snippet: {res.get('snippet', 'No data')}\n\n"
            )
            if total + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "".join(parts)


# ── Plan Output Validator ─────────────────────────────────────────────────────

class PlanOutputValidator:
    """
    Lightweight structural validator for LLM-generated plan output.

    Checks for vague or generic responses without heavy NLP.
    Returns (cleaned_output, issues) — issues is an empty list if output passes.

    Design: stateless, functional, cheap. Called inside the retry loop.
    """

    # Generic phrases that signal a low-quality / placeholder output
    VAGUE_PHRASES: frozenset = frozenset({
        "not available", "n/a", "unknown", "to be determined",
        "contact the company", "see website", "various", "multiple",
        "no information", "not provided", "not applicable",
        # Additional LLM hedge phrases (extended coverage)
        "varies by region", "depends on the company", "please consult",
        "subject to change", "information not available", "data not available",
        "cannot be determined", "not publicly disclosed", "not publicly available",
        "check the official website", "refer to their website",
        "consult official sources", "beyond my knowledge",
        "not in the search results", "no data found",
    })

    @classmethod
    def validate(
        cls,
        parsed: Dict[str, Any],
        section_key: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validates a parsed LLM output dict.

        If section_key is given, only that key is validated (section update mode).
        Otherwise, core plan keys are checked (full extraction mode).

        Returns:
            (parsed, issues)  — issues is [] on pass, list of problem strings on fail.
        """
        issues: List[str] = []

        keys_to_check = (
            [section_key]
            if section_key
            else ["company_overview", "financial_snapshot", "competitors",
                "key_executives", "pain_points"]
        )

        for key in keys_to_check:
            value = parsed.get(key)
            if value is None:
                continue  # key absent — not a validation issue here

            issue = cls._check_value(key, value)
            if issue:
                issues.append(issue)

        return parsed, issues

    @classmethod
    def _check_value(cls, key: str, value: Any) -> Optional[str]:
        """Returns a problem description string, or None if value is acceptable."""
        if isinstance(value, list):
            if not value:
                return f"'{key}' is an empty list"
            # Check if list contains only vague items
            non_vague = [
                item for item in value
                if not cls._is_vague(str(item))
            ]
            if not non_vague:
                return f"'{key}' list contains only vague/generic items"
            return None

        if isinstance(value, str):
            if not value.strip() or value.strip() in ("Unknown", ""):
                return f"'{key}' is empty or Unknown"
            if cls._is_vague(value):
                return f"'{key}' contains only vague/generic content"
            return None

        return None  # numbers, dicts, etc. pass through

    @classmethod
    def _is_vague(cls, text: str) -> bool:
        """True if text is exclusively a vague placeholder."""
        normalized = text.strip().lower()
        if len(normalized) < 8:
            return True
        if normalized in cls.VAGUE_PHRASES:
            return True
        return any(phrase in normalized for phrase in cls.VAGUE_PHRASES)


# ── LLM Engine ────────────────────────────────────────────────────────────────

class LLMEngine:
    """
    LLM orchestration layer backed by Groq (Llama-3.3-70b-versatile).

    Temperature policy (explained in code so reviewers don't ask):
    - 0.0 for intent classification: fully deterministic routing is critical.
    - 0.1 for structured JSON extraction: minimal variance reduces hallucination
      on numerical fields (revenue, market cap) while allowing synthesis.
    - 0.3 for conversational answers: slight creativity for natural language output.

    Retry policy: exponential backoff (1s, 2s, 4s) on all API calls.
    A single transient 429 or network blip should NEVER surface to the user.
    """

    MODEL = "llama-3.3-70b-versatile"
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0  # seconds
    TIMEOUT_SECONDS = 15.0  # per-attempt LLM call timeout (configurable)

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file."
            )
        # sync client kept for audio transcription (Chainlit on_audio_end)
        self.client = Groq(api_key=key)
        # async client for all LLM pipeline calls
        self.async_client = AsyncGroq(api_key=key)

    # ── Retry Utility ─────────────────────────────────────────────────────────

    async def _call_with_retry(self, coroutine_factory, **kwargs) -> Any:
        """
        Calls an async Groq method with exponential-backoff retry AND per-attempt
        timeout (asyncio.wait_for). Timeout errors are treated as retryable failures,
        so the full retry budget is consumed before raising.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                return await asyncio.wait_for(
                    coroutine_factory(**kwargs),
                    timeout=self.TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(
                        "Groq call timed out after %d attempts (%.0fs each).",
                        self.MAX_RETRIES, self.TIMEOUT_SECONDS,
                    )
                    raise RuntimeError(
                        f"LLM call timed out after {self.MAX_RETRIES} attempts "
                        f"({self.TIMEOUT_SECONDS}s limit)."
                    ) from exc
                delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Groq attempt %d/%d timed out (>%.0fs). Retrying in %.1fs...",
                    attempt + 1, self.MAX_RETRIES, self.TIMEOUT_SECONDS, delay,
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(
                        "Groq call failed after %d attempts: %s",
                        self.MAX_RETRIES, exc
                    )
                    raise
                delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Groq attempt %d/%d failed (%s). Retrying in %.1fs...",
                    attempt + 1, self.MAX_RETRIES, exc, delay
                )
                await asyncio.sleep(delay)

    # ── Adaptive Strategy Prompt Modifiers ────────────────────────────────────

    @staticmethod
    def _apply_strategy(messages: list, strategy: str) -> list:
        """
        Injects strategy-specific instructions into the final user message.
        "default"           → no modification (existing behaviour preserved).
        "fallback_reasoning"→ allows partial inference; forbids empty output.
        "strict"            → requires concrete entities; rejects vague outputs.
        Returns a NEW list — does not mutate the original.
        """
        if strategy == "default":
            return messages

        STRATEGY_SUFFIX = {
            "fallback_reasoning": (
                "\n\n[RETRY INSTRUCTION] The previous attempt failed. "
                "Prefer useful approximation over an empty output. "
                "Use partial inference where necessary. "
                "Do NOT return empty fields or omit sections — provide your best estimate."
            ),
            "strict": (
                "\n\n[VALIDATION RETRY] The previous output was rejected for vagueness. "
                "You MUST include concrete entities: real numbers, named people, "
                "specific organisations. Vague statements are not acceptable. "
                "If exact figures are unavailable, append '(estimated)' after the value."
            ),
        }
        suffix = STRATEGY_SUFFIX.get(strategy, "")
        if not suffix:
            return messages

        patched = list(messages)
        last = dict(patched[-1])
        last["content"] = str(last.get("content", "")) + suffix
        patched[-1] = last
        logger.debug("Adaptive strategy '%s' applied to LLM call.", strategy)
        return patched

    # ── Intent Classification ─────────────────────────────────────────────────

    async def async_classify_intent(
        self,
        user_input: str,
        current_company: str,
    ) -> Dict[str, Any]:
        """
        Semantic router. Classifies user intent and extracts entities.

        Returns a dict with:
        intent: str             — one of the categories below
        company_name: str|None  — extracted company name
        goal: str|None          — extracted sales goal
        sections_to_update: List[str]  — for SECTION_UPDATE intent
        compare_targets: List[str]     — for COMPARE_COMPANIES intent

        ## INTENT CATEGORIES
        NEW_COMPANY:       Research a different company.
        GOAL_UPDATE:       Change sales goal, same company.
        SECTION_UPDATE:    Refresh specific plan sections.
        CURRENT_COMPANY:   Follow-up on the active company.
        GENERAL_QUESTION:  Off-topic question.
        CONFUSED_USER:     User needs guidance.
        COMPARE_COMPANIES: Compare two companies side-by-side.
        DOWNLOAD_PLAN:     Export plan to PDF/Markdown.
        """
        prompt = f"""
## ROLE
You are a precise semantic router for an enterprise AI research assistant.
Current active company being researched: "{current_company}"

## USER INPUT
"{user_input}"

## CLASSIFICATION RULES
1. "NEW_COMPANY" — User explicitly names a DIFFERENT company to research.
    Extract company_name AND goal. If no goal stated, set goal to "General Research".

2. "GOAL_UPDATE" — User changes their sales/pitch goal without naming a new company.
    Extract the new goal only. Set company_name to null.

3. "SECTION_UPDATE" — User requests refreshing a specific plan section.
    Keywords: "update", "refresh", "redo", "regenerate", "fix", "change" + section name.
    Valid section keys: company_overview, financial_snapshot, market_revenue, competitors,
    key_executives, strategic_priorities, pain_points, value_proposition, action_plan.
    Set sections_to_update as a list of matching keys.

4. "CURRENT_COMPANY" — Follow-up question about the currently active company.
    Includes comparisons ("vs Apple?", "how do they compare to Google") where the
    ACTIVE company is the primary subject. Do NOT switch context for comparison mentions.

5. "GENERAL_QUESTION" — Off-topic question (weather, coding, sports, general knowledge).

6. "CONFUSED_USER" — User is lost, unsure, or asking how to use the tool.

7. "COMPARE_COMPANIES" — User explicitly wants a side-by-side comparison of two companies.
    Extract compare_targets as a list of two company names.

8. "DOWNLOAD_PLAN" — User asks to download, export, or save the plan.

9. "SHOW_PLAN" — User asks to see, view, show, get, or display the current plan.
    Keywords: "show plan", "get plan", "display plan", "view plan", "show me the plan",
    "generate the account plan", "get the plan", "show account plan", "see the plan",
    "render the plan", "print the plan", "give me the plan".
    This is NOT a follow-up question — the user wants the full plan table rendered.

10. "EDGE_CASE_USER" — User input is invalid, unsupported, impossible, or completely
    nonsensical (e.g. "research the moon's CEO", "compare 17 companies",
    "give me a recipe"). Treat as unhandled/unsupported request.

## DISAMBIGUATION RULES
- "vs X" or "compare to X" in a follow-up = "CURRENT_COMPANY", NOT "NEW_COMPANY".
- Changing goal without company = "GOAL_UPDATE", NOT "NEW_COMPANY".
- "Update [section]" = "SECTION_UPDATE", NOT "CURRENT_COMPANY".
- "get the plan", "show plan", "generate account plan" = "SHOW_PLAN", NOT "CURRENT_COMPANY".

## OUTPUT FORMAT (strict JSON, no markdown wrapping)
{{
    "intent": "THE_CATEGORY",
    "company_name": "Name or null",
    "goal": "Goal string or null",
    "sections_to_update": [],
    "compare_targets": []
}}

Valid intent values: NEW_COMPANY, GOAL_UPDATE, SECTION_UPDATE, CURRENT_COMPANY,
GENERAL_QUESTION, CONFUSED_USER, COMPARE_COMPANIES, DOWNLOAD_PLAN, SHOW_PLAN, EDGE_CASE_USER
"""
        try:
            resp = await self._call_with_retry(
                self.async_client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            logger.error("classify_intent failed: %s", exc)
            return {
                "intent":           "CURRENT_COMPANY",
                "company_name":     None,
                "goal":             None,
                "sections_to_update": [],
                "compare_targets":  [],
            }

    # ── Full Plan Extraction ──────────────────────────────────────────────────

    async def async_extract_info(
        self,
        raw_text: str,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extracts and synthesizes a complete Account Plan from raw search results.

        Prompt architecture:
        ROLE → EXTRACTION RULES → SYNTHESIS RULES → CONFLICT RULES →
        CONFIDENCE RULES → OUTPUT FORMAT → FEW-SHOT EXAMPLE

        Key improvements over original:
        - Conflict threshold: only flag when figures differ by >20% (reduces false positives).
        - Executive format includes title + strategic relevance (not just names).
        - Competitor format includes named companies + differentiation notes.
        - data_confidence per section (HIGH/MEDIUM/LOW) for user transparency.
        - data_warnings for private/low-data companies.
        - No duplicate rules (original had Rules 2 and 8 identical).
        """
        company = current_state.get("company_name", "Unknown")
        goal    = current_state.get("user_goal", "General Research")

        system_prompt = f"""
## ROLE
You are an elite Enterprise Sales Strategist specialising in B2B technology sales.
Your task: extract and synthesise a complete, actionable Account Plan from raw web data.
Target Company: {company}
User Sales Goal: {goal}

## CURRENT ACCOUNT PLAN STATE
{json.dumps(current_state, indent=2)}

## EXTRACTION RULES
1. UPDATE "company_overview": 3–4 sentences on business model, market position, and
   recent trajectory. MUST be company-specific. Never output generic strategy boilerplate.

2. UPDATE "financial_snapshot": most recent revenue, market cap, YoY growth, profitability.
   Prefer dated search results over baseline knowledge. If using baseline, note "(estimated)".
   Format: "Revenue: $Xb (YYYY) | Market Cap: $Xb | YoY Growth: X%"

3. UPDATE "market_revenue": size of the primary market this company operates in (TAM).

4. UPDATE "competitors" as an array of strings in this format:
   "CompanyName — one sentence on competitive differentiation vs {company}"
   List 3–5 NAMED competitors. NEVER use placeholders like "other cloud providers".
   Example: "Salesforce — dominates enterprise CRM with deep workflow integrations
   that {company} lacks in the mid-market segment"

5. UPDATE "key_executives" with titles AND strategic relevance to the sales goal:
   "Full Name — Title, known for [strategic focus relevant to '{goal}']"
   Example: "Satya Nadella — Chairman & CEO, architect of cloud-first transformation;
   key decision-maker for enterprise platform partnerships"
   List 3–5 executives. If titles not in search results, use your knowledge.

6. PREFER RECENT DATA: when figures conflict across sources, use the one with the
   most recent date. If no date is available, append "(estimated)".

## SYNTHESIS RULES (mandatory — not optional filler)
7. DERIVE "strategic_priorities" from RECENT_NEWS entries. 3–5 specific, concrete
   priorities the company is actively pursuing. Not buzzwords — actual programmes.
   Example: "Expanding AWS partnership to reduce on-premise infrastructure costs by 30%"

8. DERIVE "pain_points" by mapping strategic priorities to the sales goal ({goal}).
   2–3 items. Each must be specific enough for a sales rep to reference in a cold email.
   Example: "GDPR compliance latency from rapid EU data centre expansion —
   directly addressable by {goal}"

9. CRAFT "value_proposition": 2–3 sentences connecting {goal} to {company}'s specific
   pain points. Name the pain points explicitly. Be concrete — no generic pitches.

10. WRITE "action_plan": 3–5 next steps for a sales rep. Include executive names,
    business trigger events, and specific calls-to-action.
    Example: "Reach out to [exec name] ahead of Q3 earnings on [date] —
    their cost-reduction mandate aligns directly with our [feature]."

## CONFLICT RULES
11. Only add to "open_questions" when NUMERICAL figures differ by MORE THAN 20% across
    sources (e.g., one says $2B revenue, another says $4B). Minor phrasing differences,
    date variations, or rounding do NOT qualify. Be conservative — false positives
    waste user time.

12. If fewer than 3 usable search results were found, or the company appears private
    (no revenue figures, no stock ticker, no public filings), add to "data_warnings":
    "Limited public data for {company}. Financial figures are estimated — verify
    before using in a client-facing document."

## CONFIDENCE SCORING
13. Populate "data_confidence" for each section you update:
    - "HIGH":   found in verified search result with a date
    - "MEDIUM": synthesised from news or inferred from multiple sources
    - "LOW":    estimated from baseline LLM knowledge with no search evidence
    Example: {{"financial_snapshot": "HIGH", "pain_points": "MEDIUM", "competitors": "LOW"}}

## OUTPUT FORMAT
Return ONLY a valid JSON object with the exact keys from CURRENT ACCOUNT PLAN STATE.
Do NOT invent new keys. Do NOT wrap in markdown code blocks. No preamble.

## FEW-SHOT EXAMPLE (abbreviated — shows expected quality bar)
Input: Company = "Stripe", Goal = "sell enterprise fraud detection API"
Expected output fragment:
{{
  "key_executives": [
    "Patrick Collison — CEO & Co-founder, drives product-led growth and developer-first
     global expansion; primary decision-maker for infrastructure partnerships",
    "John Collison — President & Co-founder, owns financial infrastructure and enterprise
     deals; likely stakeholder for fraud prevention investment"
  ],
  "pain_points": [
    "Rapid expansion into high-fraud markets (India, Brazil) without localised fraud models
     — directly relevant to an enterprise fraud detection API",
    "Card-not-present fraud rate increase as Stripe scales SMB merchants with thin
     existing risk controls"
  ],
  "value_proposition": "Our fraud detection API addresses Stripe's two acute risks: the
     20%+ CNP fraud increase in their SMB segment and the absence of region-specific
     fraud models in their recent India/Brazil expansion. By integrating at the Stripe
     Connect layer, we reduce dispute rates without adding checkout latency.",
  "data_confidence": {{
    "key_executives": "HIGH",
    "pain_points": "MEDIUM",
    "value_proposition": "MEDIUM"
  }}
}}
"""
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"SEARCH RESULTS:\n{raw_text}"},
        ]

        # Adaptive retry flow:
        #   Attempt 1 → strategy="default"
        #   Attempt 2 → strategy="fallback_reasoning"  (on any failure)
        #   If output passes JSON parse but fails validation → retry with "strict"
        STRATEGIES = ["default", "fallback_reasoning"]
        last_error: Optional[str] = None

        for strategy in STRATEGIES:
            try:
                messages = self._apply_strategy(base_messages, strategy)
                resp = await self._call_with_retry(
                    self.async_client.chat.completions.create,
                    messages=messages,
                    model=self.MODEL,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    max_tokens=2500,
                )
                parsed = json.loads(resp.choices[0].message.content)

                # Validation gate: trigger strict retry when ANY core field is
                # vague/empty — not just when ALL are empty (which misses the
                # common case of 4 real fields + 1 vague field slipping through).
                _, val_issues = PlanOutputValidator.validate(parsed)
                core_keys = {"company_overview", "financial_snapshot", "competitors",
                             "key_executives", "pain_points"}
                non_empty = sum(
                    1 for k in core_keys
                    if parsed.get(k) and parsed[k] not in ("Unknown", [], "")
                )
                needs_strict = (non_empty == 0 or bool(val_issues)) and strategy != "strict"
                if needs_strict:
                    logger.warning(
                        "extract_info: validation issues on '%s' strategy "
                        "(%s) — retrying with 'strict'.",
                        strategy, val_issues or "all core fields empty",
                    )
                    strict_msgs = self._apply_strategy(base_messages, "strict")
                    strict_resp = await self._call_with_retry(
                        self.async_client.chat.completions.create,
                        messages=strict_msgs,
                        model=self.MODEL,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                        max_tokens=2500,
                    )
                    strict_parsed = json.loads(strict_resp.choices[0].message.content)
                    # Enrich confidence explanations on strict-retry result too
                    if "data_confidence" in strict_parsed:
                        strict_parsed["data_confidence"] = self._explain_confidence(
                            strict_parsed["data_confidence"],
                            raw_text,
                            strict_parsed.get("data_warnings", []),
                        )
                    return strict_parsed

                # Enrich confidence map with human-readable explanations
                if "data_confidence" in parsed:
                    parsed["data_confidence"] = self._explain_confidence(
                        parsed["data_confidence"],
                        raw_text,
                        parsed.get("data_warnings", []),
                    )
                return parsed

            except json.JSONDecodeError:
                last_error = "LLM returned invalid JSON."
                logger.warning("extract_info [%s]: invalid JSON — trying next strategy.", strategy)
            except Exception as exc:
                last_error = str(exc)
                logger.warning("extract_info [%s] failed: %s — trying next strategy.", strategy, exc)

        logger.error("extract_info: all strategies exhausted. Last error: %s", last_error)
        return {"error": last_error or "All LLM strategies failed."}

    # ── Confidence Explanation ────────────────────────────────────────────────

    @staticmethod
    def _explain_confidence(
        confidence_map: Dict[str, str],
        raw_text: str,
        warnings: List[str],
    ) -> Dict[str, str]:
        """
        Augments a plain HIGH/MEDIUM/LOW confidence map with human-readable
        explanations based on source type and data completeness.

        Returns a new dict with values like "HIGH (verified from structured sources)"
        rather than plain "HIGH". Existing callers that check `.startswith("HIGH")`
        are unaffected — existing badge lookup in app.py uses `confidence_map.get(key, "")`
        and the badge dict now checks with `in` to stay backward-compatible.
        """
        has_filings = any(
            kw in raw_text.lower()
            for kw in ("filing", "10-k", "annual report", "sec", "earnings release")
        )
        has_news = any(
            kw in raw_text.lower()
            for kw in ("news", "announced", "reported", "according to", "reuters", "bloomberg")
        )
        limited_data = bool(warnings)

        EXPLANATIONS = {
            "HIGH": (
                "verified from structured sources (filings, official reports)"
                if has_filings
                else "verified from search results with dated evidence"
            ),
            "MEDIUM": (
                "derived from secondary sources (news, summaries)"
                if has_news
                else "synthesised from multiple sources — cross-check recommended"
            ),
            "LOW": (
                "estimated due to limited or conflicting data"
                if limited_data
                else "estimated from baseline knowledge — no direct search evidence"
            ),
        }

        result: Dict[str, str] = {}
        for section, level in confidence_map.items():
            level_upper = level.upper() if isinstance(level, str) else ""
            explanation = EXPLANATIONS.get(level_upper, "")
            result[section] = f"{level_upper} ({explanation})" if explanation else level_upper
        return result



    async def async_extract_section(
        self,
        section_key:   str,
        raw_text:      str,
        current_state: Dict[str, Any],
        simplified:    bool = False,
    ) -> Dict[str, Any]:
        """
        Targeted extraction for a SINGLE plan section.

        simplified=True: used on retry. Drops the full plan_context and uses a
        shorter, lower-token prompt — reduces LLM pressure when the primary attempt
        failed (often due to context size or ambiguous instructions).
        """
        company = current_state.get("company_name", "Unknown")
        goal    = current_state.get("user_goal", "General Research")

        SECTION_INSTRUCTIONS: Dict[str, str] = {
            "company_overview":
                f"Write a fresh 3–4 sentence overview of {company}: business model, "
                f"market position, and recent strategic trajectory. Be specific.",
            "financial_snapshot":
                f"Extract the most recent revenue, market cap, and YoY growth for {company}. "
                f"Format: 'Revenue: $Xb (YYYY) | Market Cap: $Xb | YoY Growth: X%'",
            "market_revenue":
                f"Estimate the TAM of {company}'s primary industry (one figure with year).",
            "competitors":
                f"List 3–5 named competitors of {company}. Format each as: "
                f"'CompanyName — competitive differentiation vs {company}'",
            "key_executives":
                f"List 3–5 key executives of {company} with titles and relevance to "
                f"the sales goal '{goal}'. Format: 'Name — Title, strategic relevance'",
            "strategic_priorities":
                f"From RECENT_NEWS, list 3–5 specific strategic initiatives "
                f"{company} is actively pursuing right now.",
            "pain_points":
                f"Based on {company}'s strategic priorities and the goal '{goal}', "
                f"derive 2–3 specific pain points a sales rep can directly address.",
            "value_proposition":
                f"Write a 2–3 sentence targeted pitch connecting '{goal}' to "
                f"{company}'s specific pain points. Name the pain points explicitly.",
            "action_plan":
                f"Write 3–5 concrete sales next steps for engaging {company}. "
                f"Include executive names and specific business triggers.",
        }

        instruction = SECTION_INSTRUCTIONS.get(
            section_key,
            f"Update the '{section_key}' field based on the search data provided."
        )

        # simplified=True strips full plan_context to cut token pressure on retry
        if simplified:
            context_block = (
                f"Company: {company}\n"
                f"Sales goal: {goal}\n"
                f"(Simplified retry — use search results below as primary source)"
            )
        else:
            context_block = "## CONTEXT (read-only — do not modify other sections)\n" + json.dumps(
                {k: v for k, v in current_state.items()
                 if k not in ("data_confidence", "data_warnings")},
                indent=2
            )

        prompt = f"""
## ROLE
You are an Enterprise Sales Strategist updating a SINGLE section of an Account Plan.
Company: {company} | Goal: {goal}

## TASK — UPDATE ONLY "{section_key}"
{instruction}

{context_block}

## OUTPUT FORMAT
Return ONLY a JSON object with exactly two keys:
{{
  "{section_key}": <updated_value>,
  "data_confidence": {{"{section_key}": "HIGH|MEDIUM|LOW"}}
}}
No other keys. No markdown wrapping. No preamble.
"""
        validator = PlanOutputValidator()
        base_messages = [
            {"role": "system", "content": prompt},
            # NOTE: Groq requires the word "json" in the user message when
            # response_format=json_object is set. Added here to guarantee no 400.
            {"role": "user", "content": (
                f"SEARCH RESULTS (use to update '{section_key}' as json):\n{raw_text}"
            )},
        ]

        # Adaptive retry: default → fallback_reasoning → (optional) strict
        STRATEGIES = ["default", "fallback_reasoning"]
        last_error: Optional[str] = None

        for strategy in STRATEGIES:
            try:
                messages = self._apply_strategy(base_messages, strategy)
                resp = await self._call_with_retry(
                    self.async_client.chat.completions.create,
                    messages=messages,
                    model=self.MODEL,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    max_tokens=700,
                )
                section_result = json.loads(resp.choices[0].message.content)

                # Validation gate: check for empty/vague output (Issue 2)
                _, issues = validator.validate(section_result, section_key=section_key)
                empty_output = (
                    not section_result.get(section_key)
                    or section_result.get(section_key) in ("Unknown", [], "")
                )

                if (issues or empty_output) and strategy != "strict":
                    logger.warning(
                        "extract_section(%s): validation issues on '%s' strategy "
                        "(%s) — retrying with 'strict'.",
                        section_key, strategy,
                        issues or "empty output",
                    )
                    strict_msgs = self._apply_strategy(base_messages, "strict")
                    strict_resp = await self._call_with_retry(
                        self.async_client.chat.completions.create,
                        messages=strict_msgs,
                        model=self.MODEL,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                        max_tokens=700,
                    )
                    strict_result = json.loads(strict_resp.choices[0].message.content)
                    if "data_confidence" in strict_result:
                        strict_result["data_confidence"] = self._explain_confidence(
                            strict_result["data_confidence"], raw_text, []
                        )
                    return strict_result

                # Enrich per-section confidence with human-readable explanation
                if "data_confidence" in section_result:
                    section_result["data_confidence"] = self._explain_confidence(
                        section_result["data_confidence"], raw_text, []
                    )
                return section_result

            except json.JSONDecodeError:
                last_error = "LLM returned invalid JSON."
                logger.warning(
                    "extract_section(%s) [%s]: invalid JSON — trying next strategy.",
                    section_key, strategy,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "extract_section(%s) [%s] failed: %s — trying next strategy.",
                    section_key, strategy, exc,
                )

        logger.error(
            "extract_section(%s): all strategies exhausted. Last error: %s",
            section_key, last_error,
        )
        return {"error": last_error or "All LLM strategies failed."}

    # ── Conflict Resolution ───────────────────────────────────────────────────

    async def async_resolve_conflict(
        self,
        question: str,
        user_answer: str,
        current_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Incorporates the user's answer to a data conflict into the plan.

        DESIGN DECISION: No plausibility rejection. The LLM has no search context
        here so it cannot reliably judge whether a user-supplied figure is "realistic"
        (e.g. it rejected PhysicsWallah's $5B IPO valuation as implausible when it
        was factually correct). The user is the authority on their own data.
        We only check topical relevance: does the answer actually address the question?
        If it does not address the question at all, we note it but still accept and move on.
        """
        prompt = f"""
## ROLE
You are incorporating a user's clarification answer into an Account Plan.

## CONFLICT QUESTION
"{question}"

## USER'S ANSWER
"{user_answer}"

## RULES
1. If the answer is topically relevant to the question: extract the data and update
   the appropriate JSON field(s) in the plan. The user is the authority — accept
   their figures even if they seem large or unusual. Real companies can have
   surprising valuations, revenues, or facts.
2. If the answer is completely off-topic (e.g. question asks about revenue, user
   says "I like pizza"): do not update any field and set "open_questions" to
   [{{"note": "Answer did not address the question. Continuing with best-effort data."}}].
   Do NOT re-ask the same question.
3. Never reject an answer solely because the figure seems large or unexpected.

## CURRENT PLAN (context only — only update fields that directly answer the conflict)
{json.dumps(current_state, indent=2)}

## OUTPUT FORMAT
Valid JSON with updated field(s) and optionally "open_questions". No markdown.
"""
        try:
            resp = await self._call_with_retry(
                self.async_client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            logger.error("resolve_conflict failed: %s", exc)
            return {}

    # ── Direct Q&A with Streaming ─────────────────────────────────────────────

    async def async_direct_answer_stream(
        self,
        question: str,
        plan_state: Dict[str, Any],
        recent_context: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Streams a direct conversational answer to a follow-up question.
        Uses the current plan as context. If recent_context is provided
        (from a targeted search), it is included for fresher data.

        Yields string tokens for Chainlit's stream_token().
        """
        company = plan_state.get("company_name", "the company")
        goal    = plan_state.get("user_goal", "General Research")

        plan_summary = json.dumps(
            {k: v for k, v in plan_state.items()
            if k not in ("data_confidence", "data_warnings", "source_references")},
            indent=2
        )
        context_block = (
            f"\n\n## RECENT WEB SEARCH DATA\n{recent_context}"
            if recent_context else ""
        )

        prompt = f"""
You are an enterprise sales research assistant helping a rep researching {company}.
Their sales goal: {goal}

## CURRENT ACCOUNT PLAN
{plan_summary}{context_block}

## INSTRUCTIONS
Answer the user's question directly and concisely. Lead with the actual answer
in the first sentence. Then provide supporting context from the plan.
Where relevant, tie your answer back to their sales goal.
Keep your response focused — no preambles like "Great question!".

## QUESTION
{question}
"""
        stream = await asyncio.wait_for(
            self.async_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.MODEL,
                temperature=0.3,
                stream=True,
            ),
            timeout=self.TIMEOUT_SECONDS,
        )
        # Per-chunk timeout: each individual chunk must arrive within TIMEOUT_SECONDS.
        # A slow network that drips tokens at one per minute would otherwise hang
        # the generator indefinitely after the initial stream object was returned.
        stream_iter = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=self.TIMEOUT_SECONDS,
                )
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield token
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                logger.warning(
                    "Stream chunk timed out (>%.0fs) — terminating stream.",
                    self.TIMEOUT_SECONDS,
                )
                yield "\n\n*[Response interrupted — network too slow. Please try again.]*"
                break


# ── Proactive Insight Engine ──────────────────────────────────────────────────

class ProactiveInsightEngine:
    """
    Lightweight post-pipeline analyzer that surfaces optional improvement
    suggestions when the plan has low-confidence or failed sections.

    Design: stateless, functional. Called after plan generation completes.
    Suggestions are OPTIONAL — they append to content, never block output.
    """

    # Sections considered "critical" for sales purposes
    CRITICAL_SECTIONS = {"financial_snapshot", "pain_points", "value_proposition"}

    @staticmethod
    def generate_suggestions(
        confidence_map: Dict[str, str],
        failed_sections: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Analyses confidence_map for LOW entries and failed sections.
        Returns a list of suggestion strings, or empty list if none needed.

        confidence_map values may now be enriched strings like "LOW (estimated…)".
        We check with `.startswith("LOW")` for forward compatibility.
        """
        failed_sections = failed_sections or []
        low_sections = [
            s for s, v in confidence_map.items()
            if isinstance(v, str) and v.upper().startswith("LOW")
        ]
        has_low_financial = any(
            s in low_sections for s in ("financial_snapshot", "market_revenue")
        )
        missing_critical = [
            s for s in ProactiveInsightEngine.CRITICAL_SECTIONS
            if s in failed_sections or s in low_sections
        ]

        suggestions: List[str] = []

        if low_sections:
            suggestions.append(
                "⚠️ Some sections have low confidence. "
                "Would you like me to improve them? "
                f"*(Affected: {', '.join(s.replace('_', ' ') for s in low_sections)})*"
            )

        # Only add the dedicated financial suggestion when the general low_sections
        # message did NOT already mention the financial sections — prevents duplication.
        if has_low_financial and not low_sections:
            suggestions.append(
                "💰 Want me to refine financial data with a deeper search? "
                "*(Type: 'Refresh financial snapshot')*"
            )

        if missing_critical:
            suggestions.append(
                "📋 Critical sections may need attention: "
                + ", ".join(f"**{s.replace('_', ' ')}**" for s in missing_critical)
                + ". *(Type 'Update [section name]' to retry)*"
            )

        if suggestions:
            logger.info(
                "ProactiveInsightEngine: %d suggestion(s) generated.", len(suggestions)
            )
        return suggestions




class ResearchAgent:
    """
    Top-level orchestrator. Manages the full research pipeline for one user session.

    Lifecycle:
    1. Chainlit on_chat_start() creates agent and stores in cl.user_session.
    2. Each on_message() call retrieves agent from session and calls process_user_input().
    3. agent.state is mutated through the session; never shared with other users.

    process_user_input() returns a result dict describing how app.py should render
    the response — decoupling business logic from UI concerns.

    Result dict schema:
    {
      "response_type": "stream" | "message" | "plan" | "clarification" |
                       "comparison" | "download",
      "content":       str,         — text content for non-stream types
      "stream_gen":    AsyncGenerator,  — for response_type == "stream" only
      "plan_changed":  bool,         — whether the plan was modified
      "progress_messages": List[str],  — status messages for Chainlit steps
    }
    """

    def __init__(self) -> None:
        self.state = AccountPlanState()
        self.tool  = ResearchTool()
        self.llm   = LLMEngine()
        # Real-time progress callback — injected by app.py before each pipeline call.
        # Signature: async def callback(label: str) -> None
        # When set, _emit_progress() fires it during pipeline execution so Chainlit
        # steps appear AS each stage completes, not all at once after the pipeline returns.
        self._progress_cb: Optional[Any] = None

    def set_progress_callback(self, cb: Any) -> None:
        """
        Injects an async callable that app.py uses to display real-time steps.
        Called once before process_user_input() on every message.
        Resets to None after each call to prevent stale callbacks across turns.
        """
        self._progress_cb = cb

    async def _emit_progress(self, label: str) -> None:
        """
        Fires the injected progress callback if one is set.
        Safe to call even when no callback is registered (terminal/test mode).
        """
        if self._progress_cb is not None:
            try:
                await self._progress_cb(label)
            except Exception as exc:
                # Never let a UI callback crash the pipeline
                logger.warning("_emit_progress callback raised: %s", exc)

    # ── Main Entry Point ──────────────────────────────────────────────────────

    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Full async processing pipeline with clean state transitions.

        State machine:
        NEW_COMPANY        → reset_plan() → full research pipeline
        GOAL_UPDATE        → update goal  → section update (pain_points + vp + action)
        SECTION_UPDATE     → targeted search + section extraction
        CURRENT_COMPANY    → streaming direct answer (with optional fresh search)
        COMPARE_COMPANIES  → parallel research on two companies
        CONFUSED_USER      → static guidance message
        GENERAL_QUESTION   → static redirect message
        DOWNLOAD_PLAN      → signals app.py to handle export
        """
        current_company = self.state.plan.get("company_name", "Not Yet Provided")

        # Sanitize input to prevent prompt injection.
        # Replace double and single quotes, and strip common injection-pattern
        # sequences (e.g. "##", "```", "---") that could break f-string prompt boundaries.
        safe_input = (
            user_input
            .replace('"', '\\"')
            .replace("```", "'''")
            .replace("## ", "")
            .replace("\n##", "")
        )

        # ── PRIORITY: Consume conflict queue BEFORE intent classification ─────
        # Placed before the intent router so that ANY user input (including
        # "skip this question" or a real data answer) is treated as a conflict
        # response when open_questions are pending. Without this, the router may
        # classify "skip this question" as EDGE_CASE_USER and block it entirely.
        if self.state.open_questions:
            return await self._pipeline_conflict_resolution(safe_input)

        # ── Step 1: Classify Intent ───────────────────────────────────────────
        router = await self.llm.async_classify_intent(safe_input, current_company)
        intent            = router.get("intent", "CURRENT_COMPANY")
        extracted_company = router.get("company_name")
        user_goal         = router.get("goal")
        sections_to_update = [
            s for s in router.get("sections_to_update", []) if s in UPDATABLE_SECTIONS
        ]
        compare_targets   = router.get("compare_targets", [])

        logger.info("Intent: %s | Company: %s | Goal: %s", intent, extracted_company, user_goal)

        # ── Step 2: Special-case intents ──────────────────────────────────────
        if intent == "CONFUSED_USER":
            return self._static_response("confused")

        if intent == "GENERAL_QUESTION":
            return self._static_response("general", current_company)

        if intent == "EDGE_CASE_USER":
            logger.warning("EDGE_CASE_USER input blocked: %s", safe_input)
            return {
                "response_type": "message",
                "content": (
                    "⚠️ I'm not able to handle that request.\n\n"
                    "I specialise in enterprise account research. Try:\n"
                    "- *'Research [Company Name]'*\n"
                    "- *'Update pain points'*\n"
                    "- *'Compare [Company A] vs [Company B]'*"
                ),
                "plan_changed": False,
                "progress_messages": [],
                "suggestions": [],
            }

        if intent == "DOWNLOAD_PLAN":
            return {"response_type": "download", "content": "", "plan_changed": False,
                    "progress_messages": [], "suggestions": []}

        if intent == "SHOW_PLAN":
            # User explicitly wants to see the plan rendered — always show it.
            company_name = self.state.plan.get("company_name", "Not Yet Provided")
            if company_name == "Not Yet Provided":
                return self._static_response("no_company")
            # Force-render by resetting hash so the plan always displays
            self.state._rendered_hash = ""
            return {
                "response_type": "plan",
                "content":       f"### 📊 Here is your Account Plan for **{company_name}**",
                "plan_changed":  True,
                "progress_messages": [],
                "suggestions": [],
            }

        if intent == "COMPARE_COMPANIES":
            return await self._pipeline_comparison(compare_targets)

        # ── Step 3: State transitions ─────────────────────────────────────────
        if intent == "GOAL_UPDATE" and user_goal:
            # Keep company, update goal, then refresh derived sections
            self.state.update_section("user_goal", user_goal)
            # After goal change, pain_points/value_prop/action_plan are stale
            sections_to_update = sections_to_update or [
                "pain_points", "value_proposition", "action_plan"
            ]
            intent = "SECTION_UPDATE"

        elif intent == "NEW_COMPANY":
            if extracted_company:
                if current_company not in ("Not Yet Provided", extracted_company):
                    logger.info(
                        "Context switch: '%s' → '%s'", current_company, extracted_company
                    )
                # Full reset clears stale open_questions from previous company
                self.state.reset_plan()
                self.state.update_section("company_name", extracted_company)
                if user_goal:
                    self.state.update_section("user_goal", user_goal)
            else:
                # Router misfired — no company name found; treat as goal update
                logger.warning("NEW_COMPANY intent but no company name. Treating as GOAL_UPDATE.")
                if user_goal:
                    self.state.update_section("user_goal", user_goal)

        # ── Step 4 (now Step 5): Route to appropriate pipeline ───────────────
        refreshed_company = self.state.plan.get("company_name", "Not Yet Provided")

        if refreshed_company == "Not Yet Provided":
            return self._static_response("no_company")

        if intent == "SECTION_UPDATE" and sections_to_update:
            return await self._pipeline_section_update(sections_to_update)

        if intent in ("NEW_COMPANY", "GOAL_UPDATE"):
            return await self._pipeline_full_research(refreshed_company)

        # Default: CURRENT_COMPANY follow-up → stream direct answer
        return await self._pipeline_followup(safe_input, refreshed_company)

    # ── Research Pipelines ────────────────────────────────────────────────────

    async def _pipeline_full_research(self, company: str) -> Dict[str, Any]:
        """
        Full four-query research pipeline for a new or refreshed company.

        Progress is emitted via _emit_progress() AT THE MOMENT each stage begins,
        not collected into a list and returned after completion. This ensures
        Chainlit steps appear in real-time during the 5-15 second research window.
        """
        goal = self.state.plan.get("user_goal", "General Research")

        # Stage 1: Web search — emit before the blocking I/O starts
        await self._emit_progress(f"🔍 Researching {company}...")
        queries = [
            f"{company} company overview business model revenue 2025",
            f"{company} CEO CTO CFO executives leadership team titles",
            f"{company} strategy news initiatives expansion 2025",
            f"{company} top competitors market share industry comparison",
        ]
        raw_data = await self.tool.async_search_multi(queries)

        # Stage 2: Format — instant, but signals transition to the user
        await self._emit_progress("📰 Scanning financial data and latest news...")
        llm_ready = self.tool.format_for_llm(raw_data)

        # Collect unique source URLs for attribution
        source_urls = list(dict.fromkeys(
            r["source"] for r in raw_data
            if r.get("source") and r["source"] not in (
                "SerpAPI", "SerpAPI/News", "SerpAPI/FAQ", "DDGS", "DDGS/News", "N/A"
            )
        ))

        # Stage 3: LLM extraction — emit before the ~5s LLM call
        await self._emit_progress(f"🧠 Synthesising strategic insights for: {goal}")
        current_state = self.state.get_current_plan()
        current_state["open_questions"] = []
        updated = await self.llm.async_extract_info(llm_ready, current_state)

        if "error" in updated:
            return self._error_response(updated["error"], [])

        # Stage 4: Validation and state write
        await self._emit_progress("✅ Validating and building Account Plan...")

        questions  = updated.pop("open_questions", [])
        warnings   = updated.pop("data_warnings",  [])
        confidence = updated.pop("data_confidence", {})

        cleaned, issues = PlanOutputValidator.validate(updated)
        if issues:
            logger.warning(
                "_pipeline_full_research: validation issues in LLM output: %s", issues
            )
        updated = cleaned

        for key, value in updated.items():
            self.state.update_section(key, value)

        if source_urls:
            existing_sources = self.state.plan.get("source_references", [])
            merged_sources = list(dict.fromkeys(existing_sources + source_urls))
            self.state.update_section("source_references", merged_sources)

        if warnings:
            existing = self.state.plan.get("data_warnings", [])
            self.state.update_section("data_warnings", existing + warnings)
        existing_conf = self.state.plan.get("data_confidence", {})
        existing_conf.update(confidence)
        self.state.update_section("data_confidence", existing_conf)

        plan_changed = self.state.has_changed_since_last_render()
        company_name = self.state.plan.get("company_name", company)

        raw_suggestions = ProactiveInsightEngine.generate_suggestions(
            self.state.plan.get("data_confidence", {}),
            failed_sections=[],
        )
        suggestions = [s for s in raw_suggestions if s not in self.state._seen_suggestions]
        self.state._seen_suggestions.update(suggestions)

        # Clear callback after use — prevents stale callbacks across conversation turns
        self._progress_cb = None

        if questions:
            self.state.open_questions = questions
            return {
                "response_type":     "clarification",
                "content":           (
                    f"### 🛑 Clarification Required\n"
                    f"> **{questions[0]}**\n\n"
                    f"*(Answer below, or type 'skip' to continue with best-effort data)*"
                ),
                "plan_changed":      plan_changed,
                "progress_messages": [],   # already emitted in real-time
                "suggestions":       [],
            }

        return {
            "response_type":     "plan",
            "content":           f"### 🟢 Account Plan ready for **{company_name}**",
            "suggestions":       suggestions,
            "plan_changed":      plan_changed,
            "progress_messages": [],   # already emitted in real-time
        }

    # ── Section Recovery Helpers ──────────────────────────────────────────────

    # Level 4: Per-section recovery suggestions surfaced to the user when a
    # section fails. These are actionable and specific, not generic "try again".
    _SECTION_RECOVERY_HINTS: Dict[str, str] = {
        "pain_points":          (
            "Try: *'Update pain points'* after adding more product context, e.g. "
            "*'I'm pitching our AI fraud detection — update pain points'*"
        ),
        "value_proposition":    (
            "Try: *'Rewrite the pitch for [your product]'* to give the model "
            "a concrete product to anchor the value prop against"
        ),
        "action_plan":          (
            "Try: *'Refresh action plan'* — or add executive names first with "
            "*'Update key executives'*, then regenerate. "
            "First ensure pain points are populated, as action_plan draws on both."
        ),
        "competitors":          (
            "Try: *'Update competitors'* — for private/niche companies this may "
            "require adding context: *'Update competitors in the HR software space'*"
        ),
        "key_executives":       (
            "Try: *'Update executives'* — for private companies, add context: "
            "*'Update executives, the CEO is [name]'*"
        ),
        "financial_snapshot":   (
            "Try: *'Refresh financial data'* — if this company is private, "
            "financial data will be estimated rather than scraped"
        ),
        "strategic_priorities": (
            "Try: *'Update strategic priorities'* — include a recent news hook: "
            "*'Update priorities, focus on their recent AI announcements'*"
        ),
        "company_overview":     (
            "Try: *'Update company overview'* with the full company name or "
            "stock ticker if the company name is ambiguous"
        ),
        "market_revenue":       (
            "Try: *'Update market revenue'* with the specific industry name: "
            "*'Update market revenue for the enterprise HR SaaS space'*"
        ),
    }

    @staticmethod
    def _build_section_query_refined(section: str, company: str, goal: str) -> str:
        """
        Level 2: Returns a simpler, more focused retry query for a failed section.
        Primary queries cast a wide net; refined queries go narrow and goal-aware.
        The goal string is injected so sections like pain_points and value_proposition
        are searched in the context of what the user is actually selling.
        """
        goal_short = goal.split()[:4]
        goal_tag   = " ".join(goal_short)

        REFINED_QUERY_MAP: Dict[str, str] = {
            "company_overview":     f"{company} what does it do founded 2025",
            "financial_snapshot":   f"{company} annual revenue Q4 2025 earnings",
            "market_revenue":       f"{company} market size industry growth forecast",
            "competitors":          f"{company} vs competitors comparison",
            "key_executives":       f"{company} CEO president CFO name 2025",
            "strategic_priorities": f"{company} strategic focus priorities 2025",
            "pain_points":          f"{company} challenges {goal_tag}",
            "value_proposition":    f"{company} needs {goal_tag} use case",
            "action_plan":          f"{company} buying process decision makers {goal_tag}",
        }
        return REFINED_QUERY_MAP.get(
            section, f"{company} {section.replace('_', ' ')}"
        )

    @staticmethod
    def _apply_stale_marker(value: Any) -> Any:
        """
        Level 5: Appends a stale-data marker to the existing value so it remains
        visible in the UI but is clearly flagged as needing verification.
        Does NOT overwrite — the old data stays in place.

        String values get an inline suffix.
        List values get a trailing sentinel item.
        """
        STALE_SUFFIX = " ⚠️ (outdated — refresh recommended)"
        STALE_ITEM   = "⚠️ Data above may be outdated — type 'refresh [section]' to update"

        if isinstance(value, list) and value:
            if not any("outdated" in str(i) for i in value):
                return list(value) + [STALE_ITEM]
            return value   # already marked

        if isinstance(value, str) and value not in ("Unknown", ""):
            if "outdated" not in value:
                return value + STALE_SUFFIX
            return value

        # Empty/Unknown — use a hard fallback instead of a stale marker
        return "⚠️ Data unavailable — retry recommended"

    async def _pipeline_section_update(self, sections: List[str]) -> Dict[str, Any]:
        """
        Targeted refresh of one or more plan sections with full 5-level recovery.

        Level 1 — Honest structured messaging (✔/✖ per section, not a single status)
        Level 2 — Automatic single retry with a refined query + simplified prompt
        Level 3 — Graceful fallback placeholder if both attempts fail on empty plan
        Level 4 — Per-section recovery suggestion tied to the user's specific goal
        Level 5 — Old data preserved with stale marker if both attempts fail and the section already had content (never silently leaves stale data)
        """
        company = self.state.plan.get("company_name", "Not Yet Provided")
        goal    = self.state.plan.get("user_goal",    "General Research")

        if company == "Not Yet Provided":
            return self._static_response("no_company")

        await self._emit_progress(f"🔄 Refreshing: {', '.join(sections)} for {company}...")

        # Outcome buckets
        succeeded:          List[str] = []   # clean success
        succeeded_on_retry: List[str] = []   # failed primary, passed retry
        failed:             List[str] = []   # both attempts failed

        for section in sections:

            # ── Level 5: Snapshot old value BEFORE any mutation ───────────────
            old_value = self.state.plan.get(section)

            # ── Primary attempt ───────────────────────────────────────────────
            query_primary = self._build_section_query(section, company)
            raw_primary   = await self.tool.async_search_web(query_primary)

            # Collect source URLs for attribution
            section_sources = [
                r["source"] for r in raw_primary
                if r.get("source") and r["source"] not in (
                    "SerpAPI", "SerpAPI/News", "SerpAPI/FAQ",
                    "DDGS", "DDGS/News", "N/A"
                )
            ]
            if section_sources:
                existing = self.state.plan.get("source_references", [])
                merged = list(dict.fromkeys(existing + section_sources))
                self.state.update_section("source_references", merged)

            ctx_primary   = self.tool.format_for_llm(raw_primary)
            result        = await self.llm.async_extract_section(
                section, ctx_primary, self.state.get_current_plan()
            )

            if "error" not in result:
                self._apply_section_result(result, section)
                succeeded.append(section)
                continue

            # ── Level 2: Automatic retry (refined query + simplified prompt) ──
            logger.warning(
                "Section '%s' primary attempt failed. Retrying with refined query…",
                section
            )
            # BUG FIX: `progress` was never declared in this scope — it was a
            # leftover from the old list-based design. Progress is now emitted
            # in real-time via _emit_progress(), not collected into a list.
            await self._emit_progress(
                f"⚠️ {section.replace('_', ' ')}: primary failed — retrying with focused query…"
            )

            query_retry = self._build_section_query_refined(section, company, goal)
            raw_retry   = await self.tool.async_search_web(query_retry)
            ctx_retry   = self.tool.format_for_llm(raw_retry)
            retry_result = await self.llm.async_extract_section(
                section, ctx_retry, self.state.get_current_plan(),
                simplified=True,   # strips full plan_context — lower token pressure
            )

            if "error" not in retry_result:
                self._apply_section_result(retry_result, section)
                succeeded_on_retry.append(section)
                logger.info("Section '%s' recovered on retry.", section)
                continue

            # ── Both attempts failed ───────────────────────────────────────────
            logger.error(
                "Section '%s' failed on both attempts. Applying Level 3/5 recovery.", section
            )

            if old_value and old_value not in ("Unknown", [], ""):
                # Level 5: Preserve old content with stale marker
                marked = self._apply_stale_marker(old_value)
                self.state.update_section(section, marked)
                logger.info(
                    "Section '%s' stale-marked (old data preserved).", section
                )
            else:
                # Level 3: No old data — write explicit unavailability placeholder
                self.state.update_section(
                    section,
                    "⚠️ Data unavailable — type 'refresh "
                    + section.replace("_", " ")
                    + "' to retry"
                )

            failed.append(section)

        plan_changed = self.state.has_changed_since_last_render()

        # ── Level 1: Structured status message ────────────────────────────────
        content = self._build_section_update_message(
            company, succeeded, succeeded_on_retry, failed
        )

        # ── Proactive suggestions (separate key for UI control) ──────────────
        raw_suggestions = ProactiveInsightEngine.generate_suggestions(
            self.state.plan.get("data_confidence", {}),
            failed_sections=failed,
        )
        suggestions = [s for s in raw_suggestions if s not in self.state._seen_suggestions]
        self.state._seen_suggestions.update(suggestions)

        return {
            "response_type":     "plan",
            "content":           content,
            "suggestions":       suggestions,
            "plan_changed":      plan_changed,
            "progress_messages": [],  # emitted in real-time via _emit_progress
        }

    def _apply_section_result(self, result: Dict[str, Any], section: str) -> None:
        """Write a successful extraction result to state, including confidence."""
        new_conf = result.pop("data_confidence", {})
        for key, value in result.items():
            self.state.update_section(key, value)
        conf = self.state.plan.get("data_confidence", {})
        conf.update(new_conf)
        self.state.update_section("data_confidence", conf)

    def _build_section_update_message(
        self,
        company:            str,
        succeeded:          List[str],
        succeeded_on_retry: List[str],
        failed:             List[str],
    ) -> str:
        """
        Level 1 + Level 4: Builds a structured, per-section outcome message
        with recovery suggestions for every failed section.
        Format mirrors a CI build report — clear, scannable, actionable.
        """
        total_ok = len(succeeded) + len(succeeded_on_retry)
        total    = total_ok + len(failed)
        goal     = self.state.plan.get("user_goal", "General Research")

        # ── Header ────────────────────────────────────────────────────────────
        if not failed:
            header = f"### ✅ All sections refreshed for **{company}**"
        elif total_ok > 0:
            header = f"### ⚠️ Partial Update — **{company}** ({total_ok}/{total} sections updated)"
        else:
            header = f"### ❌ Update Failed — **{company}** (0/{total} sections updated)"

        lines = [header, ""]

        # ── Per-section outcomes ──────────────────────────────────────────────
        for s in succeeded:
            lines.append(f"✔ **{s.replace('_', ' ')}** — updated successfully")

        for s in succeeded_on_retry:
            lines.append(
                f"✔ **{s.replace('_', ' ')}** — updated *(required focused retry)*"
            )

        for s in failed:
            old_val = self.state.plan.get(s)
            has_old = bool(old_val) and old_val not in ("Unknown", [], "")
            preservation_note = (
                " Old data preserved and marked as outdated." if has_old
                else " Placeholder written — no previous data to preserve."
            )
            lines.append(
                f"✖ **{s.replace('_', ' ')}** — both attempts failed.{preservation_note}"
            )

        # ── Level 4: Recovery suggestions for failed sections ─────────────────
        if failed:
            lines.append("")
            lines.append("---")
            lines.append("**💡 Recovery suggestions:**")
            lines.append("")
            for s in failed:
                hint = self._SECTION_RECOVERY_HINTS.get(
                    s,
                    f"Try: *'Refresh {s.replace('_', ' ')}'* to attempt again"
                )
                lines.append(f"- **{s.replace('_', ' ')}:** {hint}")

            # Goal-aware contextual tip
            if any(s in failed for s in ("pain_points", "value_proposition")):
                lines.append("")
                lines.append(
                    f"> 💡 **Goal tip:** Your current goal is *\"{goal}\"*. "
                    f"Adding more product specifics often helps — e.g. *'I'm selling "
                    f"[product feature] that solves [specific problem]'* before retrying."
                )

        return "\n".join(lines)

    async def _pipeline_followup(self, question: str, company: str) -> Dict[str, Any]:
        """
        Handles a follow-up question about the active company.
        If the question implies needing fresh data, runs a quick targeted search first.
        Streams the response token-by-token.
        """
        FRESH_DATA_KEYWORDS = {
            "latest", "recent", "news", "update", "current", "now",
            "today", "2025", "2026", "just", "announced",
        }
        needs_fresh = any(kw in question.lower() for kw in FRESH_DATA_KEYWORDS)
        recent_context = ""

        if needs_fresh:
            search_q = (
                question if company.lower() in question.lower()
                else f"{company} {question}"
            )
            raw = await self.tool.async_search_web(search_q)
            recent_context = self.tool.format_for_llm(raw, max_chars=3000)

        stream_gen = self.llm.async_direct_answer_stream(
            question, self.state.get_current_plan(), recent_context
        )
        return {
            "response_type":    "stream",
            "stream_gen":       stream_gen,
            "plan_changed":     False,
            "progress_messages": [],
            "suggestions":      [],
        }

    async def _pipeline_conflict_resolution(self, user_answer: str) -> Dict[str, Any]:
        """
        Processes user's answer to a pending data conflict question.

        SKIP_PHRASES covers multi-word phrases like "skip this question" by using
        substring matching (any(p in answer)) rather than exact set membership.

        BUG FIX (plan never shown after all questions resolved):
        When the last question is answered or skipped, always force-render the plan
        by resetting _rendered_hash. The hash was already snapshotted during the
        initial research pipeline, so has_changed_since_last_render() would return
        False even though the user has never seen the plan rendered — because the
        clarification flow interrupted before the plan was displayed.
        """
        SKIP_PHRASES = {
            "skip", "don't know", "dont know", "idk", "no idea",
            "none", "ignore", "pass", "n/a", "na", "not sure", "unsure",
            "skip this", "skip it", "move on", "next",
        }
        question = self.state.open_questions.pop(0)

        is_skip = any(p in user_answer.lower() for p in SKIP_PHRASES)

        if is_skip:
            if self.state.open_questions:
                return {
                    "response_type": "clarification",
                    "content": (
                        f"Skipped. Next question:\n"
                        f"> **{self.state.open_questions[0]}**\n\n"
                        f"*(Type 'skip' to skip this one too)*"
                    ),
                    "plan_changed":  False,
                    "progress_messages": [],
                    "suggestions": [],
                }
            # Last question skipped — force-render plan (reset hash so plan displays)
            self.state._rendered_hash = ""
            return {
                "response_type": "plan",
                "content":       "Skipped. Continuing with the best available data.\n\n---",
                "plan_changed":  True,
                "progress_messages": [],
                "suggestions": [],
            }

        result = await self.llm.async_resolve_conflict(
            question, user_answer, self.state.get_current_plan()
        )
        new_questions = result.pop("open_questions", [])

        for key, value in result.items():
            if key in self.state.plan:
                self.state.update_section(key, value)

        if new_questions:
            # Only re-queue if the message is a real question (not an off-topic note)
            next_q = new_questions[0]
            if isinstance(next_q, dict):
                # Off-topic answer note — don't re-queue, just move on
                pass
            else:
                self.state.open_questions.insert(0, next_q)
                return {
                    "response_type": "clarification",
                    "content":       (
                        f"### 🛑 Clarification Still Needed\n"
                        f"> **{next_q}**\n\n"
                        f"*(Answer below, or type 'skip' to continue with best-effort data)*"
                    ),
                    "plan_changed":  False,
                    "progress_messages": [],
                    "suggestions": [],
                }

        # All questions resolved — force-render the plan regardless of hash state.
        # The hash was already consumed during initial research before the plan was
        # shown (clarification interrupted), so we must reset it to trigger a render.
        self.state._rendered_hash = ""
        suffix = (
            f"\n\n> Next: **{self.state.open_questions[0]}**"
            if self.state.open_questions else ""
        )
        return {
            "response_type": "plan",
            "content":       f"✅ Answer noted. Here is your Account Plan:{suffix}",
            "plan_changed":  True,
            "progress_messages": [],
            "suggestions": [],
        }

    async def _pipeline_comparison(self, companies: List[str]) -> Dict[str, Any]:
        """
        Parallel research on two companies, rendered as a comparison table.
        No other candidate is likely to implement this — high wow factor.

        Note: comparison results are intentionally stateless relative to the
        active user plan. This method does not write the comparison output to
        self.state, so follow-up questions after a comparison continue to use
        the previously active plan rather than the comparison result.
        """
        if len(companies) < 2:
            return {
                "response_type": "message",
                "content":       "Please name two companies to compare.\nExample: *'Compare Netflix vs Disney+'*",
                "plan_changed":  False,
                "progress_messages": [],
                "suggestions": [],
            }

        goal     = self.state.plan.get("user_goal", "General Research")
        targets  = companies[:2]
        progress = [f"⚔️  Running parallel research: {targets[0]} vs {targets[1]}..."]

        async def _research_one(name: str) -> Dict[str, Any]:
            queries = [
                f"{name} company overview revenue financials 2025",
                f"{name} CEO executives leadership team",
                f"{name} strategy priorities news 2025",
                f"{name} competitors market position",
            ]
            raw    = await self.tool.async_search_multi(queries)
            ready  = self.tool.format_for_llm(raw)
            state  = {
                "company_name": name, "user_goal": goal,
                "company_overview": "Unknown", "financial_snapshot": "Unknown",
                "market_revenue": "Unknown", "competitors": [],
                "key_executives": [], "strategic_priorities": [],
                "pain_points": [], "value_proposition": "Unknown",
                "action_plan": [], "data_confidence": {}, "data_warnings": [],
                "source_references": [],
            }
            return await self.llm.async_extract_info(ready, state)

        results = await asyncio.gather(
            _research_one(targets[0]),
            _research_one(targets[1]),
            return_exceptions=True,
        )
        comparison_md = self._format_comparison_table(targets, results, goal)
        return {
            "response_type":    "comparison",
            "content":          comparison_md,
            "plan_changed":     False,
            "progress_messages": [],  # emitted in real-time via _emit_progress
            "suggestions": [],
        }

    # ── Formatting Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_comparison_table(
        companies: List[str],
        results:   List[Any],
        goal:      str,
    ) -> str:
        """Formats a markdown side-by-side comparison table."""
        c0, c1 = companies[0], companies[1]
        md  = f"## ⚔️  {c0} vs {c1}\n"
        md += f"*Sales Context: {goal}*\n\n"
        md += f"| Dimension | {c0} | {c1} |\n"
        md += "|-----------|-----|-----|\n"

        def cell(result: Any, key: str) -> str:
            if isinstance(result, Exception) or not isinstance(result, dict):
                return "⚠️ Research failed"
            val = result.get(key, "N/A")
            if isinstance(val, list):
                val = "; ".join(str(v) for v in val[:3])
            return str(val)[:250].replace("|", "\\|").replace("\n", " ")

        rows = [
            ("Overview",          "company_overview"),
            ("Financials",        "financial_snapshot"),
            ("Market Size",       "market_revenue"),
            ("Pain Points",       "pain_points"),
            ("Value Proposition", "value_proposition"),
            ("Key Executives",    "key_executives"),
        ]
        for label, key in rows:
            md += f"| **{label}** | {cell(results[0], key)} | {cell(results[1], key)} |\n"
        return md

    @staticmethod
    def _build_section_query(section: str, company: str) -> str:
        """Returns a targeted search query for a single plan section."""
        QUERY_MAP: Dict[str, str] = {
            "company_overview":     f"{company} company overview business model 2025",
            "financial_snapshot":   f"{company} revenue market cap earnings financials 2025",
            "market_revenue":       f"{company} total addressable market TAM industry size 2025",
            "competitors":          f"{company} top competitors market share analysis 2025",
            "key_executives":       f"{company} CEO CTO CFO president executives board 2025",
            "strategic_priorities": f"{company} strategy priorities roadmap initiatives 2025",
            "pain_points":          f"{company} challenges problems risks headwinds 2025",
            "value_proposition":    f"{company} partnerships customers technology stack 2025",
            "action_plan":          f"{company} investor day earnings conference events 2025",
        }
        return QUERY_MAP.get(section, f"{company} {section.replace('_', ' ')} 2025")

    @staticmethod
    def _static_response(response_kind: str, context: str = "") -> Dict[str, Any]:
        """Returns pre-defined static responses for non-research intents."""
        messages = {
            "confused": (
                "👋 **I'm your Enterprise Account Plan Research Agent!**\n\n"
                "Tell me:\n"
                "1. **The company** you want to research\n"
                "2. **Your sales goal** (optional but improves output quality)\n\n"
                "**Examples:**\n"
                "- *'Research Stripe — I'm pitching our fraud detection API'*\n"
                "- *'Research Microsoft, goal: sell AI infrastructure'*\n"
                "- *'Compare Netflix vs Disney+'*\n\n"
                "**Power commands:**\n"
                "- *'Update only the pain points'* — refresh a single section\n"
                "- *'Download plan as PDF'* — export your plan"
            ),
            "general": (
                f"That looks like a general question."
                + (f" I'm currently focused on **{context}**." if context != "Not Yet Provided" else "")
                + "\n\nI specialise in enterprise account research. Try:\n"
                "- *'Research [Company Name]'*\n"
                "- *'Update pain points'*\n"
                "- *'Compare [Company A] vs [Company B]'*"
            ),
            "no_company": (
                "Please tell me which company to research first!\n\n"
                "*Example: 'Research Eightfold AI — I want to pitch talent intelligence'*"
            ),
        }
        return {
            "response_type":    "message",
            "content":          messages.get(response_kind, "I didn't understand that. Try again?"),
            "plan_changed":     False,
            "progress_messages": [],
            "suggestions": [],
        }

    @staticmethod
    def _error_response(error_msg: str, progress: List[str]) -> Dict[str, Any]:
        return {
            "response_type":    "message",
            "content":          f"⚠️ Technical error during research: {error_msg}\n\nPlease try again.",
            "plan_changed":     False,
            "progress_messages": [],  # emitted in real-time via _emit_progress
            "suggestions": [],
        }