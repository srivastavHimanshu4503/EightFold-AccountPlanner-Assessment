"""
app.py — Chainlit UI for the Enterprise Research Agent
=======================================================
KEY FIXES FROM AUDIT:
    - Per-session ResearchAgent via cl.user_session (no global singleton).
    - Startup environment validation (fast-fail before any user message).
    - Streaming responses for direct follow-up answers.
    - Plan hash check to suppress redundant re-renders.
    - Confidence badges (🟢 HIGH / 🟡 MEDIUM / 🔴 LOW) per section.
    - Data quality warnings for private/low-data companies.
    - PDF export (fpdf2) with Markdown fallback.
    - SECTION_UPDATE, COMPARE, and DOWNLOAD intents wired through.
    - Progress step labels during research.
    - Audio handler uses per-session agent (not global).
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import traceback
import wave
from typing import Optional

import chainlit as cl
from dotenv import load_dotenv

from research_agent import ResearchAgent, validate_environment

load_dotenv()

# ── Startup Validation ────────────────────────────────────────────────────────
# Fail immediately at boot if API keys are missing, rather than on first message.
try:
    validate_environment()
except EnvironmentError as exc:
    raise SystemExit(f"[FATAL] Environment check failed: {exc}") from exc

logger = logging.getLogger("app")

# ── Confidence Badges ─────────────────────────────────────────────────────────
# Values in data_confidence may now be enriched strings such as:
#   "HIGH (verified from structured sources (filings, official reports))"
# The badge lookup uses startswith() so both plain "HIGH" and enriched strings work.
CONFIDENCE_LEVELS = {
    "HIGH":   ("🟢", "HIGH confidence"),
    "MEDIUM": ("🟡", "MEDIUM confidence"),
    "LOW":    ("🔴", "LOW — estimated"),
}

def extract_parentheses_content(text: str) -> str:
    start = text.find("(")
    if start == -1:
        return ""

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            stack += 1
        elif text[i] == ")":
            stack -= 1
            if stack == 0:
                return text[start + 1:i].strip()

    return ""  # no properly closed parentheses


def _confidence_badge(confidence_value: str) -> str:
    """
    Returns a formatted confidence badge from a plain or enriched confidence value.
    Enriched values carry a parenthetical explanation appended by LLMEngine.
    Examples:
      "HIGH"                                             → "🟢 *HIGH confidence*"
      "HIGH (verified from structured sources...)"       → "🟢 *HIGH confidence — verified from structured sources...*"
      "MEDIUM (derived from secondary sources (news...))" → "🟡 *MEDIUM confidence — derived from secondary sources...*"
    """
    if not confidence_value:
        return ""

    val = confidence_value.strip()

    for level, (emoji, label) in CONFIDENCE_LEVELS.items():
        if val.upper().startswith(level):

            explanation = extract_parentheses_content(val)

            if explanation:
                return f"{emoji} *{label} — {explanation}*"
            return f"{emoji} *{label}*"

    return f"*{val}*"


# ── Plan Formatting ───────────────────────────────────────────────────────────

def format_plan_to_markdown(plan_data: dict) -> str:
    """
    Converts an Account Plan dict into a structured Markdown document.
    Includes per-section confidence badges and data quality warnings.
    """
    company        = plan_data.get("company_name", "Unknown")
    goal           = plan_data.get("user_goal",    "General Research")
    confidence_map = plan_data.get("data_confidence", {})
    warnings       = plan_data.get("data_warnings",   [])

    md  = f"### 📊 ACCOUNT PLAN: **{company.upper()}**\n"
    md += f"**🎯 Sales Goal:** {goal}\n\n"

    if warnings:
        for w in warnings:
            md += f"> ⚠️ **Data Notice:** {w}\n"
        md += "\n"

    md += "---\n\n"

    sections = [
        ("🏢 Company Overview",      "company_overview"),
        ("💰 Financial Snapshot",    "financial_snapshot"),
        ("📈 Market Size (TAM)",     "market_revenue"),
        ("⚔️  Top Competitors",       "competitors"),
        ("👥 Key Executives",        "key_executives"),
        ("🎯 Strategic Priorities",  "strategic_priorities"),
        ("⚠️  Pain Points",           "pain_points"),
        ("💡 Value Proposition",     "value_proposition"),
        ("🚀 Action Plan",           "action_plan"),
    ]

    for title, key in sections:
        value = plan_data.get(key)
        if not value or value in ("Unknown", []):
            continue

        badge = _confidence_badge(confidence_map.get(key, ""))
        badge_str = f"  {badge}" if badge else ""

        md += f"#### {title}{badge_str}\n"

        if isinstance(value, list):
            for item in value:
                clean = str(item).replace("<br>", "").replace("•", "").strip()
                md += f"* {clean}\n"
        else:
            clean = str(value).replace("<br>", "\n").strip()
            md += f"{clean}\n"

        md += "\n"

    # ── Source Attribution Footer ─────────────────────────────────────────────
    # Surfaces the URLs used by the LLM so users can verify key claims.
    # A professional B2B research tool must be verifiable — this is a trust signal.
    sources = plan_data.get("source_references", [])
    if sources:
        md += "---\n\n#### 📎 Sources\n"
        for src in sources:
            clean_src = str(src).strip()
            if clean_src:
                md += f"* {clean_src}\n"
        md += "\n"

    return md

def sanitize_for_pdf(text: str) -> str:
    replacements = {
        "—": "-",
        "–": "-",
        "•": "-",
        "🟢": "",
        "🟡": "",
        "🔴": "",
        "🚀": "",
        "💡": "",
        "📊": "",
        "📈": "",
        "📎": "",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def export_plan_to_pdf(plan_data: dict) -> Optional[bytes]:
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_margins(15, 15, 15)
        pdf.add_page()

        # ── Font Setup (Hybrid Unicode + Fallback) ───────────────────────────
        use_unicode_font = False
        font_path = os.path.join(os.getcwd(), "DejaVuSans.ttf")

        if os.path.exists(font_path):
            try:
                pdf.add_font("DejaVu", "", font_path, uni=True)
                pdf.add_font("DejaVu", "B", font_path, uni=True)
                use_unicode_font = True
            except Exception:
                use_unicode_font = False

        def set_font(style="", size=10):
            font_name = "DejaVu" if use_unicode_font else "Helvetica"
            pdf.set_font(font_name, style, size)

        def wrap_long_words(text: str, max_len=60):
            words = text.split()
            out = []
            for w in words:
                if len(w) > max_len:
                    out.append(w[:max_len] + "-")
                    out.append(w[max_len:])
                else:
                    out.append(w)
            return " ".join(out)

        def clean_text(text: str) -> str:
            text = str(text).replace("<br>", "\n").strip()
            text = wrap_long_words(text)
            return text if use_unicode_font else sanitize_for_pdf(text)

        def safe_multi_cell(text: str, line_height=7):
            pdf.set_x(pdf.l_margin)
            usable_width = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.multi_cell(usable_width, line_height, text)

        company        = plan_data.get("company_name", "Unknown").upper()
        goal           = plan_data.get("user_goal", "General Research")
        confidence_map = plan_data.get("data_confidence", {})
        warnings       = plan_data.get("data_warnings", [])

        # ── Header ───────────────────────────────────────────────────────────
        pdf.set_fill_color(20, 20, 80)
        pdf.set_text_color(255, 255, 255)
        set_font("B", 18)
        pdf.cell(0, 14, clean_text(f"ACCOUNT PLAN: {company}"), fill=True, new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.ln(2)

        pdf.set_text_color(50, 50, 50)
        set_font("", 11)
        pdf.cell(0, 8, clean_text(f"Sales Goal: {goal}"), new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.ln(5)

        # ── Warnings ─────────────────────────────────────────────────────────
        if warnings:
            set_font("I", 9)
            pdf.set_text_color(160, 80, 0)
            for w in warnings:
                safe_multi_cell(clean_text(f"[!] {w}"), 6)
            pdf.ln(3)

        # ── Sections ─────────────────────────────────────────────────────────
        CONF_COLORS = {
            "HIGH":   (0, 140, 0),
            "MEDIUM": (180, 120, 0),
            "LOW":    (180, 0, 0),
        }

        sections = [
            ("Company Overview", "company_overview"),
            ("Financial Snapshot", "financial_snapshot"),
            ("Market Size (TAM)", "market_revenue"),
            ("Top Competitors", "competitors"),
            ("Key Executives", "key_executives"),
            ("Strategic Priorities", "strategic_priorities"),
            ("Pain Points", "pain_points"),
            ("Value Proposition", "value_proposition"),
            ("Action Plan", "action_plan"),
        ]

        for title, key in sections:
            value = plan_data.get(key)
            if not value or value in ("Unknown", []):
                continue

            conf = confidence_map.get(key, "")
            conf_short = conf.split("(")[0].strip() if conf else ""
            conf_label = f" [{conf_short}]" if conf_short else ""

            # Section header
            pdf.set_fill_color(235, 235, 255)
            pdf.set_text_color(20, 20, 80)
            set_font("B", 12)
            pdf.cell(0, 9, clean_text(f"{title}{conf_label}"), fill=True, new_x="LMARGIN", new_y="NEXT")

            if conf_short in CONF_COLORS:
                r, g, b = CONF_COLORS[conf_short]
                pdf.set_draw_color(r, g, b)
                y = pdf.get_y()
                pdf.line(15, y, 195, y)

            pdf.ln(2)

            # Content
            set_font("", 10)
            pdf.set_text_color(40, 40, 40)

            if isinstance(value, list):
                for item in value:
                    safe_multi_cell(f"  - {clean_text(item)}", 7)
            else:
                safe_multi_cell(clean_text(value), 7)

            pdf.ln(5)

        # ── Sources ──────────────────────────────────────────────────────────
        sources = plan_data.get("source_references", [])
        if sources:
            pdf.set_fill_color(245, 245, 245)
            pdf.set_text_color(80, 80, 80)
            set_font("B", 11)
            pdf.cell(0, 9, "Sources & References", fill=True, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(2)

            set_font("", 9)
            for src in sources:
                safe_multi_cell(f"  - {clean_text(src)}", 6)

            pdf.ln(3)

        return bytes(pdf.output())

    except ImportError:
        logger.warning("fpdf2 not installed. Install with: pip install fpdf2")
        return None
    except Exception as exc:
        logger.error("PDF export error: %s", exc, exc_info=True)
        raise RuntimeError(f"PDF generation failed: {exc}") from exc
    
    
def _should_render_plan(result: dict) -> bool:
    """
    Returns True only when a full plan re-render is warranted.
    Prevents the screen filling with identical plan tables on every message.
    """
    return (
        result.get("response_type") == "plan"
        and result.get("plan_changed", False)
    )


# ── Session Helpers ───────────────────────────────────────────────────────────

def _get_session_agent() -> ResearchAgent:
    """Retrieves the per-session agent, creating a new one if missing."""
    agent = cl.user_session.get("agent")
    if agent is None:
        logger.warning("No agent in session — creating new one.")
        agent = ResearchAgent()
        cl.user_session.set("agent", agent)
    return agent


# ── Chainlit Handlers ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def start() -> None:
    """
    Runs once when a user opens the chat.
    Creates a fresh ResearchAgent per session — never shared globally.
    """
    agent = ResearchAgent()
    cl.user_session.set("agent", agent)

    await cl.Message(content=(
        "**⚡ Enterprise Research Agent — Ready**\n\n"
        "I synthesise real-time, actionable Account Plans for enterprise sales teams.\n\n"
        "**Get started:**\n"
        "- *'Research Stripe — I'm pitching our fraud detection API'*\n"
        "- *'Research Microsoft, goal: sell AI infrastructure'*\n"
        "- *'Compare Netflix vs Disney+'*\n\n"
        "**Power commands:**\n"
        "- Refresh a section: *'Update competitors'* / *'Regenerate pain points'*\n"
        "- Follow-ups: *'Who is the CFO?'* / *'What are their latest earnings?'*\n"
        "- Export: *'Download plan as PDF'* / *'Export as Markdown'*"
    )).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    """
    Per-message handler. Retrieves the per-session agent from cl.user_session
    so each user's state is isolated.
    """
    agent     = _get_session_agent()
    user_input = message.content.strip()
    if not user_input:
        return

    # ── Export shortcut (intercept before routing) ────────────────────────────
    export_keywords = {"download", "export", "save", "pdf", "markdown"}
    if any(kw in user_input.lower() for kw in export_keywords):
        await _handle_export(agent, user_input)
        return

    # ── Main pipeline ─────────────────────────────────────────────────────────
    result: Optional[dict] = None

    async with cl.Step(name="🤔 Processing...") as thinking_step:
        thinking_step.output = "Classifying intent..."
        try:
            result = await agent.process_user_input(user_input)
            thinking_step.output = "Complete."
        except Exception as exc:
            logger.error("process_user_input error: %s\n%s", exc, traceback.format_exc())
            await cl.Message(
                content=f"⚠️ Unexpected error: {exc}\n\nPlease try again."
            ).send()
            return

    if result is None:
        await cl.Message(content="⚠️ No response generated. Please try again.").send()
        return

    # ── Show research progress steps ──────────────────────────────────────────
    for label in result.get("progress_messages", []):
        async with cl.Step(name=label):
            pass

    await _dispatch_result(agent, result, user_input)


async def _dispatch_result(agent: ResearchAgent, result: dict, user_input: str = "") -> None:
    """
    Routes a result dict to the appropriate Chainlit output method.
    user_input is passed through so DOWNLOAD_PLAN can distinguish pdf vs markdown.
    """
    response_type = result.get("response_type", "message")

    if response_type == "stream":
        # Stream direct Q&A answer token-by-token
        msg = cl.Message(content="")
        await msg.send()
        async for token in result["stream_gen"]:
            await msg.stream_token(token)
        await msg.update()

    elif response_type == "comparison":
        await cl.Message(content=result["content"]).send()

    elif response_type == "clarification":
        await cl.Message(content=result["content"]).send()

    elif response_type == "plan":
        # Confirmation message (e.g. "✅ Refreshed pain_points for Netflix")
        if result.get("content"):
            await cl.Message(content=result["content"]).send()
        # Re-render plan only when it actually changed — prevents redundant scroll
        if _should_render_plan(result):
            plan_md = format_plan_to_markdown(agent.state.get_current_plan())
            await cl.Message(content=plan_md).send()
        # Render proactive suggestions below main content (clean bullet list)
        suggestions = result.get("suggestions", [])
        if suggestions:
            suggestion_md = "\n\n---\n**💡 Proactive suggestions:**\n"
            suggestion_md += "\n".join(f"- {s}" for s in suggestions)
            await cl.Message(content=suggestion_md).send()
        # If plan_changed is False, the confirmation message above is sufficient.
        # We deliberately do NOT show an "unchanged" system message to the user.

    elif response_type == "download":
        # Preserve user intent: route to PDF or Markdown based on original message
        await _handle_export(agent, user_input or "markdown")

    else:
        # Generic message — show content, then plan if it changed
        content = result.get("content", "")
        if content:
            await cl.Message(content=content).send()
        if result.get("plan_changed", False):
            plan_md = format_plan_to_markdown(agent.state.get_current_plan())
            await cl.Message(content=plan_md).send()


async def _handle_export(agent: ResearchAgent, user_input: str) -> None:
    """
    Exports the current plan to PDF (if fpdf2 is available) or Markdown.
    In-memory only — no disk I/O, safe for concurrent cloud deployments.
    """
    plan = agent.state.get_current_plan()

    if plan.get("company_name") == "Not Yet Provided":
        await cl.Message(
            content="📭 No active plan to export. Research a company first!"
        ).send()
        return

    company_slug = (
        plan["company_name"].replace(" ", "_").replace("/", "-").replace("\\", "-")
    )

    want_pdf = "pdf" in user_input.lower()

    if want_pdf:
        pdf_bytes = None
        pdf_error = None
        async with cl.Step(name="📄 Generating PDF..."):
            try:
                pdf_bytes = await _run_in_thread(export_plan_to_pdf, plan)
            except RuntimeError as exc:
                pdf_error = str(exc)

        if pdf_bytes:
            await cl.Message(
                content=f"📄 PDF Account Plan for **{plan['company_name']}** is ready:",
                elements=[
                    cl.File(
                        name=f"{company_slug}_Account_Plan.pdf",
                        content=pdf_bytes,
                        display="inline",
                        mime="application/pdf",
                    )
                ],
            ).send()
            return
        else:
            # Show the real error reason, not the generic "fpdf2 not installed" message
            if pdf_error:
                await cl.Message(
                    content=f"⚠️ {pdf_error}\n\nExporting as Markdown instead..."
                ).send()
            else:
                await cl.Message(
                    content="⚠️ fpdf2 is not installed. Run `pip install fpdf2` then restart.\n\nExporting as Markdown instead..."
                ).send()
            # Fall through to Markdown

    # Markdown export
    plan_md = format_plan_to_markdown(plan)
    await cl.Message(
        content=f"📄 Markdown Account Plan for **{plan['company_name']}** ready for download:",
        elements=[
            cl.File(
                name=f"{company_slug}_Account_Plan.md",
                content=plan_md.encode("utf-8"),
                display="inline",
                mime="text/markdown",
            )
        ],
    ).send()



async def _run_in_thread(fn, *args):
    """Runs a blocking function in a thread pool to avoid blocking the event loop."""
    return await asyncio.to_thread(fn, *args)




@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk) -> None:
    """Accumulates PCM audio chunks in the per-session buffer."""
    buffer = cl.user_session.get("audio_buffer") or []
    if isinstance(chunk, dict):
        buffer.append(chunk.get("data", b""))
    elif hasattr(chunk, "data"):
        buffer.append(chunk.data)
    else:
        buffer.append(bytes(chunk))
    cl.user_session.set("audio_buffer", buffer)


@cl.on_audio_end
async def on_audio_end(*_args, **_kwargs) -> None:
    """Transcribes accumulated audio and routes to the main message handler."""
    chunks = cl.user_session.get("audio_buffer") or []
    cl.user_session.set("audio_buffer", [])  # clear immediately

    if not chunks:
        await cl.Message(
            content="⚠️ No audio captured. Please check your microphone permissions."
        ).send()
        return

    try:
        async with cl.Step(name="🎙️ Transcribing audio...") as step:
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24_000)
                wf.writeframes(b"".join(chunks))
            wav_io.seek(0)
            audio_bytes = wav_io.read()

            agent = _get_session_agent()
            # Transcription uses the sync Groq client (Whisper is not async)
            transcription = await asyncio.to_thread(
                agent.llm.client.audio.transcriptions.create,
                file=("voice_memo.wav", audio_bytes),
                model="whisper-large-v3",
                response_format="text",
            )
            user_input = (transcription or "").strip()

            if not user_input:
                step.output = "Could not transcribe audio."
                await cl.Message(
                    content="⚠️ I couldn't hear any words. Please try speaking clearly."
                ).send()
                return

            step.output = f'Transcribed: "{user_input}"'

        await cl.Message(content=f"🎤 *Heard:* {user_input}").send()

        # Audio ambiguity is too high for export commands — require typing
        export_keywords = {"download", "export", "save", "pdf"}
        if any(kw in user_input.lower() for kw in export_keywords):
            await cl.Message(
                content="To export your plan, please type the command instead of speaking it."
            ).send()
            return

        # Re-use the main message handler
        await main(cl.Message(content=user_input))

    except Exception as exc:
        logger.error("Audio processing error: %s\n%s", exc, traceback.format_exc())
        await cl.Message(
            content=f"⚠️ Audio processing error: {str(exc)}"
        ).send()