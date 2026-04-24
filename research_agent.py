from dotenv import load_dotenv
from typing import Dict, Any, List
from ddgs import DDGS
from serpapi import GoogleSearch
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
import time
import json
import os

load_dotenv()

class AccountPlanState:
    """
    Manages the memory and data structure of the Account Plan.
    Acts as the single source of truth during the research process.
    """
    def __init__(self):
        # The core data structure for our Account Plan.
        # Initialized with default values to ensure type safety.
        self.plan: Dict[str, Any] = {
            "company_name": "Not Yet Provided",
            "user_goal": "Not Yet Provided", # e.g., "Sell cloud security"
            "company_overview": "Unknown",
            "financial_snapshot": "Unknown",
            "key_executives": [],
            "strategic_priorities": [], # Extracted from news
            "pain_points": [], # Synthesized based on the user's goal
            "value_proposition": "Unknown", # How the user's product helps them
            "action_plan": [] # Suggested next steps
        }
        
        # Agentic State Tracking: 
        # This is crucial for the "Agentic Behaviour" evaluation. 
        # It holds conflicting data or clarifying questions for the user.
        self.open_questions: List[str] = []
    
    def reset_plan(self):
        """Wipes the memory clean to start a new account plan."""
        self.plan: Dict[str, Any] = {
            "company_name": "Not Yet Provided",
            "user_goal": "Not Yet Provided", # e.g., "Sell cloud security"
            "company_overview": "Unknown",
            "financial_snapshot": "Unknown",
            "key_executives": [],
            "strategic_priorities": [], # Extracted from news
            "pain_points": [], # Synthesized based on the user's goal
            "value_proposition": "Unknown", # How the user's product helps them
            "action_plan": [] # Suggested next steps
        }
        self.open_questions = []

    def update_section(self, section_key: str, data: Any) -> bool:
        """
        Safely updates a specific section of the account plan.
        Returns True if successful, False if the key doesn't exist.
        """
        if section_key in self.plan:
            self.plan[section_key] = data
            return True
        print(f"Error: Attempted to update invalid section '{section_key}'")
        return False

    def add_open_question(self, question: str) -> None:
        """Adds a conflict or clarifying question to the queue."""
        self.open_questions.append(question)

    def clear_open_questions(self) -> None:
        """Clears the queue once the user resolves the conflicts."""
        self.open_questions = []

    def get_current_plan(self) -> Dict[str, Any]:
        """Returns the current state of the plan as a dictionary."""
        return self.plan

class ResearchTool:
    """
    Handles fetching real-world data from the web using DuckDuckGo.
    This acts as the agent's eyes, gathering raw information.
    """
    def __init__(self, max_results: int = 5):
        # We limit results to keep the LLM context window small and focused
        self.max_results = max_results
        self.api_key = os.getenv("SERP_API_KEY")

    def search_web(self, query: str) -> List[Dict[str, str]]:
        """
        Executes a web search (Standard + News). Tries SerpAPI first. 
        Falls back to DuckDuckGo gracefully if rate-limited or unavailable.
        """
        print(f"\n[Agent is searching the web for: '{query}'...]")
        results = []

        # Phase 1: Try SerpAPI
        try:
            if not self.api_key:
                raise Exception("No SerpAPI key provided.")
            
            # Standard Search
            standard_search = GoogleSearch({"q": query, "api_key": self.api_key}).get_dict()
            if "error" in standard_search:
                raise Exception(f"SerpAPI Error: {standard_search['error']}")

            # News Search
            news_search = GoogleSearch({"q": query, "tbm": "nws", "api_key": self.api_key}).get_dict()

            # Parse Google Featured Answers (Highest Quality)
            if "related_questions" in standard_search:
                for q in standard_search["related_questions"][:2]:
                    results.append({
                        "category": "COMPANY OVERVIEW (Verified)",
                        "title": q.get("question", "FAQ"),
                        "snippet": q.get("snippet", "")
                    })

            # Parse Standard Organic Results
            if "organic_results" in standard_search:
                for r in standard_search["organic_results"][:self.max_results]:
                    results.append({
                        "category": "COMPANY OVERVIEW",
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", "")
                    })

            # Parse News Results
            if "news_results" in news_search:
                for n in news_search["news_results"][:self.max_results]:
                    results.append({
                        "category": "RECENT NEWS & SIGNALS",
                        "title": n.get("title", ""),
                        "snippet": n.get("snippet", "")
                    })
                    
            print(f"[Search complete. Gathered {len(results)} sources via SerpAPI.]")
            return results

        # Phase 2: The Graceful Fallback (DuckDuckGo)
        except Exception as e:
            print(f"[!] Primary API unavailable ({e}). Falling back to DuckDuckGo...")
            
            try:
                with DDGS() as ddgs:
                    # 1. Get standard text results
                    raw_text = ddgs.text(query, max_results=self.max_results)
                    for r in raw_text:
                        results.append({
                            "category": "COMPANY OVERVIEW",
                            "title": r.get("title", "No Title"),
                            "snippet": r.get("body", "No content")
                        })
                    
                    # 2. Get news results using DDGS native news search!
                    raw_news = ddgs.news(query, max_results=self.max_results)
                    for r in raw_news:
                        results.append({
                            "category": "RECENT NEWS & SIGNALS",
                            "title": r.get("title", "No Title"),
                            "snippet": r.get("body", "No content")
                        })
                
                print(f"[Search complete. Gathered {len(results)} sources via DDGS.]")
                return results

            except Exception as ddgs_error:
                print(f"[!] Critical Error: Both Search APIs failed. {ddgs_error}")
                return [{"category": "ERROR", "title": "Search Failed", "snippet": str(ddgs_error)}]

    def format_for_llm(self, search_results: List[Dict[str, str]]) -> str:
        """
        Takes the unified dictionary list and formats it into a clean string 
        so Llama-3 can read the categories (Overview vs News).
        """
        formatted_text = ""
        for res in search_results:
            formatted_text += f"--- {res.get('category', 'INFO')} ---\n"
            formatted_text += f"Title: {res.get('title', 'Unknown')}\n"
            formatted_text += f"Snippet: {res.get('snippet', 'No data')}\n\n"
            
        return formatted_text

class LLMEngine:
    """
    The brain of the agent. Connects to Groq's Llama 3 model to parse
    messy web text into clean, structured JSON data.
    """
    def __init__(self, api_key: str = None):
        # Securely load the API key from environment variables or direct input
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is missing. Please set GROQ_API_KEY.")
        
        self.client = Groq(api_key=self.api_key)
        # We use Llama 3 70B because it has excellent reasoning and instruction following
        self.model = "llama-3.3-70b-versatile"
    
    def classify_intent(self, user_input: str, current_company: str) -> dict:
        """
        Acts as a semantic router. Classifies the user's input and extracts context.
        """
        
        router_prompt = f"""
        You are an intent classification router for an AI assistant.
        The user is currently researching the company: "{current_company}".
        
        USER INPUT: "{user_input}"
        
        Classify the intent into EXACTLY one of these four categories:
        1. "NEW_COMPANY": The user wants to abandon the current company and start a completely new account plan.
        2. "GENERAL_QUESTION": The user is asking a general definition or off-topic question.
        3. "CURRENT_COMPANY": The user is asking a follow-up about the active company.
        4. "CONFUSED_USER": The user is asking for help or doesn't know what to do.
        
        *** CRITICAL COMPARISON RULE ***
        If the user mentions a new company in order to COMPARE it to the current company (e.g., "How do they compare to Apple?", "What is their revenue vs Google?"), the intent is STILL "CURRENT_COMPANY". Do NOT trigger "NEW_COMPANY" unless the user explicitly wants to stop researching {current_company}.
        
        If the intent is "NEW_COMPANY", you must also extract the target 'company_name' AND the user's underlying 'goal' (why they are researching them, e.g., 'selling cloud security', 'preparing for an interview'). If they don't state a goal, set 'goal' to "General Research".
        
        Output ONLY a JSON object with this exact structure:
        {{
            "intent": "THE_CATEGORY",
            "company_name": "Extracted Name or None",
            "goal": "Extracted Goal or General Research"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": router_prompt}],
                model=self.model,
                temperature=0.0, # Zero temperature for absolute strict routing
                response_format={"type": "json_object"}
            )
            parsed = json.loads(response.choices[0].message.content)
            return parsed # We now return the whole dictionary, not just the string!
            
        except Exception as e:
            print(f"[!] Router Error: {e}")
            # Fallback dictionary if the API hiccups
            return {"intent": "CURRENT_COMPANY", "company_name": None, "goal": "General Research"}

    def extract_info(self, raw_text: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the raw text to Llama 3 and forces it to return ONLY a JSON object
        updating our Account Plan structure, synthesizing strategic sales insights.
        """
        print("[Agent is synthesizing strategic data...]")
        
        # The Upgraded Sales Strategist Prompt
        system_prompt = f"""
        You are an elite Enterprise Sales Strategist. Your job is to update the Account Plan based on the SEARCH RESULTS and the USER GOAL.

        CURRENT ACCOUNT PLAN STATE:
        {json.dumps(current_state, indent=2)}

        USER GOAL (Why the user is researching this company): {current_state.get('user_goal', 'General Research')}

        RULES:
        1. EXTRACT FACTS: Update basic fields (company_overview, financial_snapshot, key_executives, action_plan) using the provided text. Be highly detailed and descriptive.
        2. SYNTHESIZE STRATEGIC PRIORITIES: Read the 'Recent News & Signals' in the text and deduce what the company is currently focused on (e.g., European expansion, cost-cutting, AI adoption, new product launches). Add these to 'strategic_priorities'.
        3. SYNTHESIZE PAIN POINTS: Based on their Strategic Priorities and the USER GOAL, deduce 2-3 likely pain points. (e.g., If the goal is selling cybersecurity, and news shows they are expanding to Europe, a pain point is "GDPR compliance latency"). Add these to 'pain_points'.
        4. CRAFT VALUE PROPOSITION: Write a targeted 2-3 sentence pitch connecting the USER GOAL to the company's pain points and strategic priorities. Place this in 'value_proposition'.
        5. AGENTIC CONFLICT: If you find conflicting factual information, add a clear question to the "open_questions" list asking the user how to resolve it.
        6. STRICT DATA TYPES: 'key_executives', 'strategic_priorities', 'pain_points', and 'action_plan' MUST be JSON arrays of strings. DO NOT output a single string with HTML `<br>` tags or bullet points. 
        7. STRICT OUTPUT: Return ONLY a valid JSON object matching the exact keys in the CURRENT ACCOUNT PLAN STATE. Do not invent new keys. Do not wrap in markdown blocks.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"SEARCH RESULTS:\n{raw_text}"}
                ],
                model=self.model,
                temperature=0.1, # Low temperature ensures factual, deterministic output
                response_format={"type": "json_object"} # Forces the model to output strict JSON
            )
            
            # Extract the string output
            llm_output = response.choices[0].message.content
            
            # Parse the string into a Python dictionary
            parsed_json = json.loads(llm_output)
            return parsed_json

        except json.JSONDecodeError:
            print("[!] Error: LLM did not return valid JSON.")
            return {"error": "Invalid JSON returned."}
        except Exception as e:
            print(f"[!] Groq API Error: {e}")
            return {"error": str(e)}

class ResearchAgent:
    """
    The Maestro. Manages the execution flow between the User, Tool, LLM, and State.
    """
    def __init__(self):
        self.state = AccountPlanState()
        self.tool = ResearchTool()
        self.llm = LLMEngine() 

    def process_user_input(self, user_input: str) -> str:
        
        # 1. ALWAYS ROUTE FIRST: Get the context before making any decisions
        current_company = self.state.plan.get("company_name", "Not Yet Provided")
        print("[Agent is routing the query...]")
        
        router_data = self.llm.classify_intent(user_input, current_company)
        intent = router_data.get("intent", "CURRENT_COMPANY")
        extracted_company = router_data.get("company_name")
        user_goal = router_data.get("goal", "General Research")

        # 2. Handle Manual Commands & Non-Research Intents
        if "show" in user_input.lower() or "plan" in user_input.lower() and len(user_input) < 15:
            return "Here is the current state of the Account Plan."
            
        if intent == "CONFUSED_USER":
            return "I am an Account Plan Research Agent! Tell me the name of a company you want to research, and optionally what you are trying to sell them (e.g., 'Research TechNova, I want to sell them cloud security'). Do you have a company in mind?"

        if intent == "GENERAL_QUESTION":
            return "That seems like a general question. I am currently focused on researching companies for Account Plans. How can I help you with your company research?"
            
        # 3. Handle Context Switches (The Hostage Fix)
        if intent == "NEW_COMPANY":
            if current_company != "Not Yet Provided":
                print("[!] Context switch detected. Wiping memory for new company...")
            
            # Wiping the memory ALSO clears self.state.open_questions!
            self.state.reset_plan()
            
            if extracted_company:
                self.state.update_section("company_name", extracted_company)
            self.state.update_section("user_goal", user_goal)

        # 4. NOW check Conflict Resolution
        # (If the user changed companies above, this is now empty, bypassing the trap)
        if len(self.state.open_questions) > 0:
            return self.handle_conflict_resolution(user_input)
            
        # 5. Information Retrieval Setup
        search_query = user_input
        
        if intent == "CURRENT_COMPANY" and current_company != "Not Yet Provided":
            search_query = f"{current_company} {user_input}"
        elif intent == "NEW_COMPANY" and extracted_company:
            search_query = f"{extracted_company} recent news and company strategy"

        # 6. Web Search & Formatting
        raw_data = self.tool.search_web(search_query)
        llm_ready_string = self.tool.format_for_llm(raw_data)
        
        # 7. LLM Extraction
        current_plan_copy = self.state.get_current_plan().copy()
        current_plan_copy["open_questions"] = []
        
        updated_data = self.llm.extract_info(llm_ready_string, current_plan_copy)
        
        if "error" in updated_data:
            return "I encountered a technical error while analyzing the data."

        # 8. Queue Management & State Update
        questions = updated_data.get("open_questions", [])
        if "open_questions" in updated_data:
            del updated_data["open_questions"]
            
        for key, value in updated_data.items():
            self.state.update_section(key, value)
            
        if questions:
            self.state.open_questions = questions
            return f"[!] I need clarification before proceeding:\n- {questions[0]}\n(Type your answer, or type 'skip' if you don't know)"
        else:
            return f"I have successfully analyzed the data and updated the Account Plan for {self.state.plan.get('company_name')}."

    
    def handle_conflict_resolution(self, user_input: str) -> str:
        """
        Evaluates the user's answer to a pending conflict and manages the question queue.
        """
        # Pop removes the first question from the list and stores it in the variable
        question = self.state.open_questions.pop(0) 
        
        # 1. Edge Case: User wants to skip
        if user_input.lower() in ["skip", "i don't know", "none", "ignore"]:
            if len(self.state.open_questions) > 0:
                return f"Skipped. Next clarification needed:\n- {self.state.open_questions[0]}"
            return "Understood. I will leave that field as is."
        
        # 2. Execution: Parsing and Validating the answer
        print("[Agent is validating and updating the plan...]")
        
        resolution_text = f"""
        The user was asked to resolve this conflict: '{question}'.
        The user provided this answer: '{user_input}'.
        
        RULES:
        1. CATEGORY CHECK: Evaluate if the user's answer fits the category of the question.
        2. PLAUSIBILITY CHECK: Use your internal knowledge to evaluate if the answer is factually possible or realistic. For example, no single company has a revenue of 100 Trillion dollars. 
        3. If the answer is valid and plausible, update the appropriate field in the JSON.
        4. If the answer is factually absurd, clearly a joke, or irrelevant, DO NOT update the field. Instead, return a polite but firm rejection in the "open_questions" array, such as: "That figure ($100 Trillion) is mathematically highly improbable. Please provide a realistic revenue figure for: {question}"
        5. MISSING CRITICAL DATA: If the 'Company Name' or 'Industry' fields are currently 'Unknown' and you cannot find them in the search results, add a polite question to 'open_questions' asking the user for guidance (e.g., 'I couldn't find the exact industry for this company. Do you happen to know it, or should I leave it blank?'). Do NOT do this for Revenue or Competitors, as those are often private.
        """
        
        current_plan_copy = self.state.get_current_plan().copy()
        current_plan_copy["open_questions"] = []
        
        updated_data = self.llm.extract_info(resolution_text, current_plan_copy)
        
        # Did the LLM reject the user's answer and return a new question?
        new_questions = updated_data.get("open_questions", [])
        if "open_questions" in updated_data:
            del updated_data["open_questions"]
            
        for key, value in updated_data.items():
            self.state.update_section(key, value)
            
        # 3. Queue Management
        if new_questions:
            # User gave a bad answer, put the question back at the front of the line
            self.state.open_questions.insert(0, new_questions[0])
            return f"[!] {new_questions[0]}"
            
        if len(self.state.open_questions) > 0:
            # Move on to the next question in the queue
            return f"Got it. Next clarification needed:\n- {self.state.open_questions[0]}"
            
        return "Thank you! All conflicts resolved and the plan is updated."
    
class TerminalUI:
    """
    Handles the hacker-style aesthetic and user conversation loop using 'rich'.
    """
    def __init__(self, agent: ResearchAgent):
        self.agent = agent
        self.console = Console()

    def display_plan(self):
        """Renders the Account Plan as a beautiful Markdown-style table."""
        plan_data = self.agent.state.get_current_plan()
        
        table = Table(title="\n📊 CURRENT ACCOUNT PLAN", show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan", width=25)
        table.add_column("Data", style="green")

        for key, value in plan_data.items():
            # Format lists cleanly
            if isinstance(value, list):
                value = "\n".join([f"• {item}" for item in value]) if value else "None"
            table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(table)

    def start_chat_loop(self):
        self.console.clear()
        self.console.print(Panel.fit("[bold blue]Eightfold.ai Research Agent[/bold blue]\nType 'exit' to quit. Type 'show plan' to view the current data.", border_style="blue"))
        
        while True:
            # 1. Get input
            user_input = Prompt.ask("\n[bold yellow]User[/bold yellow]")
            
            if user_input.lower() in ['exit', 'quit']:
                self.console.print("[bold red]Exiting Agent... Good luck with the research![/bold red]")
                break
                
            # 2. Process input through the Agent
            self.console.print("\n[bold magenta]Agent is thinking...[/bold magenta]")
            response = self.agent.process_user_input(user_input)
            
            # 3. Display response and current plan
            self.console.print(f"\n[bold cyan]Agent:[/bold cyan] {response}")
            time.sleep(1) # Small pause for terminal UX

            # Only display the plan if it's NOT a clarifying question or general chat
            if "[!]" not in response and "general question" not in response.lower():
                self.display_plan()

class TestExecution:
    def account_plan_state_test(self):
        # A quick dry-run test of our State class
        state = AccountPlanState()
        print("Initial State:", json.dumps(state.get_current_plan(), indent=2))
        
        # Simulating a safe update
        state.update_section("company_name", "Eightfold AI")
        state.update_section("industry", "Talent Intelligence")
        
        print("\nUpdated State:", json.dumps(state.get_current_plan(), indent=2))

    def research_tool_test(self):
        # 1. Initialize the tool
        tool = ResearchTool(max_results=3)
        
        # 2. Execute a test search
        test_query = "Who is the CEO of Eightfold AI?"
        raw_data = tool.search_web(test_query)
        
        # 3. Format and print the results
        llm_ready_string = tool.format_for_llm(raw_data)
        print("\nWhat the LLM will see:")
        print(llm_ready_string)

    def llm_engine_test(self):
        # Note: To run this test, you must have your Groq API key set in your environment
        # e.g., export GROQ_API_KEY="your_key_here" (Linux/Mac) or set GROQ_API_KEY="your_key_here" (Windows)
        
        state_manager = AccountPlanState()
        tool = ResearchTool(max_results=3)
        
        # Initialize the engine (it will grab the key from your environment)
        try:
            engine = LLMEngine()
            
            # Simulating the pipeline we've built so far:
            test_query = "Who is the CEO of Eightfold AI?"
            raw_data = tool.search_web(test_query)
            llm_ready_string = tool.format_for_llm(raw_data)
            
            # Pass the formatted string and the current empty state to the brain
            updated_data = engine.extract_info(llm_ready_string, state_manager.get_current_plan())
            
            print("\nLLM Output (Parsed JSON):")
            print(json.dumps(updated_data, indent=2))
            
        except ValueError as e:
            print(e)

    def agent_execution_test(self):
        app_agent = ResearchAgent()
        ui = TerminalUI(agent=app_agent)
        ui.start_chat_loop()

# --- Execution ---
if __name__ == "__main__":
    # TestExecution().account_plan_state_test()
    # TestExecution().research_tool_test()
    # TestExecution().llm_engine_test()
    TestExecution().agent_execution_test()