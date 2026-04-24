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
            "industry": "Unknown",
            "headquarters": "Unknown",
            "revenue_or_size": "Unknown",
            "key_executives": [],
            "recent_news_and_initiatives": [],
            "competitors": [],
        }
        
        # Agentic State Tracking: 
        # This is crucial for the "Agentic Behaviour" evaluation. 
        # It holds conflicting data or clarifying questions for the user.
        self.open_questions: List[str] = []
    
    def reset_plan(self):
        """Wipes the memory clean to start a new account plan."""
        self.plan = {
            "company_name": "Not Yet Provided",
            "industry": "Unknown",
            "headquarters": "Unknown",
            "revenue_or_size": "Unknown",
            "key_executives": [],
            "recent_news_and_initiatives": [],
            "competitors": [],
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

    def search_web(self, query: str) -> List[Dict[str, str]]:
        """
        Executes a web search and returns a list of result dictionaries.
        """
        print(f"\n[Agent is searching the web for: '{query}'...]")
        results = []
        
        try:
            # Note: You'll need to install the 'google-search-results' package
            search = GoogleSearch({"q": query, "api_key": "YOUR_SERPAPI_KEY"})
            results = search.get_dict()
            
            if "error" in results:
                raise Exception("SerpAPI Quota Exceeded")
                
            # Parse SerpAPI results...
            parsed_results = [] # Add your parsing logic here
            return parsed_results

        # Phase 2: The Graceful Fallback
        except Exception as e:
            print(f"[!] Primary search API unavailable ({e}). Falling back to DuckDuckGo...")
            # Context manager ensures the connection closes properly
            with DDGS() as ddgs:
                # Fetch text-based search results
                raw_results = ddgs.text(query, max_results=self.max_results)
                
                for r in raw_results:
                    results.append({
                        "title": r.get("title", "No Title"),
                        "link": r.get("href", "No Link"),
                        "snippet": r.get("body", "No content available.")
                    })
            
            print(f"[Search complete. Gathered {len(results)} sources.]")
            return results
            
        except Exception as e:
            # Prevents the entire program from crashing if the network drops
            print(f"[!] Error during web search: {e}")
            return [{"error": str(e)}]

    def format_for_llm(self, results: List[Dict[str, str]]) -> str:
        """
        Takes the raw dictionary results and formats them into a clean string
        so the LLM can easily read and parse the information.
        """
        if not results or "error" in results[0]:
            return "No search results available."

        formatted_text = "--- SEARCH RESULTS ---\n"
        for i, res in enumerate(results, 1):
            formatted_text += f"Source {i} ({res.get('title')}): {res.get('snippet')}\n"
        formatted_text += "----------------------\n"
        
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
    
    def classify_intent(self, user_input: str, current_company: str) -> str:
        """
        Acts as a semantic router. Classifies the user's input into one of three categories.
        """
        router_prompt = f"""
        You are an intent classification router for an AI assistant.
        The user is currently researching the company: "{current_company}".
        
        USER INPUT: "{user_input}"
        
        Classify the intent into EXACTLY one of these three categories:
        1. "NEW_COMPANY": The user is asking to research a completely different company.
        2. "GENERAL_QUESTION": The user is asking a general definition, coding, or conversational question unrelated to the current company.
        3. "CURRENT_COMPANY": The user is asking a follow-up question about the current company, or using pronouns like "they/their" implying the current company.
        4. "CONFUSED_USER": The user appears uncertain, asks for help without a clear direction, provides irrelevant or incoherent responses, or explicitly requests guidance on what to do. The user may also ask for suggestions (e.g., which company to research) or show signs of confusion about the task. In such cases, the assistant should interpret the intent as needing clarification, guidance, or structured suggestions.
        
        Output ONLY a JSON object with a single key "intent" and the string value of the category.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": router_prompt}],
                model=self.model,
                temperature=0.0, # Zero temperature for absolute strict routing
                response_format={"type": "json_object"}
            )
            parsed = json.loads(response.choices[0].message.content)
            return parsed.get("intent", "CURRENT_COMPANY")
        except Exception:
            # Fallback to current company if the API hiccups
            return "CURRENT_COMPANY"

    def extract_info(self, raw_text: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the raw text to Llama 3 and forces it to return ONLY a JSON object
        updating our Account Plan structure.
        """
        print("[Agent is analyzing the data...]")
        
        # The System Prompt: This dictates the Agentic Behaviour
        system_prompt = f"""
        You are an expert corporate research assistant. Your job is to extract 
        information from the provided SEARCH RESULTS and update the CURRENT STATE 
        of the Account Plan.

        CURRENT STATE:
        {json.dumps(current_state, indent=2)}

        RULES:
        1. You must output ONLY a valid JSON object. Do not wrap it in markdown block quotes. No explanations.
        2. Keep the exact same keys as the CURRENT STATE. Update the values if you find new information.
        3. AGENTIC BEHAVIOUR: If you find CONFLICTING information for a single key (e.g., two different CEOs or conflicting revenue numbers), do NOT guess. Instead, add a clear question to the "open_questions" list asking the user how to resolve it.
        4. If you do not find new info for a key, leave it as its current value.
        5. MISSING CRITICAL DATA: If the 'Company Name' or 'Industry' fields are currently 'Unknown' and you cannot find them in the search results, add a polite question to 'open_questions' asking the user for guidance (e.g., 'I couldn't find the exact industry for this company. Do you happen to know it, or should I leave it blank?'). Do NOT do this for Revenue or Competitors, as those are often private.
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
        """
        Takes the user's string, decides the action, and executes the pipeline step-by-step.
        """
        # --- BRANCH 1: CONFLICT RESOLUTION ---
        # If there is an active question in memory, we treat the user input as the answer.
        if len(self.state.open_questions) > 0:
            return self.handle_conflict_resolution(user_input)

        # --- BRANCH 2: NORMAL RESEARCH ---
        if "show" in user_input.lower() or "plan" in user_input.lower() and len(user_input) < 15:
            return "Here is the current state of the Account Plan."

        # --- THE NEW INTENT ROUTER ---
        current_company = self.state.plan.get("company_name", "None")
        print("[Agent is routing the query...]")
        intent = self.llm.classify_intent(user_input, current_company)

        if intent == "GENERAL_QUESTION":
            return "That seems like a general question. I am currently focused on researching companies for Account Plans. How can I help you with your company research?"
            
        elif intent == "NEW_COMPANY":
            if current_company != "Not Yet Provided":
                print("[!] Context switch detected. Wiping memory for new company...")
            # The memory is wiped, so it will now naturally start a new plan!
            self.state.reset_plan()
        
        elif intent == "CONFUSED_USER":
            return  "It looks like you might need a starting point. I can research and build a detailed account profile for any company.\nJust share a company name (e.g., 'Apple', 'Stripe', or any other), and I’ll take it from there.\nWhat company would you like me to look into?"
            
        # 3. Information Retrieval (Slightly modified to ensure good search results)
        search_query = user_input
        if intent == "CURRENT_COMPANY" and current_company != "Not Yet Provided":
            # If the user says "what is their revenue?", we append the company name 
            # so DuckDuckGo knows exactly who "their" is.
            search_query = f"{current_company} {user_input}"
            
        raw_data = self.tool.search_web(search_query)
        llm_ready_string = self.tool.format_for_llm(raw_data)

        # 3. State Injection (We temporarily inject open_questions so the LLM can use it)
        # Fix: Use .copy() so we don't accidentally mutate our core state memory
        current_plan_copy = self.state.get_current_plan().copy()
        current_plan_copy["open_questions"] = []

        # 4. LLM Processing
        updated_data = self.llm.extract_info(llm_ready_string, current_plan_copy)

        if "error" in updated_data:
            return "I encountered a technical error while analyzing the data."

        # 5. Agentic Evaluation: Check if the LLM flagged any conflicts
        # Extract the questions safely
        questions = updated_data.get("open_questions", [])
        
        # Clean up the dictionary before saving it to state
        if "open_questions" in updated_data:
            del updated_data["open_questions"]

        # 6. Update the plan in memory
        for key, value in updated_data.items():
            self.state.update_section(key, value)

        # 7. Agentic Routing Evaluation
        if questions:
            # We save the conflict to our State class to trigger Branch 1 on the next loop
            self.state.open_questions = questions
            # The bot halts the update and asks the user for direction
            return f"[!] I need clarification before proceeding:\n- {questions[0]}\n(Type your answer, or type 'skip' if you don't know)"
        else:
            return "I have successfully analyzed the data and updated the Account Plan."

    
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