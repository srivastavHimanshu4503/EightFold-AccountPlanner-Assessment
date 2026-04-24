import chainlit as cl
from research_agent import ResearchAgent

# Initialize our Maestro agent
agent = ResearchAgent()

def format_plan_to_markdown(plan_data: dict) -> str:
    """Converts the Account Plan dictionary into a beautiful structured Markdown document."""
    company = plan_data.get('company_name', 'Unknown')
    goal = plan_data.get('user_goal', 'General Research')
    
    md = f"### 📊 ACCOUNT PLAN: **{company.upper()}**\n"
    md += f"**🎯 User Goal:** {goal}\n\n---\n\n"

    # Define the sections and their emojis
    sections = [
        ("🏢 Company Overview", "company_overview"),
        ("💰 Financial Snapshot", "financial_snapshot"),
        ("👥 Key Executives", "key_executives"),
        ("📈 Strategic Priorities", "strategic_priorities"),
        ("⚠️ Pain Points", "pain_points"),
        ("💡 Value Proposition", "value_proposition"),
        ("🚀 Action Plan", "action_plan")
    ]

    for title, key in sections:
        value = plan_data.get(key)
        
        # Skip empty or unknown sections to keep it clean
        if not value or value == "Unknown" or value == []:
            continue

        md += f"#### {title}\n"
        
        # If the LLM correctly returned a list, format it as markdown bullets
        if isinstance(value, list):
            for item in value:
                # Clean up any rogue HTML or bullets the LLM might have hallucinated
                clean_item = str(item).replace("<br>", "").replace("•", "").strip()
                md += f"* {clean_item}\n"
        else:
            # If it's a string, just print it (cleaning up any rogue <br> tags)
            clean_val = str(value).replace("<br>", "\n").strip()
            md += f"{clean_val}\n"
            
        md += "\n"

    return md

@cl.on_chat_start
async def start():
    """Executes when a user opens the web UI."""
    # Reset the agent state so every new browser refresh is a clean slate
    agent.state.reset_plan()
    
    welcome_message = (
        "Welcome to the **Eightfold AI Research Agent**! 🚀\n\n"
        "I can help you build comprehensive Account Plans. Just give me the name of a company you want to research (e.g., 'Apple', 'Stripe', or 'Eightfold'), or ask me a question."
    )
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    """Executes every time the user sends a message."""
    user_input = message.content

    # Create a UI Step to show the Agent's "Thought Process" visually
    async with cl.Step(name="Agent Reasoning") as step:
        step.output = "Analyzing intent and fetching data..."
        
        # Call our existing Python logic from research_agent.py
        bot_response = agent.process_user_input(user_input)
        
        # Update the step to show completion
        step.output = "Analysis complete."

    # Fetch the newly updated state
    current_plan = agent.state.get_current_plan()
    
    # Format the response: The Bot's text + The Markdown Table
    final_output = f"{bot_response}\n\n"
    
    # Only append the table if it's not a general conversation rejection
    if "general question" not in bot_response.lower() or "I am an Account Plan Research Agent" in bot_response:
        final_output += format_plan_to_markdown(current_plan)

    # Send the final formatted response back to the UI
    await cl.Message(content=final_output).send()