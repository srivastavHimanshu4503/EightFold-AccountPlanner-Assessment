import chainlit as cl
from research_agent import ResearchAgent
import io
import wave
import traceback

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
        "**System Initialized: Enterprise Research Agent** 🟢\n\n"
        "I am ready to synthesize your next Account Plan. To get the highest quality output, please provide the target company and your specific strategic goal.\n\n"
        "*Example: 'Research Stripe. We are pitching our new enterprise fraud detection API.'*"
    )
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    """Executes every time the user sends a message."""
    user_input = message.content

    # --- NEW: DOWNLOAD INTERCEPTOR ---
    if "download" in user_input.lower() or "export" in user_input.lower():
        current_plan = agent.state.get_current_plan()
        if current_plan.get("company_name") == "Not Yet Provided":
            await cl.Message(content="There is no active plan to download yet!").send()
            return

        # 1. Generate the Markdown string
        plan_md = format_plan_to_markdown(current_plan)
        company_safe_name = current_plan['company_name'].replace(' ', '_')
        file_name = f"{company_safe_name}_Account_Plan.md"
        
        # 2. Generate the file purely in-memory (no hard drive saving!)
        # Explicitly setting mime="text/markdown" prevents the frontend crash.
        elements = [
            cl.File(
                name=file_name, 
                content=plan_md.encode('utf-8'), # Encode string to raw bytes
                display="inline",
                mime="text/markdown" 
            )
        ]
        
        # 3. Serve the file to the user
        await cl.Message(content="Here is your Account Plan ready for download! 📄", elements=elements).send()
        return

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

import io
import wave
import traceback # Added for deep error hunting

@cl.on_audio_start
async def on_audio_start():
    print("\n--- [AUDIO EVENT] Microphone Started! ---")
    cl.user_session.set("audio_buffer", [])
    return True 

@cl.on_audio_chunk
async def on_audio_chunk(chunk):
    buffer = cl.user_session.get("audio_buffer")
    
    # Failsafe: If start wasn't triggered properly
    if buffer is None:
        buffer = []
        cl.user_session.set("audio_buffer", buffer)
        
    # Safely extract bytes regardless of the Chainlit version's internal data structures
    if isinstance(chunk, dict):
        buffer.append(chunk.get("data", b""))
    elif hasattr(chunk, "data"):
        buffer.append(chunk.data)
    else:
        buffer.append(chunk)

# The *args and **kwargs make this immune to signature crashes
@cl.on_audio_end
async def on_audio_end(*args, **kwargs): 
    print("--- [AUDIO EVENT] Microphone Stopped! ---")
    chunks = cl.user_session.get("audio_buffer")
    
    if not chunks:
        print("[!] ERROR: Audio buffer is completely empty.")
        await cl.Message(content="⚠️ No audio was captured. Please check your microphone permissions.").send()
        return
        
    print(f"[AUDIO EVENT] Stitching {len(chunks)} audio chunks together...")
    
    try:
        async with cl.Step(name="Transcribing Audio...") as step:
            # 1. Convert chunks to WAV
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b"".join(chunks))
                
            wav_io.seek(0)
            audio_bytes = wav_io.read()
            
            print(f"[AUDIO EVENT] Sending {len(audio_bytes)} bytes to Groq Whisper...")
            
            # 2. Transcribe with Groq
            transcription = agent.llm.client.audio.transcriptions.create(
                file=("voice_memo.wav", audio_bytes),
                model="whisper-large-v3",
                response_format="text"
            )
            
            user_input = transcription.strip()
            print(f"[AUDIO EVENT] Groq Transcribed: '{user_input}'")
            
            if not user_input:
                step.output = "Failed to hear anything."
                await cl.Message(content="⚠️ I couldn't hear any words. Please try speaking closer to the mic.").send()
                return

            step.output = f'"{user_input}"'
            await cl.Message(content=f"🎤 *Heard:* {user_input}").send()

        # 3. Check for manual commands
        if "download" in user_input.lower() or "export" in user_input.lower():
            await cl.Message(content="To download, please type the command instead of speaking it.").send()
            return

        # 4. Pass the text to Maestro
        async with cl.Step(name="Agent Reasoning") as step:
            bot_response = agent.process_user_input(user_input)
            step.output = "Analysis complete."

        # 5. Output rendering
        current_plan = agent.state.get_current_plan()
        final_output = f"{bot_response}\n\n"
        
        if "general question" not in bot_response.lower() and "Account Plan Research Agent" not in bot_response:
            final_output += format_plan_to_markdown(current_plan)

        await cl.Message(content=final_output).send()

    except Exception as e:
        # If ANYTHING fails, it will print the exact line of code that caused the crash in the terminal
        print(f"\n[!!!] CRITICAL AUDIO ERROR: {e}")
        traceback.print_exc() 
        await cl.Message(content=f"⚠️ A system error occurred while processing the audio: {str(e)}").send()