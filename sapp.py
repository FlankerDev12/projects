# Copyright 2025 Quantum Mechanics Assistant
# Enhanced Edition with Better Error Handling

from htbuilder.units import rem, px
from htbuilder import div, styles
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import datetime
import textwrap
import time
import requests
import json

import streamlit as st

st.set_page_config(
    page_title="Quantum Mechanics Assistant", 
    page_icon="Ψ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Refined Custom CSS - Clean & Professional

st.markdown("""
<style>
    /* Clean dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Mode selector cards */
    .mode-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 2px solid transparent;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .mode-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4);
        border-color: #6366f1;
    }
    
    .mode-card.active {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-color: #8b5cf6;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5);
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        font-size: 0.95rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 10px;
    }
    
    /* Error messages */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #ef4444;
    }
    
    /* Success messages */
    .element-container .stSuccess {
        border-radius: 12px;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuration

executor = ThreadPoolExecutor(max_workers=5)

API_RAG = "http://127.0.0.1:8000/rag"
API_AI = "http://127.0.0.1:8000/troubleshoot"
API_HEALTH = "http://127.0.0.1:8000/health"

HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
RAG_CONTEXT_LEN = 10

DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

INSTRUCTIONS = textwrap.dedent("""
    - You are a helpful AI assistant focused on answering questions about
      Quantum Mechanics, quantum physics, and related mathematical concepts.
    - Use context and history to provide a coherent answer.
    - Use markdown such as headers (starting with ##), code blocks, bullet
      points, indentation for sub bullets, and backticks for inline code.
    - Be clear and educational. Provide examples and derivations when helpful.
    - Include relevant equations using LaTeX notation.
""")

SUGGESTIONS = {
    "🌊 Wave-Particle Duality": "Explain wave-particle duality and its experimental evidence",
    "⚛️ Schrödinger Equation": "Explain the time-dependent Schrödinger equation",
    "🔗 Quantum Entanglement": "What is quantum entanglement and why is it important?",
    "📊 Uncertainty Principle": "Explain Heisenberg's uncertainty principle",
    "🚇 Quantum Tunneling": "How does quantum tunneling work?",
}

# -----------------------------------------------------------------------------
# Session State Initialization

def init_session_state():
    """Initialize all session state variables."""
    if 'rag_messages' not in st.session_state:
        st.session_state.rag_messages = []
    if 'ai_messages' not in st.session_state:
        st.session_state.ai_messages = []
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'last_request_time' not in st.session_state:
        st.session_state.last_request_time = None
    if 'api_status' not in st.session_state:
        st.session_state.api_status = None
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None

# -----------------------------------------------------------------------------
# Helper Functions

TaskInfo = namedtuple("TaskInfo", ["name", "function", "args"])
TaskResult = namedtuple("TaskResult", ["name", "result"])

def check_api_health():
    """Check if the API server is running and healthy."""
    try:
        response = requests.get(API_HEALTH, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, f"API returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API server"
    except requests.exceptions.Timeout:
        return False, "API health check timed out"
    except Exception as e:
        return False, str(e)

def build_prompt(**kwargs):
    """Builds a prompt string with the kwargs as HTML-like tags."""
    prompt = []
    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")
    return "\n".join(prompt)

def history_to_text(chat_history):
    """Converts chat history into a string."""
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)

def get_response(prompt, mode="ai"):
    """Get response from the appropriate API endpoint with enhanced error handling."""
    try:
        # Record request time
        st.session_state.last_request_time = datetime.datetime.now()
        
        # Select endpoint based on mode
        if mode == "rag":
            url = API_RAG
            param_key = "query"
        else:
            url = API_AI
            param_key = "issue"
        
        # Make API request
        if DEBUG_MODE:
            st.info(f"🔍 Calling {url} with {param_key}='{prompt[:50]}...'")
        
        response = requests.get(url, params={param_key: prompt}, timeout=60)
        
        # Check response status
        if response.status_code != 200:
            error_msg = f"⚠️ API Error (Status {response.status_code})"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f"\n\n**Details:** {error_data['error']}"
            except:
                error_msg += f"\n\n**Response:** {response.text[:200]}"
            
            st.session_state.last_error = error_msg
            return error_msg
        
        # Parse JSON response
        try:
            data = response.json()
        except json.JSONDecodeError:
            error_msg = f"⚠️ Invalid JSON response from API\n\n**Raw response:** {response.text[:500]}"
            st.session_state.last_error = error_msg
            return error_msg
        
        # Debug: Show response structure
        if DEBUG_MODE:
            st.write("**📦 API Response Structure:**")
            st.json(data)
        
        # Extract response from various possible keys
        possible_keys = [
            "response",       # Primary key from new API
            "answer",         # Alternative
            "troubleshooting_guide",  # Legacy AI mode
            "context",        # RAG mode
            "result",
            "output",
            "text",
            "message"
        ]
        
        for key in possible_keys:
            if key in data and data[key]:
                content = data[key]
                # Check if content is not empty
                if isinstance(content, str) and content.strip():
                    st.session_state.last_error = None  # Clear any previous errors
                    return content
                elif isinstance(content, dict) or isinstance(content, list):
                    return json.dumps(content, indent=2)
        
        # Check for error in response
        if "error" in data:
            error_msg = f"⚠️ API Error: {data['error']}"
            st.session_state.last_error = error_msg
            return error_msg
        
        # If no known key found, show available keys
        error_msg = f"⚠️ Unexpected response format.\n\n**Available keys:** {', '.join(data.keys())}\n\n**Full response:**\n```json\n{json.dumps(data, indent=2)}\n```"
        st.session_state.last_error = error_msg
        return error_msg
            
    except requests.exceptions.Timeout:
        error_msg = "⚠️ Request timed out after 60 seconds. The model might be loading or the server is overloaded. Please try again."
        st.session_state.last_error = error_msg
        return error_msg
    
    except requests.exceptions.ConnectionError:
        error_msg = f"⚠️ Cannot connect to API server at `{url}`\n\n**Troubleshooting:**\n1. Make sure FastAPI server is running: `python your_api_file.py`\n2. Check if the URL is correct\n3. Verify no firewall is blocking the connection"
        st.session_state.last_error = error_msg
        return error_msg
    
    except Exception as e:
        error_msg = f"⚠️ Unexpected error: {str(e)}\n\n**Type:** {type(e).__name__}"
        st.session_state.last_error = error_msg
        if DEBUG_MODE:
            import traceback
            error_msg += f"\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"
        return error_msg

def export_chat_to_markdown(messages, mode):
    """Export chat history as Markdown."""
    markdown = f"# Quantum Mechanics Assistant - {mode.upper()} Mode\n\n"
    markdown += f"*Exported on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    markdown += "---\n\n"
    
    for msg in messages:
        role = msg['role'].title()
        content = msg['content']
        markdown += f"## {role}\n\n{content}\n\n"
    
    return markdown

# -----------------------------------------------------------------------------
# UI Components

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Ψ Quantum Mechanics Assistant</h1>
        <p class="main-subtitle">Your AI-powered guide to quantum physics</p>
    </div>
    """, unsafe_allow_html=True)

def render_mode_selector():
    """Render mode selection interface with API status check."""
    # Check API health before showing modes
    with st.spinner("Checking API status..."):
        api_healthy, api_info = check_api_health()
    
    if not api_healthy:
        st.error(f"⚠️ **API Server Not Available**\n\n{api_info}\n\nPlease start your FastAPI server before using the assistant.")
        st.code("python your_fastapi_file.py", language="bash")
        
        if st.button("🔄 Retry Connection", use_container_width=True):
            st.rerun()
        return
    
    # Show success message
    st.success("✅ Connected to API server")
    
    st.markdown("### 🎯 Choose Your Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📚 RAG Mode\n\nDocument-based answers", use_container_width=True, key="rag_btn"):
            st.session_state.mode = "rag"
            st.rerun()
    
    with col2:
        if st.button("🤖 AI Mode\n\nDirect AI reasoning", use_container_width=True, key="ai_btn"):
            st.session_state.mode = "ai"
            st.rerun()
    
    st.markdown("---")
    
    # Mode descriptions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📚 RAG Mode**
        - Searches quantum mechanics PDFs
        - Returns document-based answers
        - Best for textbook-style questions
        - Includes source citations
        """)
    
    with col2:
        st.markdown("""
        **🤖 AI Mode**
        - Uses LLM reasoning directly
        - More flexible and conversational
        - Can explain concepts creatively
        - Handles complex reasoning
        """)
    
    st.markdown("---")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2.5rem;">⚡</div>
            <h4>Fast Responses</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Get instant answers to your quantum questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2.5rem;">🎓</div>
            <h4>Educational</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Learn quantum mechanics step by step</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2.5rem;">🔬</div>
            <h4>Research Ready</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Access detailed quantum physics knowledge</p>
        </div>
        """, unsafe_allow_html=True)

def render_chat_interface(mode):
    """Render the chat interface for the selected mode."""
    messages = st.session_state.get(f"{mode}_messages", [])
    
    # Show mode badge
    mode_color = "#8b5cf6" if mode == "ai" else "#6366f1"
    mode_icon = "🤖" if mode == "ai" else "📚"
    st.markdown(f"""
    <div style="background: {mode_color}; padding: 0.5rem 1rem; border-radius: 8px; 
                display: inline-block; margin-bottom: 1rem; font-weight: 600;">
        {mode_icon} {mode.upper()} Mode Active
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show welcome message if no messages
    if not messages:
        st.info(f"💡 **Tip:** In {mode.upper()} mode, you can ask questions about quantum mechanics. Try one of the suggestions in the sidebar!")
    
    # Chat input
    if prompt := st.chat_input(f"Ask a quantum mechanics question... ({mode.upper()} mode)"):
        # Add user message
        messages.append({"role": "user", "content": prompt})
        st.session_state[f"{mode}_messages"] = messages
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = get_response(prompt, mode)
                st.markdown(response)
        
        # Add assistant message
        messages.append({"role": "assistant", "content": response})
        st.session_state[f"{mode}_messages"] = messages
        st.rerun()

def render_sidebar(mode):
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown(f"### ⚙️ {mode.upper()} Mode")
        st.markdown("---")
        
        # API Status indicator
        with st.expander("🌐 API Status", expanded=False):
            if st.button("Check Status", use_container_width=True):
                with st.spinner("Checking..."):
                    healthy, info = check_api_health()
                    if healthy:
                        st.success("✅ API is healthy")
                        st.json(info)
                    else:
                        st.error(f"❌ {info}")
        
        # Debug toggle
        debug_enabled = st.checkbox("🐛 Debug Mode", value=DEBUG_MODE)
        if debug_enabled != DEBUG_MODE:
            st.query_params.update({"debug": str(debug_enabled).lower()})
            st.rerun()
        
        # Message count
        message_count = len(st.session_state.get(f"{mode}_messages", []))
        st.metric("💬 Messages", message_count)
        
        # Show last error if exists
        if st.session_state.last_error and not debug_enabled:
            with st.expander("⚠️ Last Error", expanded=False):
                st.error(st.session_state.last_error)
        
        st.markdown("---")
        
        # API Configuration
        with st.expander("🔧 API Settings"):
            st.text_input("RAG API", value=API_RAG, key="api_rag_display", disabled=True)
            st.text_input("AI API", value=API_AI, key="api_ai_display", disabled=True)
            st.text_input("Health API", value=API_HEALTH, key="api_health_display", disabled=True)
            
            # Test specific endpoint
            if st.button("Test Current Mode", use_container_width=True):
                with st.spinner("Testing..."):
                    try:
                        test_url = API_RAG if mode == "rag" else API_AI
                        test_param = "query" if mode == "rag" else "issue"
                        response = requests.get(test_url, params={test_param: "test"}, timeout=10)
                        st.success(f"✅ Status: {response.status_code}")
                        st.json(response.json())
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state[f"{mode}_messages"] = []
            st.session_state.last_error = None
            st.rerun()
        
        # Export button
        if message_count > 0:
            if st.button("📤 Export Chat", use_container_width=True):
                messages = st.session_state.get(f"{mode}_messages", [])
                markdown_content = export_chat_to_markdown(messages, mode)
                filename = f"quantum_chat_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                st.download_button(
                    "📥 Download Markdown",
                    markdown_content,
                    filename,
                    use_container_width=True
                )
        
        # Switch mode button
        st.markdown("---")
        if st.button("🔄 Switch Mode", use_container_width=True):
            st.session_state.mode = None
            st.rerun()
        
        # Suggestions
        if message_count == 0:
            st.markdown("---")
            st.markdown("### 💡 Try asking:")
            for emoji_label, question in SUGGESTIONS.items():
                if st.button(emoji_label, use_container_width=True, key=f"sug_{emoji_label}"):
                    messages = st.session_state.get(f"{mode}_messages", [])
                    messages.append({"role": "user", "content": question})
                    
                    with st.spinner("Thinking..."):
                        response = get_response(question, mode)
                    
                    messages.append({"role": "assistant", "content": response})
                    st.session_state[f"{mode}_messages"] = messages
                    st.rerun()
        
        # Debug info
        if debug_enabled:
            st.markdown("---")
            st.markdown("### 🐛 Debug Info")
            st.json({
                "mode": mode,
                "messages": message_count,
                "api_rag": API_RAG,
                "api_ai": API_AI,
                "last_request": str(st.session_state.last_request_time),
                "last_error": st.session_state.last_error,
                "api_status": st.session_state.api_status
            })

# -----------------------------------------------------------------------------
# Main Application

def main():
    """Main application entry point."""
    init_session_state()
    
    # Render header
    render_header()
    
    # Check if mode is selected
    if st.session_state.mode is None:
        render_mode_selector()
    else:
        # Render sidebar and chat
        render_sidebar(st.session_state.mode)
        render_chat_interface(st.session_state.mode)

if __name__ == "__main__":
    main()