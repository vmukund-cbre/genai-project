import streamlit as st
import requests
import json

st.set_page_config(page_title="CBRE Real Estate Graph QA", page_icon="🏢")
st.title("🏢 CBRE Real Estate Graph QA Assistant")

# Initialize session state for conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_state" not in st.session_state:
    st.session_state.current_state = None
if "needs_clarification" not in st.session_state:
    st.session_state.needs_clarification = False

# Main question input
question = st.chat_input("Ask something about CBRE's real estate knowledge graph...")

# Clarification input (shown when needed)
clarification = None
if st.session_state.needs_clarification:
    clarification = st.chat_input("Provide additional details or clarification...")

RETRIEVER_ICONS = {
    "text2cypher": "📊 Cypher Query",
    "vector": "🔍 Semantic Search",
    "default": "🛰️ Unknown"
}

def process_response(data):
    """Process API response and update session state"""
    st.session_state.current_state = data
    
    if data.get("needs_clarification"):
        st.session_state.needs_clarification = True
        response_content = f"🤔 **Clarification Needed:**\n\n{data.get('clarification_request', 'I need more information to answer your question.')}"
    else:
        st.session_state.needs_clarification = False
        response_content = ""
        
        # Add LLM interpretation if available
        if data.get("llm_only_response"):
            response_content += f"🧠 **Initial Interpretation:**\n\n{data.get('llm_only_response')}\n\n"
        
        # Add formatted response
        if data.get("formatted_response"):
            response_content += f"📈 **Graph Insight:**\n\n{data.get('formatted_response')}\n\n"
        
        # Add Cypher query if available
        if data.get("cypher_generated"):
            response_content += f"🛠️ **Generated Cypher Query:**\n```cypher\n{data.get('cypher_generated')}\n```\n"
        
        # Add results summary
        if data.get("records_found", 0) > 0:
            response_content += f"📋 **Found {data.get('records_found')} matching records**\n"
        elif data.get("records_found") == 0 and not data.get("needs_clarification"):
            response_content += "📋 **No matching records found**\n"
    
    return response_content

# Handle main question
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    with st.spinner("Analyzing query..."):
        try:
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": question},
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            
            response_content = process_response(data)
            st.session_state.chat_history.append({"role": "assistant", "content": response_content})
            
        except requests.exceptions.RequestException as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"❌ API request failed: {e}"})
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"❌ Unexpected error: {e}"})

# Handle clarification
if clarification and st.session_state.current_state:
    st.session_state.chat_history.append({"role": "user", "content": f"Clarification: {clarification}"})
    
    with st.spinner("Processing clarification..."):
        try:
            response = requests.post(
                "http://localhost:8000/clarify",
                json={
                    "clarification": clarification,
                    "previous_state": st.session_state.current_state
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            response_content = process_response(data)
            st.session_state.chat_history.append({"role": "assistant", "content": response_content})
            
        except requests.exceptions.RequestException as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"❌ API request failed: {e}"})
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"❌ Unexpected error: {e}"})

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Add a button to clear conversation
if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.session_state.current_state = None
    st.session_state.needs_clarification = False
    st.rerun()
