import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import re
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import datetime # Import for generating unique chat IDs

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Jass Chatbot (สำนักงานยุติธรรม)",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded" # Keep sidebar expanded by default for easier navigation
)

# --- Load Environment Variables ---
load_dotenv()
groq_api = os.getenv('GROQ_API_KEY')
if not groq_api:
    st.error("GROQ_API_KEY environment variable not set. Please set it to proceed.")
    st.stop()

# --- Embeddings and Vector Store Loading ---
@st.cache_resource
def load_resources():
    try:
        embeddings = OllamaEmbeddings(model="bge-m3")
        # Adjust this path for the deployment environment in the Ministry of Justice
        # For demonstration, keeping the original path but it should be configured appropriately.
        # Make sure the vector store directory exists and contains FAISS files.
        vector_store_path = ""
        if not os.path.exists(vector_store_path):
            st.error(f"Vector store not found at: {vector_store_path}. Please ensure it is available.")
            st.stop()
        # Ensure allow_dangerous_deserialization=True is used if loading from untrusted sources,
        # but for internal use with known data, it might be acceptable.
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        return embeddings, vector_store
    except Exception as e:
        st.error(f"Error loading resources: {e}. Please check the vector store path and model availability.")
        st.stop()

embeddings, vector_store = load_resources()

# --- Chat Model Initialization ---
@st.cache_resource
def init_groq_chat_model(api_key):
    return ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b", streaming=True, api_key=api_key)

chat = init_groq_chat_model(groq_api)

# --- RAG Functions ---
def augmented_prompt(query: str):
    """
    Generates an augmented prompt by retrieving relevant information from the vector store.
    """
    try:
        results = vector_store.similarity_search(query, k=3)
        print(results)

        source_knowledge = "\n".join([result.page_content for result in results])

        prompt_template = f"""The following context has been retrieved from documents to assist in answering the user's question:

    ------------------------------------------------
    Context:
    {source_knowledge}
    ------------------------------------------------
    Please answer the following question using only the information provided in the above context:
    Question: {query}

    Guidelines for answering:
    - Respond in polite and formal Thai language.

    """
        
    #         - Respond in polite and formal Thai language.
    # - Do not use offensive, impolite, or inappropriate words.
    # - If the answer cannot be found in the above context, reply: "ขออภัยค่ะ ไม่พบข้อมูลเพียงพอที่จะตอบคำถามนี้จากฐานความรู้ที่มีอยู่"
    # - If the question is unclear, reply: "คำถามไม่ชัดเจน กรุณาอธิบายเพิ่มเติมเพื่อให้สามารถให้ข้อมูลที่ถูกต้องได้"
    # - Do not answer questions unrelated to the provided context.
    # - Do not use any information outside of the given context.
    # - Do not use irrelevant or incorrect information.
        return prompt_template.strip()
    except Exception as e:
        st.error(f"Error augmenting prompt: {e}")
        return query # Fallback to original query if there's an error

def remove_think_tags(text):
    """
    Removes <think> and </think> tags from the response text.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # return text

def generate_response(user_input: str):
    """
    Generates a response using the RAG model and cleans it.
    """
    if not user_input.strip():
        return "กรุณาป้อนคำถามค่ะ"

    try:
        full_prompt_content = augmented_prompt(user_input)
        
        # System message for the model's role
        # Note: The system message is placed at the beginning of the chat_history.
        # It sets the persona for the entire conversation within this chat session.
        chat_history_for_model = [
            SystemMessage(content="คุณคือผู้ช่วยอัจฉริยะด้านข้อมูลสำหรับสำนักงานยุติธรรม ซึ่งมีหน้าที่ตอบคำถามโดยอ้างอิงจากฐานความรู้ที่ได้รับมาเท่านั้น กรุณาให้ข้อมูลที่ถูกต้อง ครบถ้วน และใช้ภาษาที่สุภาพและเป็นทางการเสมอ"),
            HumanMessage(content=full_prompt_content)
        ]
        
        with st.spinner("กำลังค้นหาและสร้างคำตอบ..."):
            # Pass the constructed chat history to the model
            response = chat(messages=chat_history_for_model)
        
        cleaned_response = remove_think_tags(response.content)
        return cleaned_response
    
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)} กรุณาลองใหม่อีกครั้ง หรือติดต่อผู้ดูแลระบบหากปัญหายังคงอยู่"

# --- Session State Initialization ---
if "chats" not in st.session_state:
    st.session_state.chats = {} # Stores all chat sessions: {chat_id: [{"role": ..., "content": ...}, ...]}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None # ID of the currently active chat
if "messages" not in st.session_state:
    st.session_state.messages = [] # Messages for the currently active chat

def new_chat():
    """Resets the current chat and saves the old one if it had messages."""
    if st.session_state.current_chat_id and st.session_state.messages:
        # Save the current chat before starting a new one
        if st.session_state.current_chat_id not in st.session_state.chats:
             # If it's a brand new chat, give it a title based on the first user message or a timestamp
             first_user_msg = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), f"สนทนาเมื่อ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
             st.session_state.chats[st.session_state.current_chat_id] = {
                 "title": first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg,
                 "messages": st.session_state.messages
             }
        else:
            # Update existing chat
            st.session_state.chats[st.session_state.current_chat_id]["messages"] = st.session_state.messages

    # Generate a new unique chat ID
    new_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(len(st.session_state.chats))
    st.session_state.current_chat_id = new_id
    st.session_state.messages = [] # Clear messages for the new chat
    st.rerun() # Rerun to clear chat display

def load_chat(chat_id):
    """Loads a specific chat from history."""
    if st.session_state.current_chat_id and st.session_state.messages:
         # Save the current chat before loading a new one
        if st.session_state.current_chat_id not in st.session_state.chats:
            first_user_msg = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), f"สนทนาเมื่อ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.session_state.chats[st.session_state.current_chat_id] = {
                "title": first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg,
                "messages": st.session_state.messages
            }
        else:
            st.session_state.chats[st.session_state.current_chat_id]["messages"] = st.session_state.messages

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.chats[chat_id]["messages"]
    st.rerun() # Rerun to display loaded chat

# --- Streamlit UI ---
st.title("⚖️ ระบบตอบคำถามอัตโนมัติ (Jass Chatbot) - สำนักงานยุติธรรม")
st.markdown("""
ระบบนี้ถูกออกแบบมาเพื่อช่วยตอบคำถามที่เกี่ยวข้องกับข้อมูลของสำนักงานยุติธรรม 
โดยดึงข้อมูลจากฐานความรู้ที่ได้จัดเก็บไว้เท่านั้น เพื่อให้มั่นใจในความถูกต้องและความน่าเชื่อถือของข้อมูล
""")

# --- Sidebar for Navigation ---
with st.sidebar:
    st.header("เมนู")
    if st.button("💬 สนทนาใหม่", use_container_width=True):
        new_chat()

    st.markdown("---")
    st.header("ประวัติการสนทนา")
    if not st.session_state.chats:
        st.info("ยังไม่มีประวัติการสนทนา")
    else:
        # Display chat history, sorted by recency
        sorted_chats = sorted(st.session_state.chats.items(), 
                              key=lambda item: item[0], reverse=True)
        for chat_id, chat_data in sorted_chats:
            is_current_chat = (chat_id == st.session_state.current_chat_id)
            button_label = chat_data.get("title", f"สนทนา ID: {chat_id}")
            if st.button(button_label, key=f"chat_btn_{chat_id}", use_container_width=True, 
                         type="primary" if is_current_chat else "secondary"):
                load_chat(chat_id)

# --- Main Chat Interface ---

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
# User input
if prompt := st.chat_input("กรุณาป้อนคำถามของคุณ เช่น วันที่ ๓๑ มีนาคม พ.ศ. ๒๕๓๕ หมายถึงอะไร?"):
    # If starting a new chat without explicitly clicking "New Chat" button,
    # generate a current_chat_id for it and initialize its entry in st.session_state.chats
    if not st.session_state.current_chat_id:
        new_chat_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_0"
        st.session_state.current_chat_id = new_chat_id
        # Initialize the chat entry *before* appending messages or trying to access its title
        st.session_state.chats[st.session_state.current_chat_id] = {"title": "", "messages": []}

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Update chat title if it's the first user message of this particular chat session
    # We need to ensure that st.session_state.current_chat_id is a valid key in st.session_state.chats
    # and that its 'title' is empty before attempting to set it.
    if st.session_state.current_chat_id in st.session_state.chats and \
       not st.session_state.chats[st.session_state.current_chat_id]["title"] and \
       len([m for m in st.session_state.messages if m["role"] == "user"]) == 1:
        first_user_msg = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), f"สนทนาเมื่อ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.session_state.chats[st.session_state.current_chat_id]["title"] = first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
# Custom CSS for better styling (optional)
st.markdown("""
<style>
    /* Adjust main content padding for better use of space */
    .block-container {
        padding-top: 1rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 1rem;
    }
    /* Style for sidebar buttons */
    .stButton > button {
        border-radius: 0.5rem;
        border: 1px solid #d3d3d3;
        font-weight: normal;
    }
    .stButton > button.secondary {
        background-color: #f0f2f6; /* Light gray for non-selected */
        color: #333;
    }
    .stButton > button.primary {
        background-color: #007bff; /* Blue for selected/active */
        color: white;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
    /* Chat messages styling */
    .st-emotion-cache-1c7y2qn { /* This class might change with Streamlit updates, verify with dev tools */
        background-color: #e0f2f1; /* Light teal for assistant messages */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .st-emotion-cache-kqswzi { /* This class might change with Streamlit updates, verify with dev tools */
        background-color: #e3f2fd; /* Light blue for user messages */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)
