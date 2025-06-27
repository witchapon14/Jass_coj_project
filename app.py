import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
import re
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Jass Chatbot (สำนักงานยุติธรรม)",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
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
        vector_store_path = "<link to vector store>"
        if not os.path.exists(vector_store_path):
            st.error(f"Vector store not found at: {vector_store_path}. Please ensure it is available.")
            st.stop()
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
        results = vector_store.similarity_search(query, k=5)
        source_knowledge = "\n".join([result.page_content for result in results])
        
        prompt_template = f"""บริบทต่อไปนี้คือข้อมูลที่ถูกดึงมาจากเอกสารเพื่อใช้ในการตอบคำถามของผู้ใช้:

    ------------------------------------------------
    บริบท:
    {source_knowledge}
    ------------------------------------------------
    กรุณาตอบคำถามต่อไปนี้โดยใช้เฉพาะข้อมูลจากบริบทด้านบนเท่านั้น:
    คำถาม: {query}

    ข้อกำหนดในการตอบคำถาม:
    - ตอบเป็นภาษาไทยด้วยถ้อยคำที่สุภาพและเป็นทางการ
    - ห้ามใช้คำหยาบ คำไม่สุภาพ หรือคำที่ไม่เหมาะสม
    - หากไม่พบคำตอบในข้อมูลด้านบน ให้ตอบว่า: "ขออภัยค่ะ ไม่พบข้อมูลเพียงพอที่จะตอบคำถามนี้จากฐานความรู้ที่มีอยู่"
    - หากคำถามไม่ชัดเจน ให้ตอบว่า: "คำถามไม่ชัดเจน กรุณาอธิบายเพิ่มเติมเพื่อให้สามารถให้ข้อมูลที่ถูกต้องได้"
    - ห้ามตอบคำถามที่ไม่เกี่ยวข้องกับข้อมูลในบริบทที่ให้มา
    - ห้ามใช้ข้อมูลนอกเหนือจากที่ให้มาในบริบท
    - ห้ามใช้ข้อมูลที่ไม่เกี่ยวข้องกับคำถาม
    - ห้ามใช้ข้อมูลที่ไม่ถูกต้องหรือไม่เป็นความจริง
    """
        return prompt_template.strip()
    except Exception as e:
        st.error(f"Error augmenting prompt: {e}")
        return query # Fallback to original query if there's an error

def remove_think_tags(text):
    """
    Removes <think> and </think> tags from the response text.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def generate_response(user_input: str):
    """
    Generates a response using the RAG model and cleans it.
    """
    if not user_input.strip():
        return "กรุณาป้อนคำถามค่ะ"

    try:
        full_prompt_content = augmented_prompt(user_input)
        
        chat_history = [
            SystemMessage(content="คุณคือผู้ช่วยอัจฉริยะด้านข้อมูลสำหรับสำนักงานยุติธรรม ซึ่งมีหน้าที่ตอบคำถามโดยอ้างอิงจากฐานความรู้ที่ได้รับมาเท่านั้น กรุณาให้ข้อมูลที่ถูกต้อง ครบถ้วน และใช้ภาษาที่สุภาพและเป็นทางการเสมอ"),
            HumanMessage(content=full_prompt_content)
        ]
        
        with st.spinner("กำลังค้นหาและสร้างคำตอบ..."):
            response = chat(messages=chat_history)
        
        cleaned_response = remove_think_tags(response.content)
        return cleaned_response
    
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)} กรุณาลองใหม่อีกครั้ง หรือติดต่อผู้ดูแลระบบหากปัญหายังคงอยู่"

# --- Streamlit UI ---
st.title("⚖️ ระบบตอบคำถามอัตโนมัติ (Jass Chatbot) - สำนักงานยุติธรรม")
st.markdown("""
ระบบนี้ถูกออกแบบมาเพื่อช่วยตอบคำถามที่เกี่ยวข้องกับข้อมูลของสำนักงานยุติธรรม 
โดยดึงข้อมูลจากฐานความรู้ที่ได้จัดเก็บไว้เท่านั้น เพื่อให้มั่นใจในความถูกต้องและความน่าเชื่อถือของข้อมูล
""")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("กรุณาป้อนคำถามของคุณ เช่น วันที่ ๓๑ มีนาคม พ.ศ. ๒๕๓๕ หมายถึงอะไร?"):
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

st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)