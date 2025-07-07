# Jass Chatbot (ระบบตอบคำถามอัตโนมัติ - สำนักงานยุติธรรม)

## ⚖️ Project Overview

This project is an intelligent chatbot application designed for the Ministry of Justice of Thailand (สำนักงานยุติธรรม). It serves as an AI-powered assistant to answer questions based on a specialized knowledge base. The application is built using Python with Streamlit for the user interface and leverages a Retrieval-Augmented Generation (RAG) architecture for providing accurate and context-aware responses in Thai.

The core of the chatbot relies on a local vector store for retrieving relevant information and a high-speed Large Language Model (LLM) via the Groq API to generate human-like, formal responses.

### Features

- **Interactive Chat Interface:** A user-friendly web interface built with Streamlit.
- **Chat History:** Saves and allows users to revisit previous conversations.
- **Retrieval-Augmented Generation (RAG):** Enhances response accuracy by retrieving relevant context from a local FAISS vector store before generating an answer.
- **High-Performance LLM:** Utilizes the Groq API for fast and efficient language model inference.
- **Thai Language Support:** Specifically designed to understand and respond in polite, formal Thai.
- **Dockerized Deployment:** Comes with a `Dockerfile` for easy containerization and deployment.

## 🚀 Tech Stack

- **Application Framework:** [Streamlit](https://streamlit.io/)
- **LLM Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM Provider:** [Groq](https://groq.com/)
- **Embeddings Model:** [Ollama](https://ollama.com/) with `bge-m3`
- **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **Deployment:** [Docker](https://www.docker.com/)

## ⚙️ Setup and Installation

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.9 or later
- [Docker](https://www.docker.com/get-started) (for containerized deployment)
- Access to the Groq API and a valid API key.

### Local Development

1.  **Clone the Repository:**
    ```sh
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a Virtual Environment:**
    It's recommended to use a virtual environment to manage project dependencies.
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Install all the required Python packages using the `requirements.txt` file.
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the root directory of the project and add your Groq API key:
    ```env
    GROQ_API_KEY="your_groq_api_key_here"
    ```
    The application uses `python-dotenv` to load this key automatically.

5.  **Prepare the Vector Store:**
    The application requires a local FAISS vector store. Make sure the directory named `vector_store1` exists in the root of the project and contains the necessary `index.faiss` and `index.pkl` files.

6.  **Run the Application:**
    Once the setup is complete, you can run the Streamlit application with the following command:
    ```sh
    streamlit run app.py
    ```
    The application will be available at `http://localhost:8501`.

## 🐳 Docker Deployment

The project includes a `Dockerfile` for easy containerization.

1.  **Build the Docker Image:**
    From the root directory of the project, run the following command to build the image:
    ```sh
    docker build -t jass-chatbot .
    ```

2.  **Run the Docker Container:**
    Run the container, making sure to pass the `GROQ_API_KEY` as an environment variable and map the port.
    ```sh
    docker run -p 8501:8501 \
           -e GROQ_API_KEY="your_groq_api_key_here" \
           --name jass-chatbot-container \
           jass-chatbot
    ```
    The application will be accessible at `http://localhost:8501`.
