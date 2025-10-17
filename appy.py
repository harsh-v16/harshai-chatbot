# Description:
# A multi-functional Streamlit web application that serves as a versatile
# AI assistant. This app features two primary modes: a conversational AI
# with customizable personality and voice input, and a document analysis
# tool that allows users to ask questions about an uploaded PDF file.
#
# Technologies Used:
# - Streamlit: For the web application interface.
# - OpenAI API: For accessing GPT models (chat completion) and Whisper (speech-to-text).
# - LangChain: For the Retrieval-Augmented Generation (RAG) pipeline to chat with PDFs.
# - PyPDF2: For text extraction from PDF documents.
# - FAISS: For efficient similarity search in the vector store.
# ==============================================================================

# --- Core Libraries ---
import streamlit as st
import openai
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# --- Libraries for PDF "Chat with Document" Feature ---
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# --- 1. INITIAL SETUP AND CONFIGURATION ---

# Load environment variables from the .env file (for the OpenAI API key)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure the Streamlit page. This must be the first Streamlit command.
st.set_page_config(
    page_title="HarshAI Super-App",
    page_icon="🧠",
    layout="wide"
)

# --- 2. CUSTOM STYLING ---

# Inject custom CSS to style the voice input button for a more polished look.
st.markdown("""
<style>
    /* Targets the container of the mic recorder button for custom styling */
    div[data-testid="stVerticalBlock"]>div>div>div[data-testid="stButton"]>button {
        background-color: #4CAF50; color: white; border: none; border-radius: 8px;
        padding: 10px 24px; text-align: center; text-decoration: none;
        display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;
        transition-duration: 0.4s;
    }
    /* Adds a hover effect for better user experience */
    div[data-testid="stVerticalBlock"]>div>div>div[data-testid="stButton"]>button:hover {
        background-color: white; color: black; border: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR INTERFACE ---

# The sidebar holds the main controls for the application mode and settings.
with st.sidebar:
    st.header("🧠 HarshAI Super-App")
    
    # Mode selector allows the user to switch between the two main functionalities.
    app_mode = st.radio(
        "Choose App Mode",
        ["General Chat", "Chat with PDF"]
    )
    
    st.markdown("---") # Visual separator

    # --- Conditional UI for "General Chat" Mode ---
    if app_mode == "General Chat":
        st.info("Configure your general assistant.")
        # Allow user to select the OpenAI model for the chat
        selected_model = st.selectbox("Choose AI Model", ["gpt-4o", "gpt-3.5-turbo"], key="model")
        # Text area for the user to define the AI's personality (system prompt)
        system_prompt = st.text_area("The Sarcastic Robot", "I am an incredibly enthusiastic and positive sidekick! I use lots of exclamation points and emojis! I think every question is the best question I've ever heard and I'm super excited to help! ✨🚀", height=150, key="prompt")
        # Slider to control the creativity/randomness of the AI's responses (temperature)
        temperature = st.slider("Creativity", 0.0, 2.0, 0.7, 0.1, key="temp")

    # --- Conditional UI for "Chat with PDF" Mode ---
    elif app_mode == "Chat with PDF":
        st.info("Ask questions about your document.")
        # File uploader widget for PDF files
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        if pdf is not None:
            # Button to trigger the PDF processing
            if st.button("Process Document"):
                with st.spinner("Processing..."):
                    # Step 1: Extract text from the uploaded PDF
                    pdf_reader = PdfReader(pdf)
                    text = "".join(page.extract_text() for page in pdf_reader.pages)
                    
                    # Step 2: Split the extracted text into smaller, manageable chunks
                    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
                    chunks = text_splitter.split_text(text)
                    
                    # Step 3: Create embeddings and a FAISS vector store from the chunks
                    embeddings = OpenAIEmbeddings()
                    knowledge_base = FAISS.from_texts(chunks, embeddings)
                    
                    # Store the processed knowledge base in the session state for later use
                    st.session_state.knowledge_base = knowledge_base
                    st.success("Document processed!")

    st.markdown("---") # Visual separator

    # --- Common Sidebar Elements ---
    # Button to clear the chat history and reset the session
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        # Also clear the knowledge base if it exists
        if 'knowledge_base' in st.session_state:
            del st.session_state['knowledge_base']
        st.rerun() # Rerun the app to reflect the changes immediately

# --- 4. MAIN CHAT INTERFACE ---

st.title("HarshAI 🤖")

# Initialize the chat message history in Streamlit's session state if it doesn't exist.
# This ensures the conversation is not lost on each interaction.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all past messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. APPLICATION LOGIC (Based on selected mode) ---

# --- Logic for "General Chat" Mode ---
if app_mode == "General Chat":
    # Display the voice recorder component
    audio_info = mic_recorder(start_prompt="Click to Speak 🎤", stop_prompt="Recording...", key='recorder')
    
    # If audio is recorded, transcribe it using OpenAI's Whisper model
    if audio_info:
        audio_bytes = audio_info['bytes']
        # Temporarily save audio to a file to send to the API
        with open("audio.wav", "wb") as f: f.write(audio_bytes)
        with st.spinner("Transcribing..."):
            with open("audio.wav", "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
            # The transcribed text becomes the user's prompt
            user_prompt = transcript.text
    else:
        # If no audio, use the standard text input
        user_prompt = st.chat_input("What can I help you with?")

    # If there is a prompt (from voice or text), process it
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"): st.markdown(user_prompt)
        
        # Generate and display the AI's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # The full context sent to the API includes the system prompt and the entire chat history
                full_prompt = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                response = openai.chat.completions.create(model=selected_model, messages=full_prompt, temperature=temperature)
                ai_response = response.choices[0].message.content
                st.markdown(ai_response)
        # Add the AI's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

# --- Logic for "Chat with PDF" Mode ---
elif app_mode == "Chat with PDF":
    # Get user's question via the text input
    if user_question := st.chat_input("Ask a question about your PDF:"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"): st.markdown(user_question)

        # Check if a document has been processed and its knowledge base is available
        if 'knowledge_base' in st.session_state:
            knowledge_base = st.session_state.knowledge_base
            with st.spinner("Searching for answers..."):
                # Step 1: Find the most relevant document chunks based on the user's question
                docs = knowledge_base.similarity_search(user_question)
                
                # Step 2: Use a LangChain Question-Answering chain to get a response
                llm = ChatOpenAI(model_name="gpt-4o")
                chain = load_qa_chain(llm, chain_type="stuff")
                # The chain feeds the relevant chunks (docs) and the question to the AI
                response = chain.run(input_documents=docs, question=user_question)
                ai_response = response
        else:
            # If no document has been processed, inform the user
            ai_response = "Please upload and process a document first."

        # Add the AI's response to the session state and display it
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"): st.markdown(ai_response)