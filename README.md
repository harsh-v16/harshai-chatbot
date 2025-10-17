# 🧠 HarshAI Chatbot

Welcome to **HarshAI Chatbot**, an advanced conversational AI built with **Streamlit** and **OpenAI API** — your intelligent companion for chatting, voice interactions, and PDF understanding! 💬📄🎤  

---

## 🚀 Live Demo  
👉 [**Try HarshAI on Streamlit Cloud**](https://harshai-chatbot-uzyk3lmwt3k6py4epf3dps.streamlit.app/)

---

## ✨ Features

### 🤖 General Chat Mode
- Talk naturally with your AI assistant.  
- Supports **voice input** (via microphone 🎤).  
- Choose between **GPT-4o** and **GPT-3.5 Turbo** models.  
- Customize the AI’s **personality and tone**!

### 📚 Chat with Your PDF
- Upload any PDF document and start asking questions instantly!  
- Uses **LangChain**, **FAISS**, and **OpenAI embeddings** for accurate retrieval.  
- Perfect for study notes, research papers, or work documents.  

### 🧩 Smart & Fast Interface
- Built entirely with **Streamlit** for a clean, responsive, and fast UI.  
- Stylish sidebar controls for easy mode switching.  
- One-click **clear chat** button to reset conversations.  

---

## 🧠 Tech Stack

| Category | Technology |
|-----------|-------------|
| Frontend | Streamlit |
| Backend | Python |
| AI Models | OpenAI GPT (Chat) + Whisper (Speech-to-Text) |
| Document Understanding | LangChain + FAISS + PyPDF2 |
| Environment | Python 3.11 |
| Deployment | Streamlit Cloud |

---
# ⚙️ Installation (Run Locally)

### If you want to run this project on your system:

## Clone the repository
git clone https://github.com/harsh-v16/harshai-chatbot.git

## Move into the project folder
cd harshai-chatbot

## Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

## Or for Mac/Linux
source venv/bin/activate

## Install dependencies
pip install -r requirements.txt

## Run the app
streamlit run appy.py
Then open your browser at http://localhost:8501 🎉

----
 
## 🔐 Environment Variable

Before running the app, create a .env file in your project folder and add your OpenAI key:

OPENAI_API_KEY=your_api_key_here

### ⚠️ Keep this key private – never share or upload it anywhere (especially GitHub)

---

##  📸 Preview

![HarshAI Screenshot](screenshot.png)

---

# 💡 Future Enhancements

### 🌍 Add multi-language support

### 🗣️ Integrate text-to-speech for AI voice replies

### 🧰 Add memory for longer conversations

### 💾 Allow saving chat sessions locally

---

# 👨‍💻 About the Developer

### Made with ❤️ by Harsh

"I’m an aspiring AI developer, learning by building cool projects like HarshAI Chatbot!" 🚀
