# 🏛️ JurisMind - AI-Powered Legal Assistant

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.1+-61DAFB.svg)](https://reactjs.org/)


*An intelligent legal chatbot powered by LLMs and RAG to provide accurate legal information based on Indian law*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API Documentation](#-api-endpoints) • [Tech Stack](#-tech-stack)

</div>

---

## 📋 Overview

**JurisMind** is a sophisticated AI-powered legal assistance platform designed to help users navigate complex legal documents, understand legal concepts, and get accurate information about Indian laws including the Indian Penal Code (IPC), Constitution of India, Companies Act, Labour Laws, and more.

### 🎯 Key Capabilities

- **💬 Intelligent Q&A**: Ask legal questions in natural language and receive accurate, well-structured answers
- **📄 Document Processing**: Upload and analyze legal documents (PDF, DOCX, TXT)
- **🖼️ Image OCR**: Extract and analyze text from legal document images
- **🎤 Voice Input**: Ask questions using voice recordings
- **🌍 Multi-Language Support**: Translate legal information to multiple languages
- **💾 Conversation History**: Save and manage chat conversations with authentication
- **🔍 RAG-Powered Search**: Leverage vector embeddings for context-aware legal information retrieval

---

## ✨ Features

### 🤖 AI-Powered Features
- **LLM Integration**: Powered by Groq's LLaMA 3.3 70B model for accurate legal responses
- **Vector Search**: FAISS-based similarity search for relevant legal context
- **Related Questions**: AI-generated follow-up questions based on conversation context
- **Smart Summarization**: Automatic conversation title generation

### 📱 User Experience
- **Modern UI**: Clean, responsive React frontend with dark/light theme support
- **Authentication**: Secure user registration and login with session management
- **Conversation Management**: Create, view, and delete chat histories
- **File Upload**: Support for multiple document formats (PDF, DOCX, TXT)
- **Voice Recording**: Speech-to-text transcription for hands-free interaction

### 🔒 Security & Privacy
- **Password Hashing**: Secure password storage using industry-standard hashing
- **Token-Based Auth**: JWT-like session tokens with expiration
- **User Isolation**: Each user's conversations are private and isolated

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **FastAPI** | Modern async web framework (recommended) |
| **Flask** | Alternative lightweight web framework |
| **LangChain** | LLM orchestration and RAG pipeline |
| **Groq API** | LLaMA 3.3 70B model inference |
| **FAISS** | Vector similarity search |
| **HuggingFace** | Sentence embeddings (all-MiniLM-L6-v2) |
| **SQLite** | User and conversation storage |
| **Tesseract OCR** | Image text extraction |
| **Deep Translator** | Multi-language translation |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 19.1** | UI framework |
| **Axios** | HTTP client |
| **React Markdown** | Markdown rendering |
| **React Syntax Highlighter** | Code syntax highlighting |
| **CSS3** | Styling and animations |

---

## 📦 Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **Node.js 16+ and npm** ([Download](https://nodejs.org/))
- **Tesseract OCR** ([Download for Windows](https://github.com/UB-Mannheim/tesseract/wiki))
- **Git** ([Download](https://git-scm.com/downloads))

### 🔧 Setup Instructions

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ramyasreelanka5/JurisMind.git
cd JurisMind
```

#### 2️⃣ Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt
```

**Configure Environment Variables:**

Create a `.env` file in the `backend` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

> 💡 **Get API Keys:**
> - Groq API: [https://console.groq.com](https://console.groq.com)
> - Google API: [https://ai.google.dev](https://ai.google.dev)

**Update Tesseract Path (if needed):**

If Tesseract is installed in a different location, update the path in:
- `backend/app.py` (line 43)
- `backend/app1.py` (line 40)

#### 3️⃣ Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install Node.js dependencies
npm install
```

---

## 🚀 Usage

### Running the Application

You need to run both the **backend** and **frontend** servers simultaneously.

#### Option 1: FastAPI Backend (Recommended ⭐)

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn app1:app --host 0.0.0.0 --port 5001 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

#### Option 2: Flask Backend (Alternative)

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### 🌐 Access the Application

- **Frontend UI**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:5001](http://localhost:5001)
- **API Docs** (FastAPI only): [http://localhost:5001/docs](http://localhost:5001/docs)

---

## 📚 API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register` | Register a new user |
| `POST` | `/login` | Login and receive auth token |
| `POST` | `/logout` | Invalidate auth token |
| `GET` | `/verify` | Verify token validity |

### Chat & AI

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a text question to the chatbot |
| `POST` | `/chat_vision` | Send an image with OCR analysis |
| `POST` | `/transcribe` | Transcribe audio to text |
| `POST` | `/upload_document` | Upload legal documents for RAG |
| `POST` | `/reset` | Reset conversation context |

### Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/conversations` | Get all user conversations |
| `GET` | `/conversations/{id}` | Get specific conversation messages |
| `POST` | `/conversations/new` | Create new conversation |
| `DELETE` | `/conversations/{id}` | Delete a conversation |

### Languages (FastAPI only)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/languages` | Get supported languages for translation |

---

## 📁 Project Structure

```
JurisMind/
│
├── backend/
│   ├── app.py                    # Flask implementation
│   ├── app1.py                   # FastAPI implementation (recommended)
│   ├── requirements.txt          # Python dependencies
│   ├── translation_utils.py      # Multi-language support
│   ├── ingestion.py              # Document ingestion script
│   ├── .env                      # Environment variables (create this)
│   ├── conversations.db          # SQLite database (auto-generated)
│   ├── LEGAL-DATA/               # Legal document corpus
│   │   ├── ipc_act.pdf
│   │   ├── COI.pdf
│   │   ├── CompaniesAct2013.pdf
│   │   └── ...
│   └── my_vector_store/          # FAISS vector database
│       ├── index.faiss
│       └── index.pkl
│
├── frontend/
│   ├── public/                   # Static assets
│   ├── src/
│   │   ├── App.js                # Main application component
│   │   ├── AuthModal.js          # Authentication modal
│   │   ├── Sidebar.js            # Conversation sidebar
│   │   ├── ThemeToggle.js        # Dark/Light theme toggle
│   │   ├── useAudioRecorder.js   # Audio recording hook
│   │   └── ...
│   ├── package.json              # Node dependencies
│   └── package-lock.json
│
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

----

## 🔮 Future Enhancements

- [ ] Multi-document comparison
- [ ] Case law search integration
- [ ] Legal document generation
- [ ] Advanced citation tracking
- [ ] Email notifications
- [ ] Export conversations as PDF
- [ ] Voice output (text-to-speech)
- [ ] Mobile application

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---




---

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **Groq** for LLM inference
- **HuggingFace** for embeddings
- **Indian Legal Documents** for the knowledge base
- **Open Source Community** for amazing tools and libraries

---

## 📞 Support

If you have any questions or need help, please:
- Open an issue on [GitHub](https://github.com/ramyasreelanka5/JurisMind/issues)
- Contact: [Your Email Here]

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ and ⚖️

</div>
