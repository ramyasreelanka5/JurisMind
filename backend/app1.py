import os
import traceback
from dotenv import load_dotenv
from typing import List, Optional

# --- Environment Variable Loading ---
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# --- FastAPI Imports ---
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel

# --- Core Logic Imports ---
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from PIL import Image
import speech_recognition as sr
from pydub import AudioSegment

# --- LangChain Imports ---
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document as LCDocument
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- Initialization ---
app = FastAPI()

# --- CORS Configuration ---
# Allows the frontend (running on a different port) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables & Model Loading ---
llm_text, llm_vision, BASE_RETRIEVER, embeddings = None, None, None, None
try:
    print("🚀 Initializing models and vector store...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm_text = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    llm_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.3)

    print("Initializing local embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("✅ Local embedding model loaded.")
    
    db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
    BASE_RETRIEVER = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("✅ Models and vector store initialized successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR during initialization:")
    traceback.print_exc()
    llm_text, llm_vision, BASE_RETRIEVER, embeddings = None, None, None, None

SESSION_RETRIEVERS = {}
DEFAULT_SESSION_ID = "default"
if BASE_RETRIEVER:
    SESSION_RETRIEVERS[DEFAULT_SESSION_ID] = BASE_RETRIEVER

# --- Pydantic Models for Request Bodies ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage] = []
    session_id: str = DEFAULT_SESSION_ID

class ResetRequest(BaseModel):
    session_id: str = DEFAULT_SESSION_ID

# --- Helper Functions ---
def get_related_questions(user_question, bot_answer):
    related_prompt = f"""
    You are an Indian Penal Code (IPC) legal assistant.
    Based on the user's last question and your answer, suggest 4 short, clear related legal questions
    that the user might want to ask next.
    Only output the questions as a numbered list, without extra text.

    User Question: {user_question}
    Your Answer: {bot_answer}
    """
    try:
        response = llm_text.invoke(related_prompt)
        suggestions = response.content.strip().split("\n")
        return [q.strip("0123456789. ").strip() for q in suggestions if q.strip()]
    except Exception:
        return []

# --- API Endpoints ---

@app.post('/chat')
async def chat(request: ChatRequest):
    if not llm_text:
        return JSONResponse(status_code=500, content={"error": "LLM not initialized"})

    retriever = SESSION_RETRIEVERS.get(request.session_id, BASE_RETRIEVER)
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
    for message in request.chat_history:
        if message.role == 'user':
            memory.chat_memory.add_user_message(message.content)
        else:
            memory.chat_memory.add_ai_message(message.content)

    prompt_template = """
    <s>[INST] You are an expert legal chatbot. Your primary goal is to provide clear, accurate, and well-structured information.
    **Instructions:**
    - Analyze the user's question and the provided context carefully.
    - Structure your answer using Markdown for readability.
    - Use numbered or bulleted lists for enumerations (like legal clauses or steps).
    - Use **bold text** to highlight key legal terms, parties (like "landlord" and "tenant"), and important concepts.
    - Keep your tone professional and direct.
    CONTEXT: {context}
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    ANSWER (in Markdown format):
    </s>[INST]
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_text, memory=memory, retriever=retriever, combine_docs_chain_kwargs={'prompt': prompt}
    )

    try:
        result = qa_chain.invoke(input=request.question)
        answer = result.get("answer", "Sorry, I could not find an answer.")
        related_questions = get_related_questions(request.question, answer)
        return {"answer": answer, "related_questions": related_questions}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post('/upload_document')
async def upload_document(session_id: str = Form(DEFAULT_SESSION_ID), file: UploadFile = File(...)):
    if not embeddings:
        return JSONResponse(status_code=500, content={"error": "Embeddings model not initialized"})
    if not file:
        return JSONResponse(status_code=400, content={"error": "No file part"})
    if file.filename == '':
        return JSONResponse(status_code=400, content={"error": "No selected file"})

    try:
        raw_text = ""
        file_ext = file.filename.split('.')[-1].lower()
        file_content = await file.read()

        if file_ext == "pdf":
            reader = PdfReader(BytesIO(file_content))
            for page in reader.pages:
                raw_text += page.extract_text() or ""
        elif file_ext == "docx":
            doc = Document(BytesIO(file_content))
            for para in doc.paragraphs:
                raw_text += (para.text or "") + "\n"
        elif file_ext == "txt":
            raw_text = file_content.decode('utf-8')
        else:
            return JSONResponse(status_code=400, content={"error": "File type not supported"})

        if raw_text:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_text(raw_text)
            documents = [LCDocument(page_content=t) for t in texts]
            
            temp_vector_store = FAISS.from_documents(documents, embeddings)
            
            SESSION_RETRIEVERS[session_id] = temp_vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            return {"message": f"File '{file.filename}' processed successfully."}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Failed to process file: {str(e)}"})


@app.post('/reset')
async def reset(request: ResetRequest):
    SESSION_RETRIEVERS[request.session_id] = BASE_RETRIEVER
    return {"message": "Conversation reset successfully."}


@app.post('/chat_vision')
async def chat_vision(question: str = Form(...), image: UploadFile = File(...)):
    if not llm_vision:
        return JSONResponse(status_code=500, content={"error": "Vision LLM not initialized"})
    if not image or not question:
        return JSONResponse(status_code=400, content={"error": "Image and question are required"})

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes))
        prompt_parts = [HumanMessage(content=[
            {"type": "text", "text": f"Carefully analyze this image and answer the following legal or document-related question: {question}"},
            {"type": "image_url", "image_url": pil_image}
        ])]
        response = llm_vision.invoke(prompt_parts)
        return {"answer": response.content, "related_questions": []}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to process image query: {str(e)}"})


@app.post('/transcribe')
async def transcribe_audio(audio: UploadFile = File(...)):
    if not audio:
        return JSONResponse(status_code=400, content={"error": "No audio file part"})
    
    recognizer = sr.Recognizer()
    try:
        audio_content = await audio.read()
        # Browser sends webm, convert it to WAV for speech_recognition
        sound = AudioSegment.from_file(BytesIO(audio_content), format="webm")
        wav_io = BytesIO()
        sound.export(wav_io, format="wav")
        wav_io.seek(0)
        
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
        
        text = recognizer.recognize_google(audio_data)
        return {"transcription": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Could not transcribe audio: {str(e)}"})

if __name__ == '__main__':
    import uvicorn
    # Note: The command below is for running directly with 'python app1.py'
    # For development, it's better to run from the terminal with 'uvicorn app1:app --reload'
    uvicorn.run("app1:app", host='0.0.0.0', port=5001, reload=True)