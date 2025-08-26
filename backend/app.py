import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from PIL import Image

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document as LCDocument
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain



from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Audio Processing Imports
import speech_recognition as sr
from pydub import AudioSegment





# --- Initialization ---
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Global Variables & Model Loading ---
# Load models once to be reused across requests
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm_text = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    llm_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.3)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
    BASE_RETRIEVER = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("✅ Models and vector store initialized successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR during initialization: {e}")
    # **FIX #1**: Ensure all models are None on failure
    llm_text, llm_vision, BASE_RETRIEVER = None, None, None

SESSION_RETRIEVERS = {}
DEFAULT_SESSION_ID = "default"
SESSION_RETRIEVERS[DEFAULT_SESSION_ID] = BASE_RETRIEVER


# --- Helper Functions ---
def get_related_questions(user_question, bot_answer):
    # This function remains the same as in your original code
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

# --- API Endpoints ---a

@app.route('/chat', methods=['POST'])
def chat():
    if not llm_text:
        return jsonify({"error": "LLM not initialized"}), 500

    data = request.json
    question = data.get('question')
    chat_history = data.get('chat_history', [])
    session_id = data.get('session_id', DEFAULT_SESSION_ID)

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Get the right retriever for the session (either base or from uploaded file)
    retriever = SESSION_RETRIEVERS.get(session_id, BASE_RETRIEVER)

    # Create a memory instance for this request
    memory = ConversationBufferWindowMemory(
        k=3, memory_key="chat_history", return_messages=True
    )
    for message in chat_history:
        if message['role'] == 'user':
            memory.chat_memory.add_user_message(message['content'])
        else:
            memory.chat_memory.add_ai_message(message['content'])

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
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question', 'chat_history']
    )


    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_text,
        memory=memory,
        retriever=retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    try:
        result = qa_chain.invoke(input=question)
        answer = result.get("answer", "Sorry, I could not find an answer.")
        related_questions = get_related_questions(question, answer)
        return jsonify({
            "answer": answer,
            "related_questions": related_questions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload_document', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    session_id = request.form.get('session_id', DEFAULT_SESSION_ID)

    try:
        raw_text = ""
        file_ext = file.filename.split('.')[-1].lower()
        
        if file_ext == "pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                raw_text += page.extract_text() or ""
        elif file_ext == "docx":
            doc = Document(BytesIO(file.read()))
            for para in doc.paragraphs:
                raw_text += (para.text or "") + "\n"
        elif file_ext == "txt":
            raw_text = file.read().decode('utf-8')

        if raw_text:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_text(raw_text)
            documents = [LCDocument(page_content=t) for t in texts]

            temp_vector_store = FAISS.from_documents(documents, embeddings)
            # Store the new retriever for this session
            SESSION_RETRIEVERS[session_id] = temp_vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            return jsonify({"message": f"File '{file.filename}' processed successfully."})

    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

    return jsonify({"error": "File type not supported"}), 400


@app.route('/reset', methods=['POST'])
def reset():
    session_id = request.json.get('session_id', DEFAULT_SESSION_ID)
    # Reset the session to use the base retriever
    SESSION_RETRIEVERS[session_id] = BASE_RETRIEVER
    return jsonify({"message": "Conversation reset successfully."})

@app.route('/chat_vision', methods=['POST'])
def chat_vision():
    """Handles chat queries that include an image."""
    if not llm_vision: return jsonify({"error": "Vision LLM not initialized"}), 500
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({"error": "Image and question are required"}), 400

    question = request.form['question']
    image_file = request.files['image']

    try:
        image = Image.open(image_file.stream)
        prompt_parts = [HumanMessage(content=[
            {"type": "text", "text": f"Carefully analyze this image and answer the following legal or document-related question: {question}"},
            {"type": "image_url", "image_url": image}
        ])]
        response = llm_vision.invoke(prompt_parts)
        return jsonify({"answer": response.content, "related_questions": []})
    except Exception as e:
        return jsonify({"error": f"Failed to process image query: {str(e)}"}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribes audio file to text."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400
    
    audio_file = request.files['audio']
    recognizer = sr.Recognizer()

    try:
        # Browser sends webm, convert it to WAV for speech_recognition
        sound = AudioSegment.from_file(audio_file, format="webm")
        wav_io = BytesIO()
        sound.export(wav_io, format="wav")
        wav_io.seek(0)
        
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
        
        text = recognizer.recognize_google(audio_data)
        return jsonify({"transcription": text})
    except Exception as e:
        return jsonify({"error": f"Could not transcribe audio: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)