import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import toText from 'markdown-to-text';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import useAudioRecorder from './useAudioRecorder';
import Sidebar from './Sidebar';
import AuthModal from './AuthModal';
import './App.css';

const API_BASE_URL = 'http://localhost:5001';

// --- Axios API setup ---
const api = axios.create({ baseURL: API_BASE_URL });
api.interceptors.request.use(config => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
}, error => Promise.reject(error));

api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

const speech = {
  synth: window.speechSynthesis,
  utterance: null,
  speak: function(text, onEndCallback) {
    if (this.synth.speaking) this.synth.cancel();
    this.utterance = new SpeechSynthesisUtterance(text);
    this.utterance.onend = onEndCallback;
    this.synth.speak(this.utterance);
  },
  cancel: function() { this.synth.cancel(); }
};


const App = () => {
  // --- State Hooks ---
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speakingMessageId, setSpeakingMessageId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 768);
  const [isLoadingConversations, setIsLoadingConversations] = useState(true);

  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark');

  // --- NEW: State for multilingual support ---
  const [supportedLanguages, setSupportedLanguages] = useState([]);
  const [currentLanguage, setCurrentLanguage] = useState('en'); // Default to English

  const { isRecording, startRecording, stopRecording, audioBlob } = useAudioRecorder();
  const fileInputRef = useRef(null);
  const chatWrapperRef = useRef(null);

  // --- Effects ---
  useEffect(() => { checkAuth(); }, []);
  useEffect(() => { if (isAuthenticated) loadConversations(); }, [isAuthenticated]);
  useEffect(() => { if (chatWrapperRef.current) chatWrapperRef.current.scrollTop = chatWrapperRef.current.scrollHeight; }, [messages, isLoading]);
  useEffect(() => { if (audioBlob) handleTranscription(audioBlob); }, [audioBlob]);

  useEffect(() => {
    document.body.className = '';
    document.body.classList.add(`${theme}-mode`);
    localStorage.setItem('theme', theme);
  }, [theme]);
  
  // --- NEW: Effect to load supported languages ---
  useEffect(() => {
    const loadLanguages = async () => {
      if (isAuthenticated) {
        try {
          const response = await api.get('/languages');
          setSupportedLanguages(response.data);
        } catch (error) {
          console.error("Failed to load languages:", error);
          setSupportedLanguages([{ code: 'en', name: 'English', native_name: 'English' }]); // Fallback
        }
      }
    };
    loadLanguages();
  }, [isAuthenticated]);

  const toggleTheme = () => setTheme(prevTheme => prevTheme === 'dark' ? 'light' : 'dark');

  // --- Authentication ---
  const checkAuth = async () => {
    const token = localStorage.getItem('token');
    if (!token) {
      setIsCheckingAuth(false);
      setShowAuthModal(true);
      return;
    }
    try {
      const response = await api.get('/verify');
      if (response.data.valid) {
        setIsAuthenticated(true);
        setUser(JSON.parse(localStorage.getItem('user')));
      }
    } catch (error) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      setShowAuthModal(true);
    } finally {
      setIsCheckingAuth(false);
    }
  };
  const handleLogin = async (email, password) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/login`, { email, password });
      const { token, username, user_id } = response.data;
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify({ username, user_id }));
      setIsAuthenticated(true); setUser({ username, user_id });
      setShowAuthModal(false); return { success: true };
    } catch (error) { return { success: false, error: error.response?.data?.error || 'Login failed' }; }
  };
  const handleRegister = async (username, email, password) => {
    try {
      await axios.post(`${API_BASE_URL}/register`, { username, email, password });
      return await handleLogin(email, password);
    } catch (error) { return { success: false, error: error.response?.data?.error || 'Registration failed' }; }
  };
  const handleLogout = async () => {
    try { await api.post('/logout'); } catch (error) { console.error('Logout error:', error); }
    finally {
      localStorage.clear(); localStorage.setItem('theme', theme);
      setIsAuthenticated(false); setUser(null); setMessages([]);
      setConversations([]); setCurrentConversationId(null); setShowAuthModal(true);
    }
  };

  // --- Conversations ---
  const loadConversations = async () => {
    try {
      setIsLoadingConversations(true);
      const response = await api.get('/conversations');
      setConversations(response.data.conversations);
    } catch (error) { console.error("Error loading conversations:", error); } 
    finally { setIsLoadingConversations(false); }
  };
  const loadConversation = async (conversationId) => {
    try {
      setIsLoading(true);
      const response = await api.get(`/conversations/${conversationId}`);
      setMessages(response.data.messages.map(msg => ({...msg, relatedQuestions: msg.role === 'assistant' ? [] : undefined })));
      setCurrentConversationId(conversationId);
      if (window.innerWidth <= 768 && sidebarOpen) setSidebarOpen(false);
    } catch (error) { console.error("Error loading conversation:", error); } 
    finally { setIsLoading(false); }
  };
  const createNewConversation = () => {
    setCurrentConversationId(null); setMessages([]);
    if (window.innerWidth <= 768 && sidebarOpen) setSidebarOpen(false);
  };
  const deleteConversation = async (conversationId) => {
    if (!window.confirm("Are you sure?")) return;
    try {
      await api.delete(`/conversations/${conversationId}`);
      if (conversationId === currentConversationId) createNewConversation();
      await loadConversations();
    } catch (error) { console.error("Error deleting conversation:", error); }
  };

  // --- Core Chat & File Handling ---
  const handleSendMessage = async (queryOverride) => {
    const query = queryOverride || input.trim();
    if ((!query && !selectedImage) || isLoading) return;

    const userMessage = { role: 'user', content: query, preview: imagePreview };
    setMessages(prev => [...prev.map(p => ({...p, relatedQuestions: []})), userMessage]);
    
    setInput('');
    setIsLoading(true);

    try {
      let response;
      // --- UPDATED: Add language to the request data ---
      const requestData = { 
        question: query, 
        conversation_id: currentConversationId,
        language: currentLanguage // Send the selected language to the backend
      };

      if (selectedImage) {
        const formData = new FormData();
        Object.keys(requestData).forEach(key => formData.append(key, requestData[key]));
        formData.append('image', selectedImage);
        response = await api.post('/chat_vision', formData);
        
        setSelectedImage(null); 
        setImagePreview(null);
      } else {
        response = await api.post('/chat', requestData);
      }
      
      const { answer, related_questions, conversation_id } = response.data;
      
      if (!currentConversationId && conversation_id) {
        setCurrentConversationId(conversation_id);
        await loadConversations();
      }
      
      setMessages(prev => [...prev, { role: 'assistant', content: answer, relatedQuestions: related_questions || [] }]);
    } catch (error) {
      const errorMessage = error.response?.data?.error || "Sorry, something went wrong.";
      setMessages(prev => [...prev, { role: 'assistant', content: errorMessage, relatedQuestions: [] }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleFileChange = (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    const docFiles = Array.from(files).filter(file => !file.type.startsWith('image/'));
    if (imageFiles.length > 0) {
      if (imageFiles.length > 1) {
        alert("You can only attach one image per message. The first image has been selected.");
      }
      setSelectedImage(imageFiles[0]); 
      setImagePreview(URL.createObjectURL(imageFiles[0]));
    }
    if (docFiles.length > 0) {
      handleDocumentUpload(docFiles);
    }
    event.target.value = null;
  };
  
  const handleDocumentUpload = async (files) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    const fileNames = files.map(f => f.name).join(', ');
    setMessages(prev => [...prev, {role: 'system', content: `Uploading ${files.length} document(s): ${fileNames}...`}]);
    
    try {
      const response = await api.post('/upload_document', formData);
      setMessages(prev => [...prev, {role: 'system', content: response.data.message}]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'system', content: error.response?.data?.error || "File upload failed." }]);
    }
  };

  const handleTranscription = async (blob) => {
    const formData = new FormData();
    formData.append('audio', blob, 'audio.webm');
    try {
      const response = await api.post('/transcribe', formData);
      setInput(prev => (prev ? prev + ' ' : '') + response.data.transcription);
    } catch (error) { console.error("Error transcribing audio:", error); }
  };

  const handleToggleSpeech = (messageId, markdownText) => {
    if (isSpeaking && speakingMessageId === messageId) {
      speech.cancel();
      setIsSpeaking(false);
      setSpeakingMessageId(null);
    } else {
      const plainText = toText(markdownText);
      setIsSpeaking(true);
      setSpeakingMessageId(messageId);
      speech.speak(plainText, () => {
        setIsSpeaking(false);
        setSpeakingMessageId(null);
      });
    }
  };

  const getUserInitial = (name) => (name ? name.charAt(0).toUpperCase() : '?');

  // --- Render Logic ---
  if (isCheckingAuth) return <div className="loading-screen"><div className="loading-spinner"></div></div>;
  if (!isAuthenticated) return <AuthModal isOpen={showAuthModal} onLogin={handleLogin} onRegister={handleRegister} />;

  const MarkdownRenderer = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter style={vscDarkPlus} language={match[1]} PreTag="div" {...props}>
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : ( <code className={className} {...props}>{children}</code> );
    },
  };

  return (
    <div className="app-container">
      <Sidebar 
        isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)}
        conversations={conversations} currentConversationId={currentConversationId}
        onSelectConversation={loadConversation} onNewConversation={createNewConversation}
        onDeleteConversation={deleteConversation} onLogout={handleLogout}
        user={user} isLoading={isLoadingConversations}
        theme={theme} toggleTheme={toggleTheme}
      />
      
      <div className={`main-content ${sidebarOpen ? 'sidebar-is-open' : ''}`}>
        <header className="header">
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>☰</button>
          <h1 className="header-title">
            {currentConversationId ? conversations.find(c => c.id === currentConversationId)?.title : "Legal AI Assistant"}
          </h1>
          {/* --- UPDATED: Header controls wrapper with language selector --- */}
          <div className="header-controls">
            <select 
              className="language-selector" 
              value={currentLanguage} 
              onChange={(e) => setCurrentLanguage(e.target.value)}
              title="Select language"
            >
              {supportedLanguages.map(lang => (
                <option key={lang.code} value={lang.code}>
                  {lang.native_name}
                </option>
              ))}
            </select>
            <button className="upload-data-button" onClick={() => fileInputRef.current.click()}>
              <span>📤</span>
              <span>Upload Files</span>
            </button>
          </div>
        </header>
        
        <main className="chat-area-wrapper" ref={chatWrapperRef}>
          <div className="chat-box">
            {messages.length === 0 && !isLoading && (
                 <div className="welcome-screen">
                    <div className="welcome-icon">🤖</div>
                    <h2>Welcome, {user?.username}!</h2>

                    <p>Your Guide Through Legal Complexity.</p>
                    <p>
                    You can ask a question related to law, upload a document for analysis, or start a new project to organize your research.
                    <br /><br />
                    
                    </p>
                </div>
            )}
            
            {messages.map((msg, index) => (
              msg.role === 'system' ? (
                <div key={index} className="system-msg">{msg.content}</div>
              ) : (
                <div key={index} className={`message-container ${msg.role}`}>
                  <div className="avatar">
                    {msg.role === 'user' ? getUserInitial(user.username) : '🤖'}
                  </div>
                  <div className="message-content">
                    {msg.role === 'assistant' && msg.content && (
                      <button
                        className="tts-button"
                        onClick={() => handleToggleSpeech(index, msg.content)}
                        title={isSpeaking && speakingMessageId === index ? "Stop speaking" : "Read aloud"}
                      >
                        {isSpeaking && speakingMessageId === index ? '⏹️' : '🔊'}
                      </button>
                    )}
                    <div className="message-bubble">
                      {msg.preview && <img src={msg.preview} alt="User upload" className="message-image" />}
                      <ReactMarkdown components={MarkdownRenderer}>{msg.content}</ReactMarkdown>
                    </div>
                    {msg.role === 'assistant' && msg.relatedQuestions && msg.relatedQuestions.length > 0 && (
                      <div className="suggested-actions">
                        {msg.relatedQuestions.map((q, i) => (
                          <button key={i} onClick={() => handleSendMessage(q)}>{q}</button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )
            ))}
            {isLoading && (
                <div className="message-container bot">
                    <div className="avatar">🤖</div>
                    <div className="message-content"><div className="message-bubble">Thinking...</div></div>
                </div>
            )}
          </div>
        </main>
        
        <footer className="input-area-wrapper">
          <div className="input-area-container">
            {imagePreview && (
              <div className="image-preview">
                <img src={imagePreview} alt="preview" />
                <button onClick={() => { setSelectedImage(null); setImagePreview(null); }}>×</button>
              </div>
            )}
            <div className="input-container">
              <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".pdf,.docx,.txt,image/*" style={{ display: 'none' }} multiple />
              <button className="icon-button" onClick={() => fileInputRef.current.click()} title="Upload Files">📎</button>
              <input type="text" className="input-field" value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()} placeholder="Ask a question or upload files..." />
              <button className={`icon-button ${isRecording ? 'recording' : ''}`} onClick={isRecording ? stopRecording : startRecording} title={isRecording ? 'Stop Recording' : 'Start Recording'}>
                {isRecording ? '⏹️' : '🎤'}
              </button>
              <button className="icon-button send-btn" onClick={() => handleSendMessage()} title="Send">➤</button>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;