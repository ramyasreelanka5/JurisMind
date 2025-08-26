import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import toText from 'markdown-to-text';
import useAudioRecorder from './useAudioRecorder';
import './App.css';

const API_BASE_URL = 'http://localhost:5001';

// --- TTS Utility ---
const speech = {
  synth: window.speechSynthesis,
  utterance: null,
  speak: function(text, onEndCallback) {
    if (this.synth.speaking) { this.synth.cancel(); }
    this.utterance = new SpeechSynthesisUtterance(text);
    this.utterance.onend = onEndCallback;
    this.synth.speak(this.utterance);
  },
  cancel: function() { this.synth.cancel(); }
};

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speakingMessageId, setSpeakingMessageId] = useState(null);

  const { isRecording, startRecording, stopRecording, audioBlob } = useAudioRecorder();
  const fileInputRef = useRef(null);
  const chatBoxRef = useRef(null);
  const sessionId = "user123";

  useEffect(() => {
    if (chatBoxRef.current) { chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight; }
  }, [messages]);

  useEffect(() => {
    if (audioBlob) { handleTranscription(audioBlob); }
  }, [audioBlob]);

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

  const handleSendMessage = async (queryOverride) => {
    const query = queryOverride || input.trim();
    if ((!query && !selectedImage) || isLoading) return;

    // --- CRITICAL FIX IS HERE ---
    // Use the functional form of setMessages to get the most up-to-date state
    setMessages(prevMessages => {
        let updatedMessages = [...prevMessages];
        
        // If this function was called from a related question (queryOverride exists),
        // find the last message and remove its related questions.
        if (queryOverride) {
            const lastMessageIndex = updatedMessages.length - 1;
            if (lastMessageIndex >= 0 && updatedMessages[lastMessageIndex].role === 'assistant') {
                updatedMessages[lastMessageIndex] = {
                    ...updatedMessages[lastMessageIndex],
                    relatedQuestions: [] // Set to empty array to make it disappear
                };
            }
        }
        
        // Now, add the new user message to the potentially modified array
        const userMessage = { role: 'user', content: query };
        return [...updatedMessages, userMessage];
    });

    setInput('');
    setIsLoading(true);

    try {
      // The chat history sent to the backend will be based on the state before this update,
      // which is the correct behavior as the backend's memory will handle the new question.
      const chatHistory = messages.filter(m => m.role !== 'system');
      
      let response;
      if (selectedImage) {
        const formData = new FormData();
        formData.append('question', query);
        formData.append('image', selectedImage);
        response = await axios.post(`${API_BASE_URL}/chat_vision`, formData);
        setSelectedImage(null);
        setImagePreview(null);
      } else {
        response = await axios.post(`${API_BASE_URL}/chat`, {
          question: query,
          chat_history: chatHistory,
          session_id: sessionId
        });
      }
      
      const { answer, related_questions } = response.data;
      const assistantMessage = {
        role: 'assistant',
        content: answer,
        relatedQuestions: related_questions || []
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error("Error fetching response:", error);
      const errorMessage = error.response?.data?.error || "Sorry, something went wrong.";
      setMessages(prev => [...prev, { role: 'assistant', content: errorMessage, relatedQuestions: [] }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const fileType = file.type.split('/')[0];
    if (fileType === 'image') {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
    } else {
      handleDocumentUpload(file);
    }
    event.target.value = null;
  };

  const handleDocumentUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);
    setMessages(prev => [...prev, {role: 'system', content: `Uploading ${file.name}...`}]);
    try {
      const response = await axios.post(`${API_BASE_URL}/upload_document`, formData);
      setMessages(prev => [...prev, {role: 'system', content: response.data.message}]);
    } catch (error) {
      const errorMessage = error.response?.data?.error || "File upload failed.";
      setMessages(prev => [...prev, { role: 'system', content: errorMessage }]);
    }
  };

  const handleTranscription = async (blob) => {
    const formData = new FormData();
    formData.append('audio', blob, 'audio.webm');
    try {
      const response = await axios.post(`${API_BASE_URL}/transcribe`, formData);
      setInput(prev => (prev ? prev + ' ' : '') + response.data.transcription);
    } catch (error) {
      console.error("Error transcribing audio:", error);
      alert("Audio transcription failed.");
    }
  };

  const handleReset = async () => {
    try {
      await axios.post(`${API_BASE_URL}/reset`, { session_id: sessionId });
      setMessages([]);
      speech.cancel();
      setIsSpeaking(false);
      alert("Chat has been reset.");
    } catch (error) {
      console.error("Error resetting chat:", error);
      alert("Failed to reset chat.");
    }
  };

  const MarkdownRenderer = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter style={vscDarkPlus} language={match[1]} PreTag="div" {...props}>
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props}>{children}</code>
      );
    },
  };

  return (
    <div className="app-container">
      <h1>Legal ChatBot</h1>
      <div className="chat-box" ref={chatBoxRef}>
        {messages.map((msg, index) => (
          <div key={index} id={`msg-${index}`} className={msg.role === 'user' ? 'user-msg' : 'bot-msg'}>
            {msg.role === 'assistant' && (
              <button className="icon-button tts-button" onClick={() => handleToggleSpeech(index, msg.content)} title={isSpeaking && speakingMessageId === index ? "Stop Speaking" : "Read Aloud"}>
                {isSpeaking && speakingMessageId === index ? '⏹️' : '🔊'}
              </button>
            )}
            <ReactMarkdown components={MarkdownRenderer}>{msg.content}</ReactMarkdown>
            {msg.role === 'assistant' && msg.relatedQuestions && msg.relatedQuestions.length > 0 && (
              <div className="related-questions">
                {msg.relatedQuestions.map((q, i) => (
                  <button key={i} onClick={() => handleSendMessage(q)}>{q}</button>
                ))}
              </div>
            )}
          </div>
        ))}
        {isLoading && <div className="bot-msg spinner">Thinking...</div>}
      </div>
      <div className="sticky-input-area">
        {imagePreview && (
          <div className="image-preview">
            <img src={imagePreview} alt="preview" />
            <button onClick={() => { setSelectedImage(null); setImagePreview(null); }}>×</button>
          </div>
        )}
        <div className="input-container">
          <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".pdf,.docx,.txt,image/*" style={{ display: 'none' }} />
          <button className="icon-button" onClick={() => fileInputRef.current.click()} title="Upload File or Image">📎</button>
          <input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()} placeholder="Ask a question..." />
          <button className={`icon-button ${isRecording ? 'recording' : ''}`} onClick={isRecording ? stopRecording : startRecording} title={isRecording ? 'Stop Recording' : 'Record Voice'}>
            {isRecording ? '⏹️' : '🎤'}
          </button>
          <button className="icon-button" onClick={() => handleSendMessage()} title="Send">➤</button>
          <button className="icon-button" onClick={handleReset} title="Reset Chat">🗑️</button>
        </div>
        {isRecording && <div className="recording-indicator">Recording...</div>}
      </div>
    </div>
  );
};

export default App;