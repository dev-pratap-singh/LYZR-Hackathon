import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  // State management
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [reasoningSteps, setReasoningSteps] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [metrics, setMetrics] = useState({ steps: 0, tools: 0, time: 0 });

  const messagesEndRef = useRef(null);
  const reasoningEndRef = useRef(null);
  const startTimeRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    reasoningEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, reasoningSteps]);

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/rag/documents`);
      setDocuments(response.data);
    } catch (error) {
      console.error('Error loading documents:', error);
    }
  };

  // File upload handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const pdfFile = files.find(file => file.name.endsWith('.pdf'));

    if (pdfFile) {
      await uploadFile(pdfFile);
    } else {
      setUploadStatus('❌ Please upload a PDF file');
    }
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.pdf')) {
      await uploadFile(file);
    }
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    setUploadStatus('⏳ Uploading...');

    try {
      const response = await axios.post(`${API_URL}/api/rag/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setUploadStatus(`✅ ${response.data.message}`);
      setSelectedDocument(response.data.document_id);
      loadDocuments();

      setTimeout(() => setUploadStatus(''), 3000);
    } catch (error) {
      setUploadStatus(`❌ Upload failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  // Query submission with streaming
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!query.trim()) {
      alert('Please enter a question');
      return;
    }

    if (!selectedDocument && documents.length === 0) {
      alert('Please upload a document first');
      return;
    }

    setIsProcessing(true);
    setReasoningSteps([]);
    setMessages([...messages, { role: 'user', content: query }]);

    startTimeRef.current = Date.now();

    try {
      const response = await fetch(`${API_URL}/api/rag/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          document_id: selectedDocument || documents[0]?.id
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalAnswer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              handleStreamEvent(event);

              if (event.type === 'final_answer') {
                finalAnswer = event.content;
              }
            } catch (e) {
              console.error('Error parsing event:', e);
            }
          }
        }
      }

      // Add assistant message
      if (finalAnswer) {
        setMessages(prev => [...prev, { role: 'assistant', content: finalAnswer }]);
      }

      // Update metrics
      const elapsedTime = ((Date.now() - startTimeRef.current) / 1000).toFixed(1);
      setMetrics(prev => ({
        ...prev,
        time: elapsedTime
      }));

    } catch (error) {
      console.error('Error:', error);
      setReasoningSteps(prev => [...prev, {
        type: 'error',
        content: `Error: ${error.message}`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsProcessing(false);
      setQuery('');
    }
  };

  const handleStreamEvent = (event) => {
    const step = {
      ...event,
      timestamp: event.timestamp || new Date().toISOString()
    };

    setReasoningSteps(prev => [...prev, step]);

    // Update metrics
    setMetrics(prev => ({
      steps: prev.steps + 1,
      tools: event.metadata?.tool ? prev.tools + 1 : prev.tools,
      time: prev.time
    }));
  };

  const clearMessages = () => {
    setMessages([]);
    setReasoningSteps([]);
    setMetrics({ steps: 0, tools: 0, time: 0 });
  };

  const handleDeleteDocument = async (docId, filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"? This will remove all associated data including embeddings.`)) {
      return;
    }

    try {
      const response = await axios.delete(`${API_URL}/api/rag/documents/${docId}`);

      if (response.data.success) {
        // Refresh document list
        loadDocuments();

        // Clear selection if deleted document was selected
        if (selectedDocument === docId) {
          setSelectedDocument(null);
        }

        // Show success message
        setUploadStatus('✅ Document deleted successfully');
        setTimeout(() => setUploadStatus(''), 3000);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      setUploadStatus(`❌ Failed to delete document: ${error.response?.data?.detail || error.message}`);
      setTimeout(() => setUploadStatus(''), 5000);
    }
  };

  const getStepIcon = (type) => {
    const icons = {
      'thinking': '🤔',
      'tool_start': '🔧',
      'tool_end': '✅',
      'reasoning': '💭',
      'final_answer': '🎯',
      'error': '❌',
      'metadata': 'ℹ️'
    };
    return icons[type] || '📌';
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>🧠 RAG System</h1>
        <p>PDF Document Q&A with Vector Search & Chain of Thought</p>
      </header>

      <div className="main-layout">
        {/* Upload Panel */}
        <div className="upload-panel">
          <h2>📁 Document Upload</h2>

          <div
            className={`upload-zone ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <div className="upload-icon">📄</div>
            <p>Drag & Drop PDF here</p>
            <p className="upload-subtitle">or click to browse</p>
            <input
              id="file-input"
              type="file"
              accept=".pdf"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>

          {uploadStatus && (
            <div className={`upload-status ${uploadStatus.includes('✅') ? 'success' : uploadStatus.includes('❌') ? 'error' : 'info'}`}>
              {uploadStatus}
            </div>
          )}

          <div className="documents-list">
            <h3>Uploaded Documents</h3>
            {documents.length === 0 ? (
              <p className="no-docs">No documents yet</p>
            ) : (
              documents.map(doc => (
                <div
                  key={doc.id}
                  className={`document-item ${selectedDocument === doc.id ? 'selected' : ''}`}
                >
                  <span className="doc-icon">📄</span>
                  <div className="doc-info" onClick={() => setSelectedDocument(doc.id)}>
                    <div className="doc-name">{doc.filename}</div>
                    <div className="doc-status">
                      {doc.is_processed ? `✓ ${doc.total_chunks} chunks` : '⏳ Processing...'}
                    </div>
                  </div>
                  <button
                    className="delete-doc-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteDocument(doc.id, doc.filename);
                    }}
                    title="Delete document"
                  >
                    🗑️
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Chat Panel */}
        <div className="chat-panel">
          <h2>💬 Chat</h2>

          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="empty-state">
                <p>👋 Upload a PDF and ask questions about it!</p>
                <div className="sample-queries">
                  <h4>Example questions:</h4>
                  <div className="sample-query" onClick={() => setQuery('Summarize the main points of this document')}>
                    Summarize the main points of this document
                  </div>
                  <div className="sample-query" onClick={() => setQuery('What are the key findings?')}>
                    What are the key findings?
                  </div>
                </div>
              </div>
            ) : (
              <>
                {messages.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.role}`}>
                    <div className="message-avatar">
                      {msg.role === 'user' ? '👤' : '🤖'}
                    </div>
                    <div className="message-content">{msg.content}</div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your document..."
              disabled={isProcessing}
              className="query-input"
            />
            <button type="submit" disabled={isProcessing || !documents.length} className="send-button">
              {isProcessing ? '⏳' : '🚀'} {isProcessing ? 'Processing...' : 'Send'}
            </button>
            {messages.length > 0 && (
              <button type="button" onClick={clearMessages} className="clear-button">
                🗑️ Clear
              </button>
            )}
          </form>

          {metrics.steps > 0 && (
            <div className="metrics">
              <div className="metric">
                <div className="metric-value">{metrics.steps}</div>
                <div className="metric-label">Steps</div>
              </div>
              <div className="metric">
                <div className="metric-value">{metrics.tools}</div>
                <div className="metric-label">Tools</div>
              </div>
              <div className="metric">
                <div className="metric-value">{metrics.time}s</div>
                <div className="metric-label">Time</div>
              </div>
            </div>
          )}
        </div>

        {/* Reasoning Panel */}
        <div className="reasoning-panel">
          <h2>🔍 Reasoning Process</h2>

          <div className="reasoning-steps">
            {reasoningSteps.length === 0 ? (
              <div className="empty-state">
                <p>🤖 Reasoning steps will appear here...</p>
              </div>
            ) : (
              <>
                {reasoningSteps.map((step, idx) => (
                  <div key={idx} className={`reasoning-step ${step.type}`}>
                    <div className="step-header">
                      <span className="step-icon">{getStepIcon(step.type)}</span>
                      <span className="step-type">{step.type.replace('_', ' ').toUpperCase()}</span>
                      <span className="step-timestamp">
                        {new Date(step.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="step-content">{step.content}</div>
                    {step.metadata && Object.keys(step.metadata).length > 0 && (
                      <div className="step-metadata">
                        {Object.entries(step.metadata)
                          .filter(([key]) => !['full_content', 'output_preview'].includes(key))
                          .map(([key, value]) => (
                            <span key={key} className="metadata-item">
                              {key}: {JSON.stringify(value)}
                            </span>
                          ))}
                      </div>
                    )}
                  </div>
                ))}
                <div ref={reasoningEndRef} />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
