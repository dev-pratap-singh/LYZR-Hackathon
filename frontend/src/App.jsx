import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';
import GraphVisualization from './components/GraphVisualization';

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
  const [showGraph, setShowGraph] = useState(false);
  const [graphUpdateTrigger, setGraphUpdateTrigger] = useState(0);

  const messagesEndRef = useRef(null);
  const reasoningEndRef = useRef(null);
  const startTimeRef = useRef(null);

  // Only scroll within containers, not entire page
  useEffect(() => {
    if (reasoningSteps.length > 0) {
      reasoningEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [reasoningSteps]);

  useEffect(() => {
    if (messages.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages]);

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

    setUploadStatus('⏳ Uploading and processing...');

    try {
      const response = await axios.post(`${API_URL}/api/rag/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setUploadStatus(`✅ ${response.data.message}`);
      setSelectedDocument(response.data.document_id);
      loadDocuments();

      setTimeout(() => setUploadStatus(''), 5000);
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

    // If graph was updated, trigger refresh
    if (event.type === 'graph_updated') {
      setGraphUpdateTrigger(prev => prev + 1);
    }
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
        loadDocuments();
        if (selectedDocument === docId) {
          setSelectedDocument(null);
        }
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
      'metadata': 'ℹ️',
      'graph_updated': '🔄'
    };
    return icons[type] || '📌';
  };

  const formatAnswer = (content) => {
    // Use ReactMarkdown to properly render markdown with formatting
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Ensure proper spacing for headings
          // eslint-disable-next-line jsx-a11y/heading-has-content
          h1: ({node, ...props}) => <h1 style={{marginTop: '1em', marginBottom: '0.5em'}} {...props} />,
          // eslint-disable-next-line jsx-a11y/heading-has-content
          h2: ({node, ...props}) => <h2 style={{marginTop: '1em', marginBottom: '0.5em'}} {...props} />,
          // eslint-disable-next-line jsx-a11y/heading-has-content
          h3: ({node, ...props}) => <h3 style={{marginTop: '0.8em', marginBottom: '0.4em'}} {...props} />,
          // Ensure proper spacing for paragraphs
          p: ({node, ...props}) => <p style={{marginBottom: '1em', lineHeight: '1.6'}} {...props} />,
          // Ensure proper spacing for lists
          ul: ({node, ...props}) => <ul style={{marginTop: '0.5em', marginBottom: '1em'}} {...props} />,
          ol: ({node, ...props}) => <ol style={{marginTop: '0.5em', marginBottom: '1em'}} {...props} />,
          // Ensure proper spacing for horizontal rules
          hr: ({node, ...props}) => <hr style={{margin: '1.5em 0', border: 'none', borderTop: '1px solid #ddd'}} {...props} />,
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <h1>🧠 RAG System with Knowledge Graph</h1>
            <p>Upload documents, ask questions, and explore the knowledge graph</p>
          </div>
          <button
            className="graph-toggle-button"
            onClick={() => setShowGraph(!showGraph)}
          >
            {showGraph ? '📋 Show Documents & Reasoning' : '🕸️ Show Knowledge Graph'}
          </button>
        </div>
      </header>

      {!showGraph ? (
        // Normal View: Documents + Chat + Reasoning
        <div className="three-column-layout">
          {/* Documents Panel */}
          <div className="left-panel">
            <h2>📁 Documents</h2>

            <div
              className={`upload-zone ${isDragging ? 'dragging' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-input').click()}
            >
              <div className="upload-icon">📄</div>
              <p>Drop PDF here</p>
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
              {documents.length === 0 ? (
                <p className="no-docs">No documents yet</p>
              ) : (
                documents.map(doc => (
                  <div
                    key={doc.id}
                    className={`document-item ${selectedDocument === doc.id ? 'selected' : ''}`}
                    onClick={() => setSelectedDocument(doc.id)}
                  >
                    <span className="doc-icon">📄</span>
                    <div className="doc-info">
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
          <div className="center-panel">
            <h2>💬 Chat</h2>

            <div className="messages-container">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <p>👋 Upload a PDF and ask questions!</p>
                  <div className="sample-queries">
                    <div className="sample-query" onClick={() => setQuery('Summarize the main points')}>
                      Summarize the main points
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
                      <div className="message-content">
                        {msg.role === 'assistant' ? formatAnswer(msg.content) : msg.content}
                      </div>
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
                placeholder="Ask a question..."
                disabled={isProcessing}
                className="query-input"
              />
              <button type="submit" disabled={isProcessing || !documents.length} className="send-button">
                {isProcessing ? '⏳' : '🚀'}
              </button>
              {messages.length > 0 && (
                <button type="button" onClick={clearMessages} className="clear-button">
                  🗑️
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
          <div className="right-panel">
            <h2>🔍 Reasoning</h2>

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

                      {/* Display tool output preview if available */}
                      {step.metadata?.output_preview && step.type === 'tool_end' && (
                        <div className="tool-output-preview">
                          <div className="output-preview-header">📄 Retrieved Data:</div>
                          <div className="output-preview-content">
                            {step.metadata.output_preview}
                          </div>
                        </div>
                      )}

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
      ) : (
        // Graph View: Large Graph + Chat
        <div className="graph-layout">
          {/* Graph Panel */}
          <div className="graph-panel-large">
            <GraphVisualization refreshTrigger={graphUpdateTrigger} />
          </div>

          {/* Chat Panel (smaller) */}
          <div className="chat-panel-small">
            <h2>💬 Chat</h2>

            <div className="messages-container-small">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <p>👋 Ask questions!</p>
                </div>
              ) : (
                <>
                  {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                      <div className="message-avatar">
                        {msg.role === 'user' ? '👤' : '🤖'}
                      </div>
                      <div className="message-content">
                        {msg.role === 'assistant' ? formatAnswer(msg.content) : msg.content}
                      </div>
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
                placeholder="Ask a question..."
                disabled={isProcessing}
                className="query-input"
              />
              <button type="submit" disabled={isProcessing || !documents.length} className="send-button">
                {isProcessing ? '⏳' : '🚀'}
              </button>
            </form>

            {metrics.steps > 0 && (
              <div className="metrics-small">
                <span>📊 {metrics.steps}</span>
                <span>🔧 {metrics.tools}</span>
                <span>⏱️ {metrics.time}s</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
