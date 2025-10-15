import React, { useState, useEffect } from 'react';
import { loadApiKey, saveApiKey, clearApiKey, hasStoredApiKey } from '../utils/encryption';
import './SettingsModal.css';

const SettingsModal = ({ isOpen, onClose }) => {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [savedStatus, setSavedStatus] = useState('');
  const [hasKey, setHasKey] = useState(false);

  useEffect(() => {
    if (isOpen) {
      // Check if key exists
      setHasKey(hasStoredApiKey());

      // Load key if it exists (show masked version)
      const key = loadApiKey();
      if (key) {
        setApiKey(key);
      }
    }
  }, [isOpen]);

  const handleSave = () => {
    if (!apiKey.trim()) {
      setSavedStatus('❌ Please enter an API key');
      setTimeout(() => setSavedStatus(''), 3000);
      return;
    }

    // Validate OpenAI key format (starts with sk-)
    if (!apiKey.startsWith('sk-')) {
      setSavedStatus('❌ Invalid OpenAI API key format');
      setTimeout(() => setSavedStatus(''), 3000);
      return;
    }

    // Encrypt and save
    saveApiKey(apiKey);
    setHasKey(true);
    setSavedStatus('✅ API key saved securely (encrypted)');
    setTimeout(() => {
      setSavedStatus('');
      onClose();
    }, 2000);
  };

  const handleClear = () => {
    if (window.confirm('Are you sure you want to remove your saved API key?')) {
      clearApiKey();
      setApiKey('');
      setHasKey(false);
      setSavedStatus('🗑️ API key removed');
      setTimeout(() => setSavedStatus(''), 3000);
    }
  };

  const maskApiKey = (key) => {
    if (!key || key.length < 10) return key;
    return key.slice(0, 7) + '•'.repeat(20) + key.slice(-4);
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>⚙️ Settings</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          <div className="settings-section">
            <h3>🔑 OpenAI API Key</h3>
            <p className="settings-description">
              Provide your own OpenAI API key to use your account for queries.
              Your key is encrypted and stored locally in your browser.
            </p>

            <div className="key-status">
              {hasKey ? (
                <span className="status-indicator active">✅ API key is configured</span>
              ) : (
                <span className="status-indicator inactive">⚠️ Using system default key</span>
              )}
            </div>

            <div className="input-group">
              <label htmlFor="api-key">API Key</label>
              <div className="key-input-wrapper">
                <input
                  id="api-key"
                  type={showKey ? "text" : "password"}
                  value={showKey ? apiKey : maskApiKey(apiKey)}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  className="key-input"
                  onFocus={() => setShowKey(true)}
                  onBlur={() => setShowKey(false)}
                />
                <button
                  className="toggle-visibility"
                  onClick={() => setShowKey(!showKey)}
                  type="button"
                >
                  {showKey ? '🙈' : '👁️'}
                </button>
              </div>
              <small className="input-hint">
                Your API key starts with "sk-" and is stored encrypted in your browser
              </small>
            </div>

            {savedStatus && (
              <div className={`save-status ${savedStatus.includes('✅') ? 'success' : savedStatus.includes('❌') ? 'error' : 'info'}`}>
                {savedStatus}
              </div>
            )}

            <div className="button-group">
              <button className="btn-primary" onClick={handleSave}>
                💾 Save API Key
              </button>
              {hasKey && (
                <button className="btn-danger" onClick={handleClear}>
                  🗑️ Remove Key
                </button>
              )}
            </div>

            <div className="security-note">
              <strong>🔒 Security:</strong> Your API key is encrypted using AES-256 encryption
              before being stored in your browser's localStorage. It's only decrypted when
              sending requests to the backend.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
