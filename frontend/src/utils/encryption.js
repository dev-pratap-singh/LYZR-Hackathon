/**
 * Encryption utilities for securely storing API keys
 * Uses AES encryption with a device-specific key
 */
import CryptoJS from 'crypto-js';

// Generate a device-specific encryption key based on browser fingerprint
const getEncryptionKey = () => {
  // Create a fingerprint from browser properties
  const fingerprint = [
    navigator.userAgent,
    navigator.language,
    new Date().getTimezoneOffset(),
    window.screen.width,
    window.screen.height,
  ].join('|');

  // Hash the fingerprint to create a consistent key
  return CryptoJS.SHA256(fingerprint).toString();
};

/**
 * Encrypt a string (API key) for storage
 * @param {string} text - The text to encrypt
 * @returns {string} Encrypted text
 */
export const encryptKey = (text) => {
  if (!text) return '';
  const key = getEncryptionKey();
  return CryptoJS.AES.encrypt(text, key).toString();
};

/**
 * Decrypt a string (API key) from storage
 * @param {string} encryptedText - The encrypted text
 * @returns {string} Decrypted text
 */
export const decryptKey = (encryptedText) => {
  if (!encryptedText) return '';
  try {
    const key = getEncryptionKey();
    const bytes = CryptoJS.AES.decrypt(encryptedText, key);
    return bytes.toString(CryptoJS.enc.Utf8);
  } catch (error) {
    console.error('Decryption failed:', error);
    return '';
  }
};

/**
 * Save encrypted API key to localStorage
 * @param {string} apiKey - The API key to save
 */
export const saveApiKey = (apiKey) => {
  if (!apiKey) {
    localStorage.removeItem('encrypted_openai_key');
    return;
  }
  const encrypted = encryptKey(apiKey);
  localStorage.setItem('encrypted_openai_key', encrypted);
};

/**
 * Load and decrypt API key from localStorage
 * @returns {string} Decrypted API key or empty string
 */
export const loadApiKey = () => {
  const encrypted = localStorage.getItem('encrypted_openai_key');
  if (!encrypted) return '';
  return decryptKey(encrypted);
};

/**
 * Check if an API key is stored
 * @returns {boolean} True if a key is stored
 */
export const hasStoredApiKey = () => {
  return !!localStorage.getItem('encrypted_openai_key');
};

/**
 * Clear stored API key
 */
export const clearApiKey = () => {
  localStorage.removeItem('encrypted_openai_key');
};
