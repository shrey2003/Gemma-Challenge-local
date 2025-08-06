/**
 * Speech-to-text functionality for chat-o-llama
 * Handles audio recording and transcription via Whisper API
 */

// Speech recognition state
let isRecording = false;
let mediaRecorder = null;
let audioStream = null;
let audioChunks = [];
let speechRecognitionSupported = false;

// Check for browser speech recognition support
function checkSpeechSupport() {
    speechRecognitionSupported = !!(window.MediaRecorder && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    
    const speechButton = document.getElementById('speechButton');
    if (!speechRecognitionSupported) {
        speechButton.disabled = true;
        speechButton.title = 'Speech recognition not supported in this browser';
        speechButton.innerHTML = '🚫';
    }
    
    return speechRecognitionSupported;
}

// Initialize speech functionality
function initializeSpeech() {
    checkSpeechSupport();
    loadSpeechLanguages();
}

// Load supported languages
async function loadSpeechLanguages() {
    try {
        const response = await fetch('/api/speech/languages');
        if (response.ok) {
            const data = await response.json();
            populateLanguageSelector(data.languages);
        }
    } catch (error) {
        console.warn('Could not load speech languages:', error);
    }
}

// Populate language selector
function populateLanguageSelector(languages) {
    const selector = document.getElementById('speechLanguageSelect');
    if (!selector || !languages) return;
    
    // Keep existing options and add more
    const existingOptions = Array.from(selector.options).map(opt => opt.value);
    
    languages.forEach(lang => {
        if (!existingOptions.includes(lang.code)) {
            const option = document.createElement('option');
            option.value = lang.code;
            option.textContent = lang.code.toUpperCase();
            option.title = lang.name;
            selector.appendChild(option);
        }
    });
}

// Toggle speech recording
async function toggleSpeechRecording() {
    if (!speechRecognitionSupported) {
        showSpeechStatus('Speech recognition not supported', 'error');
        return;
    }
    
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

// Start audio recording
async function startRecording() {
    try {
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        // Create media recorder
        mediaRecorder = new MediaRecorder(audioStream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await processAudioRecording(audioBlob);
        };
        
        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        isRecording = true;
        
        updateSpeechUI('recording');
        showSpeechStatus('Recording... Click to stop', 'recording');
        
        console.log('Audio recording started');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        showSpeechStatus('Microphone access denied', 'error');
        
        // Update UI to show error
        const speechButton = document.getElementById('speechButton');
        speechButton.innerHTML = '🚫';
        setTimeout(() => {
            speechButton.innerHTML = '🎤';
        }, 2000);
    }
}

// Stop audio recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Stop all audio tracks
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }
        
        updateSpeechUI('processing');
        showSpeechStatus('Processing audio...', 'processing');
        
        console.log('Audio recording stopped');
    }
}

// Process recorded audio
async function processAudioRecording(audioBlob) {
    try {
        // Convert blob to base64
        const base64Audio = await blobToBase64(audioBlob);
        
        // Get selected language
        const languageSelect = document.getElementById('speechLanguageSelect');
        const language = languageSelect.value !== 'auto' ? languageSelect.value : null;
        
        // Send to speech API
        const response = await fetch('/api/speech/transcribe/bytes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                audio_data: base64Audio.split(',')[1], // Remove data:audio/webm;base64, prefix
                format: 'webm',
                language: language,
                model: 'openai/whisper-base'
            })
        });
        
        const result = await response.json();
        
        if (result.success && result.text.trim()) {
            // Insert transcribed text into message input
            const messageInput = document.getElementById('messageInput');
            const currentText = messageInput.value;
            const newText = currentText ? currentText + ' ' + result.text : result.text;
            
            messageInput.value = newText;
            autoResize(messageInput);
            messageInput.focus();
            
            showSpeechStatus(`Transcribed: "${result.text.substring(0, 50)}${result.text.length > 50 ? '...' : ''}"`, 'success');
            
            console.log('Transcription successful:', result.text);
        } else {
            const errorMsg = result.error || 'No speech detected';
            showSpeechStatus(errorMsg, 'error');
            console.warn('Transcription failed:', errorMsg);
        }
        
    } catch (error) {
        console.error('Error processing audio:', error);
        showSpeechStatus('Failed to process audio', 'error');
    } finally {
        updateSpeechUI('idle');
    }
}

// Convert blob to base64
function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

// Update speech UI state
function updateSpeechUI(state) {
    const speechButton = document.getElementById('speechButton');
    const visualizer = document.getElementById('audioVisualizer');
    
    speechButton.classList.remove('recording', 'processing');
    visualizer.classList.remove('active');
    
    switch (state) {
        case 'recording':
            speechButton.classList.add('recording');
            speechButton.innerHTML = '⏹️';
            speechButton.title = 'Stop recording';
            visualizer.classList.add('active');
            break;
            
        case 'processing':
            speechButton.classList.add('processing');
            speechButton.innerHTML = '';
            speechButton.title = 'Processing...';
            break;
            
        default: // idle
            speechButton.innerHTML = '🎤';
            speechButton.title = 'Voice input';
            break;
    }
}

// Show speech status message
function showSpeechStatus(message, type = 'info') {
    const statusDiv = document.getElementById('speechStatus');
    const statusIcon = statusDiv.querySelector('.status-icon');
    const statusText = statusDiv.querySelector('.status-text');
    
    // Remove existing classes
    statusDiv.classList.remove('recording', 'processing', 'error', 'success');
    
    // Set icon and class based on type
    switch (type) {
        case 'recording':
            statusIcon.textContent = '🔴';
            statusDiv.classList.add('recording');
            break;
        case 'processing':
            statusIcon.textContent = '⚡';
            statusDiv.classList.add('processing');
            break;
        case 'error':
            statusIcon.textContent = '❌';
            statusDiv.classList.add('error');
            break;
        case 'success':
            statusIcon.textContent = '✅';
            statusDiv.classList.add('success');
            break;
        default:
            statusIcon.textContent = '🎤';
    }
    
    statusText.textContent = message;
    statusDiv.classList.add('show');
    
    // Auto-hide after delay (except for recording state)
    if (type !== 'recording') {
        setTimeout(() => {
            statusDiv.classList.remove('show');
        }, type === 'error' ? 4000 : 3000);
    }
}

// Hide speech status
function hideSpeechStatus() {
    const statusDiv = document.getElementById('speechStatus');
    statusDiv.classList.remove('show');
}

// Handle keyboard shortcuts for speech
function handleSpeechKeyboard(event) {
    // Space bar to toggle recording (when input is not focused)
    if (event.code === 'Space' && event.target !== document.getElementById('messageInput')) {
        if (speechRecognitionSupported) {
            event.preventDefault();
            toggleSpeechRecording();
        }
    }
    
    // Escape to stop recording
    if (event.key === 'Escape' && isRecording) {
        event.preventDefault();
        stopRecording();
    }
}

// Check speech API status
async function checkSpeechAPIStatus() {
    try {
        const response = await fetch('/api/speech/status');
        if (response.ok) {
            const status = await response.json();
            console.log('Speech API status:', status);
            return status.status?.available || false;
        }
    } catch (error) {
        console.warn('Speech API not available:', error);
    }
    return false;
}

// Initialize speech functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeSpeech();
    
    // Add keyboard event listeners
    document.addEventListener('keydown', handleSpeechKeyboard);
    
    // Check API status
    checkSpeechAPIStatus().then(available => {
        if (!available) {
            console.warn('Speech-to-text API not available');
            const speechButton = document.getElementById('speechButton');
            speechButton.disabled = true;
            speechButton.title = 'Speech-to-text service unavailable';
            speechButton.innerHTML = '🚫';
        }
    });
    
    console.log('Speech-to-text functionality initialized');
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (isRecording) {
        stopRecording();
    }
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
});