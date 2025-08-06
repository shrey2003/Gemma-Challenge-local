"""
Speech-to-text API endpoints.
"""

import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import os
from pathlib import Path

from services.speech_to_text import get_speech_service

logger = logging.getLogger(__name__)

speech_bp = Blueprint('speech', __name__, url_prefix='/api/speech')

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'ogg', 'flac', 'webm'}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@speech_bp.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio file to text.
    
    Expects multipart/form-data with:
    - audio: Audio file
    - language: Optional language code (e.g., 'en', 'es', 'fr')
    - model: Optional model name (default: openai/whisper-base)
    """
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided',
                'success': False
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                'error': 'No audio file selected',
                'success': False
            }), 400
        
        # Check file size
        audio_file.seek(0, os.SEEK_END)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'error': f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB',
                'success': False
            }), 400
        
        # Check file extension
        if not allowed_file(audio_file.filename):
            return jsonify({
                'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                'success': False
            }), 400
        
        # Get optional parameters
        language = request.form.get('language')
        model_name = request.form.get('model', 'openai/whisper-base')
        
        logger.info(f"Transcribing audio file: {audio_file.filename}, language: {language}, model: {model_name}")
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{secure_filename(audio_file.filename)}") as temp_file:
            try:
                audio_file.save(temp_file.name)
                
                # Get speech service and transcribe
                speech_service = get_speech_service(model_name)
                result = speech_service.transcribe_audio(temp_file.name, language)
                
                if result['success']:
                    logger.info(f"Transcription successful. Text length: {len(result['text'])}")
                    return jsonify(result)
                else:
                    logger.error(f"Transcription failed: {result.get('error', 'Unknown error')}")
                    return jsonify(result), 500
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass
    
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return jsonify({
            'error': 'Internal server error during transcription',
            'success': False
        }), 500


@speech_bp.route('/transcribe/bytes', methods=['POST'])
def transcribe_audio_bytes():
    """Transcribe audio from raw bytes.
    
    Expects JSON with:
    - audio_data: Base64 encoded audio bytes
    - format: Audio format (wav, mp3, etc.)
    - language: Optional language code
    - model: Optional model name
    """
    try:
        data = request.get_json()
        
        if not data or 'audio_data' not in data:
            return jsonify({
                'error': 'No audio data provided',
                'success': False
            }), 400
        
        import base64
        
        # Decode base64 audio data
        try:
            audio_bytes = base64.b64decode(data['audio_data'])
        except Exception as e:
            return jsonify({
                'error': 'Invalid base64 audio data',
                'success': False
            }), 400
        
        # Check size
        if len(audio_bytes) > MAX_FILE_SIZE:
            return jsonify({
                'error': f'Audio too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB',
                'success': False
            }), 400
        
        format_type = data.get('format', 'wav')
        language = data.get('language')
        model_name = data.get('model', 'openai/whisper-base')
        
        logger.info(f"Transcribing audio bytes: format={format_type}, language={language}, model={model_name}")
        
        # Get speech service and transcribe
        speech_service = get_speech_service(model_name)
        result = speech_service.transcribe_audio_bytes(audio_bytes, format_type, language)
        
        if result['success']:
            logger.info(f"Transcription successful. Text length: {len(result['text'])}")
            return jsonify(result)
        else:
            logger.error(f"Transcription failed: {result.get('error', 'Unknown error')}")
            return jsonify(result), 500
    
    except Exception as e:
        logger.error(f"Audio bytes transcription error: {e}")
        return jsonify({
            'error': 'Internal server error during transcription',
            'success': False
        }), 500


@speech_bp.route('/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages."""
    try:
        speech_service = get_speech_service()
        languages = speech_service.get_supported_languages()
        
        # Language names mapping (optional)
        language_names = {
            'en': 'English', 'zh': 'Chinese', 'de': 'German', 'es': 'Spanish',
            'ru': 'Russian', 'ko': 'Korean', 'fr': 'French', 'ja': 'Japanese',
            'pt': 'Portuguese', 'tr': 'Turkish', 'pl': 'Polish', 'ca': 'Catalan',
            'nl': 'Dutch', 'ar': 'Arabic', 'sv': 'Swedish', 'it': 'Italian',
            'id': 'Indonesian', 'hi': 'Hindi', 'fi': 'Finnish', 'vi': 'Vietnamese',
            'he': 'Hebrew', 'uk': 'Ukrainian', 'el': 'Greek', 'ms': 'Malay',
            'cs': 'Czech', 'ro': 'Romanian', 'da': 'Danish', 'hu': 'Hungarian',
            'ta': 'Tamil', 'no': 'Norwegian', 'th': 'Thai', 'ur': 'Urdu'
        }
        
        language_list = [
            {
                'code': lang,
                'name': language_names.get(lang, lang.upper())
            }
            for lang in languages
        ]
        
        return jsonify({
            'languages': language_list,
            'success': True
        })
    
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        return jsonify({
            'error': 'Could not retrieve supported languages',
            'success': False
        }), 500


@speech_bp.route('/models', methods=['GET'])
def get_available_models():
    """Get list of available Whisper models."""
    models = [
        {
            'name': 'openai/whisper-tiny',
            'size': '~39MB',
            'description': 'Fastest, least accurate'
        },
        {
            'name': 'openai/whisper-base',
            'size': '~74MB',
            'description': 'Good balance of speed and accuracy'
        },
        {
            'name': 'openai/whisper-small',
            'size': '~244MB',
            'description': 'Better accuracy, slower'
        },
        {
            'name': 'openai/whisper-medium',
            'size': '~769MB',
            'description': 'High accuracy, much slower'
        },
        {
            'name': 'openai/whisper-large-v2',
            'size': '~1.55GB',
            'description': 'Best accuracy, slowest'
        },
        {
            'name': 'openai/whisper-large-v3',
            'size': '~1.55GB',
            'description': 'Latest large model'
        }
    ]
    
    return jsonify({
        'models': models,
        'success': True
    })


@speech_bp.route('/status', methods=['GET'])
def get_speech_status():
    """Get speech-to-text service status."""
    try:
        import torch
        
        status = {
            'available': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
            'supported_formats': list(ALLOWED_EXTENSIONS)
        }
        
        if torch.cuda.is_available():
            status['cuda_device_count'] = torch.cuda.device_count()
            status['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        return jsonify({
            'status': status,
            'success': True
        })
    
    except Exception as e:
        logger.error(f"Error getting speech status: {e}")
        return jsonify({
            'available': False,
            'error': str(e),
            'success': False
        }), 500