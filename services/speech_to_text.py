"""
Speech-to-text service using Hugging Face Transformers with Whisper model.

Required dependencies:
- transformers, torch, librosa, soundfile

For optimal performance:
- ffmpeg: Required for webm and other audio format conversions
  Install via: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)
  or download from https://ffmpeg.org/download.html (Windows)
  
- hf_xet: Improves Hugging Face model download speed
  Install via: pip install huggingface_hub[hf_xet] or pip install hf_xet
"""

import os
import tempfile
import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import soundfile as sf
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeechToTextService:
    """Speech-to-text service using Whisper model from Hugging Face."""
    
    def __init__(self, model_name: str = "openai/whisper-base"):
        """Initialize the speech-to-text service.
        
        Args:
            model_name: Hugging Face model name for Whisper (default: whisper-base)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self._is_initialized = False
        self._check_dependencies()
        
        logger.info(f"Initializing Speech-to-Text service with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
    def _check_dependencies(self):
        """Check if all necessary dependencies are available."""
        try:
            # Check for FFMPEG (needed for webm conversion)
            import shutil
            ffmpeg_available = shutil.which("ffmpeg") is not None
            if not ffmpeg_available:
                logger.warning("FFMPEG not found in PATH. Audio format conversion may not work properly.")
                logger.warning("Please install FFMPEG for better audio format support.")
            else:
                logger.info("FFMPEG found in PATH. Audio format conversion enabled.")
                
            # Check for hf_xet package
            import importlib.util
            hf_xet_spec = importlib.util.find_spec("hf_xet")
            if hf_xet_spec is None:
                logger.warning("hf_xet package not installed. Downloading from Hugging Face will be slower.")
                logger.warning("For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`")
            else:
                logger.info("hf_xet package found. Faster downloads from Hugging Face enabled.")
        except Exception as e:
            logger.warning(f"Error checking dependencies: {e}")
            logger.warning("Some features may not work properly. Install FFMPEG and hf_xet for best performance.")
    
    def _load_model(self) -> None:
        """Load the Whisper model and processor."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            self._is_initialized = True
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Could not initialize speech-to-text service: {e}")
    
    def transcribe_audio(self, audio_file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio file to text.
        
        Args:
            audio_file_path: Path to the audio file
            language: Optional language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Dict containing transcribed text and metadata
        """
        if not self._is_initialized:
            self._load_model()
        
        try:
            # Load and preprocess audio
            audio_data, sample_rate = self._load_audio(audio_file_path)
            
            # Prepare inputs for the model
            inputs = self.processor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            input_features = inputs.input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                if language:
                    # Force specific language
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language)
                    generated_ids = self.model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids
                    )
                else:
                    # Auto-detect language
                    generated_ids = self.model.generate(input_features)
            
            # Decode the transcription
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            logger.info(f"Transcription completed. Length: {len(transcription)} characters")
            
            return {
                "text": transcription.strip(),
                "language": language or "auto",
                "model": self.model_name,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def _load_audio(self, audio_file_path: str) -> tuple:
        """Load and preprocess audio file.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Check if file is webm (which librosa has trouble with)
            file_ext = os.path.splitext(audio_file_path)[1].lower()
            
            if file_ext == '.webm':
                # For webm files, convert to wav first using ffmpeg via librosa's cache
                import subprocess
                import numpy as np
                
                temp_wav = os.path.join(tempfile.gettempdir(), f"whisper_temp_{os.path.basename(audio_file_path)}.wav")
                logger.info(f"Converting webm to wav: {temp_wav}")
                
                try:
                    # Try using ffmpeg directly for conversion (more reliable for webm)
                    subprocess.check_call([
                        "ffmpeg", "-y", "-i", audio_file_path, 
                        "-ar", "16000", "-ac", "1", temp_wav
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Load the converted WAV file
                    audio_data, sample_rate = sf.read(temp_wav)
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)  # Convert to mono if needed
                    
                    logger.info(f"Successfully converted and loaded webm via ffmpeg")
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_wav)
                    except:
                        pass
                        
                    return audio_data, sample_rate
                    
                except Exception as e:
                    logger.warning(f"FFMPEG conversion failed: {e}, trying fallback method")
                    # If ffmpeg fails, try loading with librosa anyway as fallback
            
            # Standard loading for non-webm files or as fallback
            audio_data, sample_rate = librosa.load(
                audio_file_path,
                sr=16000,  # Whisper expects 16kHz
                mono=True   # Convert to mono
            )
            
            logger.debug(f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_file_path}: {e}")
            raise ValueError(f"Could not load audio file: {e}")
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, format: str = "wav", language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio from bytes.
        
        Args:
            audio_bytes: Audio data as bytes
            format: Audio format (wav, mp3, etc.)
            language: Optional language code
            
        Returns:
            Dict containing transcribed text and metadata
        """
        # Log the audio format we're handling
        logger.info(f"Transcribing audio bytes in format: {format}")
        
        # Save bytes to temporary file and transcribe
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
            try:
                temp_file.write(audio_bytes)
                temp_file.flush()
                temp_path = temp_file.name
                
                # Make sure file is properly written before proceeding
                os.fsync(temp_file.fileno())
                
            except Exception as e:
                logger.error(f"Error writing audio bytes to temporary file: {e}")
                return {
                    "text": "",
                    "error": f"Failed to process audio data: {e}",
                    "success": False
                }
                
        # Process outside the with block to ensure file is closed
        try:
            # Check if file exists and has content
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise ValueError(f"Temporary audio file is empty or missing: {temp_path}")
                
            logger.info(f"Temporary audio file created at: {temp_path} ({os.path.getsize(temp_path)} bytes)")
            result = self.transcribe_audio(temp_path, language)
            return result
            
        except Exception as e:
            logger.error(f"Error in audio transcription process: {e}")
            return {
                "text": "",
                "error": f"Transcription error: {e}",
                "success": False
            }
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages.
        
        Returns:
            List of language codes supported by the model
        """
        # Whisper supports these languages
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", 
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", 
            "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", 
            "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", 
            "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", 
            "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", 
            "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", 
            "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_initialized = False
        logger.info("Speech-to-text service cleaned up")


# Global service instance
_speech_service = None


def get_speech_service(model_name: str = "openai/whisper-base") -> SpeechToTextService:
    """Get or create the global speech-to-text service instance.
    
    Args:
        model_name: Hugging Face model name for Whisper
        
    Returns:
        SpeechToTextService instance
    """
    global _speech_service
    
    if _speech_service is None:
        _speech_service = SpeechToTextService(model_name)
    
    return _speech_service