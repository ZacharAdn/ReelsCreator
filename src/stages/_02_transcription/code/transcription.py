"""
Transcription module using Whisper with enhanced multilingual support
"""

import logging
import torch
from typing import List, Dict, Any
from pathlib import Path
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared.models import Segment
import math

# Try faster-whisper first (for Hebrew models), then whisper-timestamped, then openai-whisper
FASTER_WHISPER_AVAILABLE = False
WHISPER_TIMESTAMPED = False

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ faster-whisper available for Hebrew-optimized models")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.info("⚠️  faster-whisper not available, Hebrew models will fall back to standard whisper")

try:
    import whisper_timestamped as whisper
    WHISPER_TIMESTAMPED = True
    logger.info("Using whisper-timestamped for enhanced multilingual support")
except ImportError:
    import whisper
    WHISPER_TIMESTAMPED = False
    logger.warning("Using standard openai-whisper (consider installing whisper-timestamped for better multilingual support)")

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Handles audio transcription using Whisper with multilingual support"""
    
    def __init__(self, model_name: str = "base", primary_language: str = "he", 
                 smart_model_selection: bool = True, force_model: bool = False,
                 force_cpu: bool = False):
        """
        Initialize Whisper transcriber with manual model control
        
        Args:
            model_name: Model name (tiny, base, small, medium, large, large-v3, large-v3-turbo, ivrit-v2-d4, etc.)
            primary_language: Primary language for transcription
            smart_model_selection: Enable automatic model selection based on audio length
            force_model: Force specified model, ignore smart selection
            force_cpu: Force CPU processing, disable MPS/CUDA acceleration
        """
        self.model_name = model_name
        self.primary_language = primary_language
        self.smart_model_selection = smart_model_selection and not force_model
        self.force_model = force_model
        self.force_cpu = force_cpu
        self.model = None
        self.faster_whisper_model = None  # For faster-whisper models
        self.use_faster_whisper = False   # Flag to track which backend is used
        self.actual_model_used = None     # Track which model was actually used
        self.device = self._setup_device()
        
        # Map common model aliases to actual model names
        self.model_mapping = self._get_model_mapping()
        
        logger.info(f"Initializing transcriber: {model_name} (force={force_model}) for language: {primary_language}")
        
        if force_model:
            logger.info(f"🔒 Model selection locked to: {model_name} (smart selection disabled)")
        elif smart_model_selection:
            logger.info(f"🤖 Smart model selection enabled (base model: {model_name})")
    def _get_model_mapping(self) -> dict:
        """Get mapping of model aliases to actual model names"""
        return {
            # Standard OpenAI Whisper models
            "tiny": "tiny",
            "base": "base", 
            "small": "small",
            "medium": "medium",
            "large": "large",
            "large-v2": "large-v2",
            "large-v3": "large-v3",
            "large-v3-turbo": "turbo",  # Maps to Whisper turbo if available
            
            # Hebrew-optimized models (Ivrit.AI)
            "ivrit-v2-d4": "ivrit-ai/faster-whisper-v2-d4",
            "ivrit-v2-d3-e3": "ivrit-ai/faster-whisper-v2-d3-e3",
            
            # Aliases
            "hebrew": "ivrit-v2-d4",  # Default Hebrew model
            "hebrew-latest": "ivrit-v2-d4",
            "turbo": "turbo",
            "auto": "auto"
        }
    
    def _setup_device(self) -> torch.device:
        """Setup device with CPU fallback option"""
        if self.force_cpu:
            logger.info("🔧 Forced CPU mode: Using CPU for transcription (MPS/CUDA disabled)")
            return torch.device("cpu")
        elif torch.backends.mps.is_available():
            try:
                # Test MPS availability more thoroughly
                test_tensor = torch.ones(1, device="mps")
                logger.info("✅ Using M1 GPU (MPS) for transcription")
                return torch.device("mps")
            except Exception as e:
                logger.warning(f"⚠️  MPS device test failed: {e}")
                logger.info("🔄 Falling back to CPU for transcription")
                return torch.device("cpu")
        else:
            logger.info("Using CPU for transcription")
            return torch.device("cpu")
    
    def load_model(self):
        """Load Whisper model with error handling, Hebrew model support, and fallback"""
        if self.model is None and self.faster_whisper_model is None:
            model_to_load = self.model_name
            
            try:
                # Handle Hebrew-optimized models with faster-whisper
                if self._is_hebrew_model(model_to_load):
                    logger.info(f"🇮🇱 Loading Hebrew-optimized model: {model_to_load}")
                    success = self._load_hebrew_model(model_to_load)
                    if success:
                        logger.info("✅ Hebrew model loaded successfully")
                        return
                    else:
                        logger.warning("🔄 Hebrew model loading failed, trying standard models")
                    
                # Handle latest Whisper models
                if model_to_load == "large-v3-turbo" or model_to_load == "turbo":
                    model_to_load = self._get_turbo_model_name()
                    logger.info(f"🚀 Loading Whisper Turbo model: {model_to_load}")
                
                logger.info(f"Loading Whisper model: {model_to_load}")
                self.model = whisper.load_model(model_to_load, device=self.device)
                logger.info("✅ Whisper model loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load model '{model_to_load}': {e}")
                
                # Check if this is an MPS backend issue and try CPU fallback
                if ("SparseMPS" in str(e) or "aten::" in str(e)) and not self.force_cpu and str(self.device) == "mps":
                    logger.warning("⚠️  Detected MPS backend compatibility issue")
                    logger.info("🔄 Automatically falling back to CPU device")
                    self.device = torch.device("cpu")
                    
                    try:
                        if self._is_hebrew_model(model_to_load):
                            self.model = self._load_hebrew_model(model_to_load)
                        else:
                            self.model = whisper.load_model(model_to_load, device=self.device)
                        logger.info("✅ Model loaded successfully with CPU fallback")
                        return
                    except Exception as cpu_error:
                        logger.error(f"❌ CPU fallback also failed: {cpu_error}")
                
                # Fallback logic for Hebrew models
                if self._is_hebrew_model(model_to_load):
                    logger.warning("🔄 Hebrew model failed, falling back to large model")
                    try:
                        self.model = whisper.load_model("large", device=self.device)
                        self.actual_model_used = "large"
                        logger.info("✅ Fallback to large model successful")
                        return
                    except Exception as fallback_error:
                        logger.error(f"❌ Large model fallback failed: {fallback_error}")
                
                # Standard fallback to smaller models
                fallback_models = ["base", "tiny"]
                for fallback in fallback_models:
                    if fallback != model_to_load:
                        try:
                            logger.warning(f"🔄 Attempting fallback to {fallback} model...")
                            self.model = whisper.load_model(fallback, device=self.device)
                            self.actual_model_used = fallback
                            logger.info(f"✅ Successfully loaded fallback model: {fallback}")
                            return
                        except Exception as fallback_error:
                            logger.error(f"❌ Fallback to {fallback} failed: {fallback_error}")
                            # If MPS error on fallback model too, try CPU for this model
                            if ("SparseMPS" in str(fallback_error) or "aten::" in str(fallback_error)) and str(self.device) == "mps":
                                try:
                                    logger.info(f"🔄 Trying {fallback} with CPU device...")
                                    cpu_device = torch.device("cpu")
                                    self.model = whisper.load_model(fallback, device=cpu_device)
                                    self.device = cpu_device  # Update device for future operations
                                    self.actual_model_used = fallback
                                    logger.info(f"✅ {fallback} model loaded with CPU fallback")
                                    return
                                except Exception as cpu_fallback_error:
                                    logger.error(f"❌ CPU fallback for {fallback} failed: {cpu_fallback_error}")
                
                # If all fallbacks fail, raise the original error
                raise RuntimeError(f"Failed to load any Whisper model. Original error: {e}")
    
    def _is_hebrew_model(self, model_name: str) -> bool:
        """Check if model is a Hebrew-optimized variant"""
        hebrew_indicators = ["ivrit", "hebrew", "ivrit-ai"]
        return any(indicator in model_name.lower() for indicator in hebrew_indicators)
    
    def _load_hebrew_model(self, model_name: str) -> bool:
        """
        Load Hebrew-optimized models with faster-whisper or fallback to standard
        
        Args:
            model_name: Hebrew model name (e.g., 'ivrit-v2-d4', 'hebrew')
            
        Returns:
            bool: True if successfully loaded, False if fallback needed
        """
        # Resolve model name to actual identifier
        actual_model_name = self._resolve_hebrew_model_name(model_name)
        
        # Try faster-whisper first if available
        if FASTER_WHISPER_AVAILABLE:
            try:
                logger.info(f"🚀 Loading Hebrew model with faster-whisper: {actual_model_name}")
                
                # Configure device for faster-whisper
                device_name = "cpu" if self.force_cpu or str(self.device) == "cpu" else "cuda"
                if str(self.device) == "mps":
                    device_name = "cpu"  # faster-whisper doesn't support MPS, use CPU
                    logger.info("🔄 Using CPU for faster-whisper (MPS not supported)")
                
                # Load the model
                self.faster_whisper_model = FasterWhisperModel(
                    actual_model_name,
                    device=device_name,
                    compute_type="float16" if device_name == "cuda" else "float32"
                )
                self.use_faster_whisper = True
                self.actual_model_used = actual_model_name
                logger.info(f"✅ Hebrew model loaded with faster-whisper: {actual_model_name}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load Hebrew model with faster-whisper: {e}")
                logger.warning("🔄 Falling back to standard whisper")
        
        # Fallback to standard whisper with Hebrew hints
        try:
            logger.info("🔄 Loading Hebrew model with standard whisper (large model + Hebrew language)")
            self.model = whisper.load_model("large", device=self.device)
            self.use_faster_whisper = False
            self.actual_model_used = "large (Hebrew fallback)"
            logger.info("✅ Hebrew fallback model loaded (standard whisper large)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Hebrew fallback model: {e}")
            return False
    
    def _resolve_hebrew_model_name(self, model_name: str) -> str:
        """Resolve Hebrew model aliases to actual model names"""
        hebrew_model_map = {
            "ivrit-v2-d4": "ivrit-ai/faster-whisper-v2-d4",
            "ivrit-v2-d3-e3": "ivrit-ai/faster-whisper-v2-d3-e3", 
            "hebrew": "ivrit-ai/faster-whisper-v2-d4",  # Default to latest
            "hebrew-latest": "ivrit-ai/faster-whisper-v2-d4"
        }
        return hebrew_model_map.get(model_name, model_name)
    
    def _get_turbo_model_name(self) -> str:
        """Get the correct turbo model name based on availability"""
        # Try different turbo model names that might be available
        turbo_variants = ["turbo", "large-v3-turbo", "large-v3"]
        
        for variant in turbo_variants:
            try:
                # Check if model exists (this is a simple check)
                whisper.available_models()
                return variant
            except:
                continue
        
        # Fallback to large-v3 or large
        logger.warning("🔄 Turbo model not available, using large-v3")
        return "large-v3"
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file with multilingual support
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        self.load_model()
        
        logger.info(f"Transcribing: {audio_path}")
        start_time = time.time()
        
        try:
            if self.use_faster_whisper and self.faster_whisper_model:
                # Faster-whisper transcription (for Hebrew models)
                segments, info = self.faster_whisper_model.transcribe(
                    audio_path,
                    language=self.primary_language,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Convert faster-whisper format to standard format
                result = {
                    "text": "",
                    "language": info.language,
                    "segments": []
                }
                
                for segment in segments:
                    result["segments"].append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "avg_logprob": segment.avg_logprob,
                        "words": [
                            {
                                "start": word.start,
                                "end": word.end,
                                "word": word.word,
                                "probability": word.probability
                            } for word in segment.words
                        ] if segment.words else []
                    })
                    result["text"] += segment.text + " "
                
                result["text"] = result["text"].strip()
                
            elif WHISPER_TIMESTAMPED:
                # Enhanced multilingual transcription with whisper-timestamped
                result = whisper.transcribe(
                    self.model,
                    audio_path,
                    language=self.primary_language,
                    verbose=True,
                    detect_disfluencies=True  # Better for mixed languages
                )
            else:
                # Standard transcription with openai-whisper
                result = self.model.transcribe(
                    audio_path,
                    language=self.primary_language,
                    word_timestamps=True,
                    verbose=True
                )
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def extract_segments(self, transcription_result: Dict[str, Any]) -> List[Segment]:
        """
        Extract segments from transcription result
        
        Args:
            transcription_result: Result from Whisper transcription
            
        Returns:
            List of Segment objects
        """
        segments = []
        
        for segment_data in transcription_result.get("segments", []):
            # Convert Whisper avg_logprob (typically negative) to a pseudo-confidence in [0,1]
            raw_logprob = float(segment_data.get("avg_logprob", -2.0))
            # Clamp to a sensible range to avoid exp underflow, then exponentiate
            raw_logprob = max(-10.0, min(0.0, raw_logprob))
            confidence_prob = math.exp(raw_logprob)

            segment = Segment(
                start_time=segment_data["start"],
                end_time=segment_data["end"],
                text=segment_data["text"].strip(),
                confidence=confidence_prob
            )
            segments.append(segment)
        
        logger.info(f"Extracted {len(segments)} segments from transcription")
        return segments
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            return duration
        except ImportError:
            # Fallback using moviepy if librosa not available
            try:
                from moviepy.editor import AudioFileClip
                with AudioFileClip(audio_path) as audio:
                    return audio.duration
            except ImportError:
                logger.warning("Cannot determine audio duration - librosa and moviepy not available")
                return 600.0  # Default to 10 minutes if we can't determine duration
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
            return 600.0  # Default fallback
    
    def select_optimal_model(self, audio_duration: float) -> str:
        """
        Select optimal Whisper model based on audio duration
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Optimal model name
        """
        duration_minutes = audio_duration / 60.0
        
        if duration_minutes < 3:  # Very short videos
            model = "tiny"
            reason = f"short video ({duration_minutes:.1f}m) - prioritizing speed"
        elif duration_minutes < 10:  # Short videos
            model = "base"
            reason = f"medium video ({duration_minutes:.1f}m) - balanced speed/accuracy"
        elif duration_minutes < 30:  # Medium videos
            model = "small"
            reason = f"longer video ({duration_minutes:.1f}m) - prioritizing accuracy"
        else:  # Long videos
            model = "base"  # Use base for very long videos to avoid memory issues
            reason = f"very long video ({duration_minutes:.1f}m) - balanced for memory efficiency"
        
        logger.info(f"🎯 Selected Whisper model '{model}' for {reason}")
        return model
    
    def _determine_model_to_use(self, audio_path: str) -> str:
        """
        Determine which model to use based on configuration and content
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Model name to use
        """
        # If model is forced, use it directly
        if self.force_model and self.model_name != "auto":
            resolved_model = self._resolve_model_name(self.model_name)
            logger.info(f"🔒 Using forced model: {resolved_model}")
            return resolved_model
        
        # If smart selection is disabled, use the specified model
        if not self.smart_model_selection and self.model_name != "auto":
            resolved_model = self._resolve_model_name(self.model_name)
            logger.info(f"🎯 Using specified model: {resolved_model}")
            return resolved_model
        
        # Smart model selection enabled - choose based on audio duration
        if self.smart_model_selection and self.model_name == "auto":
            audio_duration = self.get_audio_duration(audio_path)
            optimal_model = self.select_optimal_model(audio_duration)
            return self._resolve_model_name(optimal_model)
        
        # Fallback to specified model
        return self._resolve_model_name(self.model_name)
    
    def _resolve_model_name(self, model_name: str) -> str:
        """
        Resolve model name through mapping
        
        Args:
            model_name: Input model name (may be alias)
            
        Returns:
            Resolved model name
        """
        resolved = self.model_mapping.get(model_name, model_name)
        
        # Log Hebrew model usage
        if "ivrit" in resolved.lower() or model_name in ["hebrew", "hebrew-latest"]:
            logger.info(f"🇮🇱 Using Hebrew-optimized model: {resolved}")
            
        return resolved
    
    def _validate_model_availability(self, model_name: str) -> bool:
        """
        Check if a model is available for use
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if model is available
        """
        # For Hebrew models, check if they would need additional setup
        if "ivrit-ai" in model_name:
            logger.warning(f"⚠️  Hebrew model {model_name} may require additional setup (Ivrit.AI)")
            return True  # Assume available for now, will fail gracefully during loading
        
        # Standard Whisper models should always be available
        return True
    
    def process_audio_file(self, audio_path: str) -> List[Segment]:
        """
        Complete audio processing pipeline with smart model selection
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of Segment objects
        """
        # Validate file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Determine which model to use
        model_to_use = self._determine_model_to_use(audio_path)
        
        # Update model if different from current
        if self.actual_model_used != model_to_use:
            logger.info(f"🔄 Switching from {self.actual_model_used or 'None'} to {model_to_use} model")
            self.actual_model_used = model_to_use
            self.model = None  # Force reload with new model
            
            # Temporarily update model_name for loading
            original_model_name = self.model_name
            self.model_name = model_to_use
            self.load_model()
            self.model_name = original_model_name  # Restore original
        
        # Transcribe
        result = self.transcribe(audio_path)
        
        # Extract segments
        segments = self.extract_segments(result)
        
        return segments 