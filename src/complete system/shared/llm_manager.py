"""
LLM Manager with timeout support, GPU memory management, and fallback mechanisms
"""

import logging
import gc
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import timeout_decorator
import psutil

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM loading with timeout, memory management, and fallback support"""
    
    def __init__(self):
        self.cached_models: Dict[str, Any] = {}
        self.cached_tokenizers: Dict[str, Any] = {}
        self.fallback_models = {
            "microsoft/Phi-3-mini-4k-instruct": "microsoft/DialoGPT-medium",
            "Qwen/Qwen2.5-0.5B-Instruct": "microsoft/DialoGPT-medium",
        }
    
    def clear_gpu_memory(self):
        """Clear GPU memory before loading new models"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 Cleared CUDA cache")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("🧹 Cleared MPS cache")
            
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            memory_info = psutil.virtual_memory()
            logger.info(f"💾 System memory: {memory_info.percent}% used ({memory_info.available / 1024**3:.1f}GB available)")
            
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")
    
    @timeout_decorator.timeout(300)  # 5-minute timeout
    def _load_model_with_timeout(self, model_name: str) -> tuple:
        """Load model with timeout protection"""
        logger.info(f"⏱️  Loading model {model_name} (5-minute timeout)")
        
        # Clear memory before loading
        self.clear_gpu_memory()
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load model with appropriate device settings
        device_map = "auto"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Use MPS on M1 Macs if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_map = None  # MPS doesn't support device_map="auto"
            torch_dtype = torch.float32  # MPS prefers float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        
        # Move to MPS if available and device_map wasn't used
        if device_map is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            model = model.to('mps')
        
        # Resize embeddings if we added pad token
        if tokenizer.pad_token not in tokenizer.get_vocab():
            model.resize_token_embeddings(len(tokenizer))
        
        model.eval()
        logger.info(f"✅ Successfully loaded {model_name}")
        return tokenizer, model
    
    def load_model(self, model_name: str) -> tuple:
        """
        Load model with timeout and fallback support
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (tokenizer, model)
            
        Raises:
            Exception: If both primary and fallback models fail
        """
        # Check cache first
        if model_name in self.cached_models:
            logger.info(f"📦 Using cached model: {model_name}")
            return self.cached_tokenizers[model_name], self.cached_models[model_name]
        
        try:
            # Try to load primary model
            tokenizer, model = self._load_model_with_timeout(model_name)
            
            # Cache the loaded model
            self.cached_tokenizers[model_name] = tokenizer
            self.cached_models[model_name] = model
            
            return tokenizer, model
            
        except timeout_decorator.TimeoutError:
            logger.error(f"⏰ Model loading timed out: {model_name}")
            return self._fallback_to_smaller_model(model_name)
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {model_name} - {e}")
            return self._fallback_to_smaller_model(model_name)
    
    def _fallback_to_smaller_model(self, original_model: str) -> tuple:
        """Fallback to a smaller, more reliable model"""
        fallback_model = self.fallback_models.get(original_model, "microsoft/DialoGPT-medium")
        
        logger.warning(f"🔄 Falling back to smaller model: {fallback_model}")
        
        # Check if fallback is already cached
        if fallback_model in self.cached_models:
            logger.info(f"📦 Using cached fallback model: {fallback_model}")
            return self.cached_tokenizers[fallback_model], self.cached_models[fallback_model]
        
        try:
            # Try to load fallback model (also with timeout)
            tokenizer, model = self._load_model_with_timeout(fallback_model)
            
            # Cache the fallback model
            self.cached_tokenizers[fallback_model] = tokenizer
            self.cached_models[fallback_model] = model
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"❌ Fallback model also failed: {fallback_model} - {e}")
            raise Exception(f"Both primary ({original_model}) and fallback ({fallback_model}) models failed to load")
    
    def unload_model(self, model_name: str):
        """Unload a specific model to free memory"""
        if model_name in self.cached_models:
            del self.cached_models[model_name]
            del self.cached_tokenizers[model_name]
            self.clear_gpu_memory()
            logger.info(f"🗑️  Unloaded model: {model_name}")
    
    def unload_all_models(self):
        """Unload all cached models"""
        self.cached_models.clear()
        self.cached_tokenizers.clear()
        self.clear_gpu_memory()
        logger.info("🗑️  Unloaded all models")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
            "memory_percent": psutil.virtual_memory().percent
        }
        
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_current_device"] = torch.cuda.current_device()
            
        return info
    
    def log_system_info(self):
        """Log current system and device information"""
        info = self.get_device_info()
        logger.info("🖥️  System Information:")
        logger.info(f"   CUDA Available: {info['cuda_available']}")
        logger.info(f"   MPS Available: {info['mps_available']}")
        logger.info(f"   CPU Cores: {info['cpu_count']}")
        logger.info(f"   Memory: {info['memory_available_gb']:.1f}GB available ({info['memory_percent']:.1f}% used)")
        
        if info["cuda_available"]:
            logger.info(f"   CUDA Devices: {info['cuda_device_count']}")


# Global instance for reuse
_llm_manager = None

def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager