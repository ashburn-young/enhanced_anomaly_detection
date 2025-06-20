"""
Core configuration for Semantic Kernel integration
Provides centralized configuration and kernel initialization
"""

import os
import logging
from typing import Dict, Any, Optional
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding, OpenAIChatCompletion, OpenAITextEmbedding

logger = logging.getLogger(__name__)

class SemanticKernelConfig:
    """Configuration manager for Semantic Kernel"""
    
    def __init__(self):
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        # Support both variable names for compatibility  
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        self.azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Fallback to OpenAI if Azure not configured
        self.use_azure = bool(self.azure_openai_endpoint and self.azure_openai_api_key)
        self.use_openai = bool(self.openai_api_key and not self.use_azure)
        
    def is_configured(self) -> bool:
        """Check if any AI service is properly configured"""
        return self.use_azure or self.use_openai
    
    def get_status(self) -> Dict[str, Any]:
        """Get configuration status"""
        return {
            "configured": self.is_configured(),
            "azure_available": self.use_azure,
            "openai_available": self.use_openai,
            "azure_endpoint": bool(self.azure_openai_endpoint),
            "azure_key": bool(self.azure_openai_api_key),
            "openai_key": bool(self.openai_api_key)
        }

# Global instances
_config = None
_kernel = None

def get_config() -> SemanticKernelConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = SemanticKernelConfig()
    return _config

def get_kernel() -> Optional[Kernel]:
    """Get configured Semantic Kernel instance"""
    global _kernel
    if _kernel is not None:
        return _kernel
        
    config = get_config()
    if not config.is_configured():
        logger.warning("No AI service configured for Semantic Kernel")
        return None
    
    try:
        _kernel = Kernel()
        
        if config.use_azure:
            # Azure OpenAI services
            chat_service = AzureChatCompletion(
                deployment_name=config.azure_openai_deployment,
                endpoint=config.azure_openai_endpoint,
                api_key=config.azure_openai_api_key
            )
            
            embedding_service = AzureTextEmbedding(
                deployment_name=config.azure_openai_embedding_deployment,
                endpoint=config.azure_openai_endpoint,
                api_key=config.azure_openai_api_key
            )
            
            _kernel.add_service(chat_service)
            _kernel.add_service(embedding_service)
            
        elif config.use_openai:
            # OpenAI services
            chat_service = OpenAIChatCompletion(
                ai_model_id="gpt-4",
                api_key=config.openai_api_key
            )
            
            embedding_service = OpenAITextEmbedding(
                ai_model_id="text-embedding-ada-002",
                api_key=config.openai_api_key
            )
            
            _kernel.add_service(chat_service)
            _kernel.add_service(embedding_service)
        
        logger.info(f"Semantic Kernel initialized with {'Azure' if config.use_azure else 'OpenAI'}")
        return _kernel
        
    except Exception as e:
        logger.error(f"Failed to initialize Semantic Kernel: {e}")
        return None

def get_service_status() -> Dict[str, Any]:
    """Get service status information"""
    config = get_config()
    kernel = get_kernel()
    
    return {
        "config": config.get_status(),
        "kernel_available": kernel is not None,
        "services_loaded": len(kernel.services) if kernel else 0
    }

def reset_kernel():
    """Reset kernel instance (useful for testing)"""
    global _kernel
    _kernel = None
