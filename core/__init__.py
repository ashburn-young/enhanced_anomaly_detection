"""
Core package initialization
Provides centralized access to configuration and kernel
"""

from .config import (
    SemanticKernelConfig,
    get_config,
    get_kernel,
    get_service_status,
    reset_kernel
)

__all__ = [
    "SemanticKernelConfig",
    "get_config", 
    "get_kernel",
    "get_service_status",
    "reset_kernel"
]
