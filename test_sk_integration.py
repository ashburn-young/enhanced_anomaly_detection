#!/usr/bin/env python3
"""
Test script to diagnose Semantic Kernel agent integration
"""

import sys
import os

print("=== Semantic Kernel Agent Integration Test ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

# Test 1: Check semantic kernel installation
print("1. Testing semantic_kernel import...")
try:
    import semantic_kernel
    from semantic_kernel import Kernel
    print("✓ semantic_kernel imported successfully")
    print(f"  Version: {semantic_kernel.__version__}")
except ImportError as e:
    print(f"❌ semantic_kernel import failed: {e}")
    sys.exit(1)

# Test 2: Check core module
print("\n2. Testing core module...")
try:
    from core import get_kernel, get_config, get_service_status
    print("✓ core module imported successfully")
    
    # Test configuration
    config = get_config()
    print(f"✓ Configuration loaded: {config.is_configured()}")
    
    # Test kernel
    kernel = get_kernel()
    print(f"✓ Kernel status: {'Available' if kernel else 'Not configured'}")
    
    # Service status
    status = get_service_status()
    print(f"✓ Service status: {status}")
    
except Exception as e:
    print(f"❌ core module error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check base agents
print("\n3. Testing base agents...")
try:
    from agents.base import BaseAgent, SemanticKernelAgent, AgentOrchestrator
    print("✓ base agents imported successfully")
except Exception as e:
    print(f"❌ base agents error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check analysis agents
print("\n4. Testing analysis agents...")
try:
    from agents.analysis import ContextAgent, EmbeddingAgent
    print("✓ analysis agents imported successfully")
except Exception as e:
    print(f"❌ analysis agents error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check agent manager
print("\n5. Testing agent manager...")
try:
    from agents import AgentManager
    print("✓ AgentManager imported successfully")
    
    manager = AgentManager()
    print("✓ AgentManager instance created")
    
    init_result = manager.initialize()
    print(f"✓ AgentManager initialized: {init_result}")
    
except Exception as e:
    print(f"❌ AgentManager error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
