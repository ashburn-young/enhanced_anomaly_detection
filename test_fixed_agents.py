#!/usr/bin/env python3
"""
Test script to verify the embedding_agent and visual_agent fix
"""

import pandas as pd
import numpy as np
from agent_orchestrator import AgentOrchestrator

# Create test data
data = pd.DataFrame({
    'sales': [100, 95, 102, 98, 300, 105, 97],  # One outlier at index 4
    'price': [10.5, 10.2, 10.8, 10.1, 10.3, 10.7, 10.4],
    'product': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
})

print("=== Testing Fixed Agent Integration ===")
print(f"Test data shape: {data.shape}")
print()

# Test the orchestrator
orchestrator = AgentOrchestrator()
agents = orchestrator.list_agents()
print(f"Available agents: {agents}")
print()

# Test the specific agents that were failing
test_agents = ['embedding_agent', 'visual_agent']

for agent_name in test_agents:
    print(f"Testing {agent_name}...")
    
    if agent_name in agents:
        try:
            # Test with the specific options that the Streamlit app would use
            options = {
                'selected_agents': [agent_name],
                'max_anomalies': 10,
                'sensitivity': 'medium'
            }
            
            # This should not fail with 'detect_anomalies' error anymore
            result = orchestrator.analyze_with_agents(data, options)
            
            if agent_name in result:
                agent_result = result[agent_name]
                if 'error' in agent_result:
                    print(f"❌ {agent_name}: Error - {agent_result['error']}")
                else:
                    print(f"✅ {agent_name}: Success!")
                    print(f"   Analysis: {agent_result.get('analysis', 'No analysis')[:100]}...")
                    print(f"   Anomalies found: {len(agent_result.get('anomalies', []))}")
            else:
                print(f"❌ {agent_name}: No result returned")
                
        except Exception as e:
            print(f"❌ {agent_name}: Exception - {e}")
    else:
        print(f"❌ {agent_name}: Not found in available agents")
    
    print()

print("=== Test Complete ===")
