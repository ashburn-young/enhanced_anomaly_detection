#!/usr/bin/env python3
"""
Quick test of the agent detect_anomalies methods
"""

import pandas as pd
import numpy as np

# Create simple test data
data = pd.DataFrame({
    'sales': [100, 95, 102, 98, 500, 105, 97],  # One outlier at index 4
    'price': [10.5, 10.2, 10.8, 10.1, 10.3, 10.7, 10.4],
    'product': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
})

print("=== Quick Agent Test ===")
print(f"Test data:\n{data}")
print()

# Test individual agents
agents_to_test = [
    ('EmbeddingAgent', 'simple_agents'),
    ('VisualAgent', 'simple_agents'),
    ('StatisticalAgent', 'simple_agents')
]

for agent_name, module_name in agents_to_test:
    try:
        # Import the agent
        if module_name == 'simple_agents':
            exec(f"from simple_agents import {agent_name}")
            agent = eval(f"{agent_name}()")
        
        print(f"Testing {agent_name}...")
        
        # Test detect_anomalies method
        if hasattr(agent, 'detect_anomalies'):
            result = agent.detect_anomalies(data)
            print(f"✓ {agent_name}.detect_anomalies() works")
            print(f"  Result keys: {list(result.keys())}")
            if 'anomalies' in result:
                print(f"  Anomalies found: {len(result['anomalies'])}")
        else:
            print(f"❌ {agent_name} has no detect_anomalies method")
            
        print()
        
    except Exception as e:
        print(f"❌ {agent_name} failed: {e}")
        print()

print("=== Test Complete ===")
