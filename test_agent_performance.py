#!/usr/bin/env python3
"""
Agent Performance Analysis
Compare individual agent performance and combinations
"""

import pandas as pd
import numpy as np
from agent_orchestrator import AgentOrchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data with known anomalies"""
    np.random.seed(42)
    
    # Normal data
    normal_data = {
        'price': np.random.normal(10, 2, 95),  # Normal around $10
        'sales': np.random.normal(100, 20, 95),  # Normal around 100 units
        'weight': np.random.normal(1.5, 0.3, 95),  # Normal around 1.5 oz
        'rating': np.random.normal(4.0, 0.5, 95),  # Normal around 4.0 stars
        'description': ['Normal Product'] * 95
    }
    
    # Add clear anomalies
    anomaly_data = {
        'price': [50.0, 0.99, 25.0, 1.0, 30.0],  # Price anomalies
        'sales': [500, 5, 300, 2, 400],  # Sales anomalies  
        'weight': [5.0, 0.1, 3.0, 0.05, 4.0],  # Weight anomalies
        'rating': [1.0, 5.0, 0.5, 4.9, 1.2],  # Rating anomalies
        'description': [
            'Extremely expensive premium luxury item with gold plating',
            'Clearance item',
            'Bulk family size package',
            'Sample/trial size',
            'Limited edition collector item'
        ]
    }
    
    # Combine data
    data = {}
    for key in normal_data:
        data[key] = list(normal_data[key]) + list(anomaly_data[key])
    
    df = pd.DataFrame(data)
    df.index = range(len(df))
    
    print(f"Created test dataset with {len(df)} records")
    print(f"Known anomaly indices: {list(range(95, 100))}")
    print(f"Sample data:\n{df.tail()}")
    
    return df

def test_individual_agents():
    """Test each agent individually"""
    print("\n=== INDIVIDUAL AGENT TESTING ===")
    
    df = create_test_data()
    orchestrator = AgentOrchestrator()
    
    # Get available agents
    available_agents = orchestrator.list_agents()
    print(f"\nAvailable agents: {available_agents}")
    
    agent_results = {}
    
    # Test each agent individually
    for agent_name in available_agents:
        print(f"\n--- Testing {agent_name} ---")
        
        try:
            options = {
                'selected_agents': [agent_name],
                'custom_prompt': 'Find all types of anomalies in this retail data',
                'sensitivity': 0.5,
                'max_anomalies': 20
            }
            
            results = orchestrator.analyze_with_agents(df, options)
            agent_result = results.get(agent_name, {})
            
            anomaly_count = len(agent_result.get('anomalies', []))
            agent_results[agent_name] = {
                'count': anomaly_count,
                'anomalies': agent_result.get('anomalies', []),
                'analysis': agent_result.get('analysis', 'No analysis'),
                'error': agent_result.get('error', None)
            }
            
            if agent_result.get('error'):
                print(f"❌ {agent_name}: ERROR - {agent_result['error']}")
            else:
                print(f"✅ {agent_name}: Found {anomaly_count} anomalies")
                
                # Show first few anomalies
                for i, anomaly in enumerate(agent_result.get('anomalies', [])[:3]):
                    idx = anomaly.get('index', 'N/A')
                    col = anomaly.get('column', 'N/A')
                    val = anomaly.get('value', 'N/A')
                    conf = anomaly.get('confidence', 0.0)
                    print(f"  Anomaly {i+1}: Row {idx}, {col}={val}, confidence={conf:.2f}")
                    
        except Exception as e:
            print(f"❌ {agent_name}: EXCEPTION - {e}")
            agent_results[agent_name] = {'count': 0, 'error': str(e)}
    
    return agent_results, df

def test_agent_combinations():
    """Test different agent combinations"""
    print("\n=== AGENT COMBINATION TESTING ===")
    
    df = create_test_data()
    orchestrator = AgentOrchestrator()
    
    combinations = [
        {
            'name': 'High Performance Trio',
            'agents': ['statistical_agent', 'ai_agent', 'enhanced_statistical_agent']
        },
        {
            'name': 'All Available Agents',
            'agents': orchestrator.list_agents()
        },
        {
            'name': 'Statistical Only',
            'agents': ['statistical_agent', 'enhanced_statistical_agent']
        },
        {
            'name': 'AI Focused',
            'agents': ['ai_agent', 'embedding_agent']
        },
        {
            'name': 'Semantic Kernel Agents',
            'agents': ['embedding_agent']  # Will add context_agent if available
        }
    ]
    
    # Add context_agent if available
    if 'context_agent' in orchestrator.list_agents():
        combinations[4]['agents'].append('context_agent')
    
    combination_results = {}
    
    for combo in combinations:
        print(f"\n--- Testing: {combo['name']} ---")
        print(f"Agents: {combo['agents']}")
        
        try:
            options = {
                'selected_agents': combo['agents'],
                'custom_prompt': 'Find all types of anomalies including pricing errors, unusual sales patterns, and data quality issues',
                'sensitivity': 0.5,
                'max_anomalies': 30
            }
            
            results = orchestrator.analyze_with_agents(df, options)
            
            # Count total anomalies across all agents
            total_anomalies = 0
            unique_indices = set()
            
            for agent_name in combo['agents']:
                agent_result = results.get(agent_name, {})
                anomalies = agent_result.get('anomalies', [])
                total_anomalies += len(anomalies)
                
                # Track unique anomaly indices
                for anomaly in anomalies:
                    if 'index' in anomaly:
                        unique_indices.add(anomaly['index'])
                
                if agent_result.get('error'):
                    print(f"  ❌ {agent_name}: {agent_result['error']}")
                else:
                    print(f"  ✅ {agent_name}: {len(anomalies)} anomalies")
            
            combination_results[combo['name']] = {
                'total_anomalies': total_anomalies,
                'unique_anomalies': len(unique_indices),
                'unique_indices': unique_indices,
                'agents_used': combo['agents'],
                'summary': results.get('orchestrator_summary', 'No summary')
            }
            
            print(f"📊 Total anomalies: {total_anomalies}")
            print(f"📊 Unique anomaly records: {len(unique_indices)}")
            print(f"📊 Known anomalies detected: {len(unique_indices.intersection(set(range(95, 100))))}/5")
            
        except Exception as e:
            print(f"❌ Combination failed: {e}")
            combination_results[combo['name']] = {'error': str(e)}
    
    return combination_results

def analyze_agent_effectiveness():
    """Analyze why some combinations might find fewer anomalies"""
    print("\n=== AGENT EFFECTIVENESS ANALYSIS ===")
    
    # Test individual agents
    individual_results, df = test_individual_agents()
    
    # Test combinations
    combination_results = test_agent_combinations()
    
    print("\n=== SUMMARY & INSIGHTS ===")
    
    # Individual agent summary
    print("\n📈 Individual Agent Performance:")
    for agent_name, result in individual_results.items():
        if 'error' not in result or not result['error']:
            print(f"  {agent_name}: {result['count']} anomalies")
        else:
            print(f"  {agent_name}: ERROR - {result.get('error', 'Unknown')}")
    
    # Combination summary
    print("\n📈 Combination Performance:")
    for combo_name, result in combination_results.items():
        if 'error' not in result:
            print(f"  {combo_name}: {result['unique_anomalies']} unique anomalies")
        else:
            print(f"  {combo_name}: ERROR - {result.get('error', 'Unknown')}")
    
    # Insights
    print("\n💡 INSIGHTS:")
    
    # Check if embedding_agent or visual_agent are working
    problem_agents = []
    for agent_name, result in individual_results.items():
        if result.get('error') or result['count'] == 0:
            problem_agents.append(agent_name)
    
    if problem_agents:
        print(f"⚠️  Agents with issues: {problem_agents}")
        print("   These agents may be reducing overall anomaly detection")
    
    # Check for overlap vs dilution
    best_individual = max(individual_results.items(), 
                         key=lambda x: x[1]['count'] if 'error' not in x[1] else 0)
    print(f"🏆 Best individual agent: {best_individual[0]} ({best_individual[1]['count']} anomalies)")
    
    # Find best combination
    best_combo = max(combination_results.items(),
                    key=lambda x: x[1].get('unique_anomalies', 0))
    print(f"🏆 Best combination: {best_combo[0]} ({best_combo[1].get('unique_anomalies', 0)} unique anomalies)")
    
    return individual_results, combination_results

if __name__ == "__main__":
    print("🔍 ANOMALY DETECTION AGENT ANALYSIS")
    print("=" * 50)
    
    try:
        individual_results, combination_results = analyze_agent_effectiveness()
        
        print("\n✅ Analysis complete!")
        print("\nNext steps:")
        print("1. Check which agents have errors")
        print("2. Focus on combinations that maximize unique anomaly detection")
        print("3. Consider agent-specific strengths for different data types")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
