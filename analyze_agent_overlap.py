#!/usr/bin/env python3
"""
Analyze agent overlap and effectiveness for anomaly detection
Investigate why all agents together might find fewer unique anomalies
"""

import pandas as pd
import numpy as np
import json
import logging
from agent_orchestrator import AgentOrchestrator
from data_processing import DataProcessor
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def create_test_dataset_with_clear_anomalies():
    """Create a test dataset with clearly identifiable anomalies"""
    # Create normal data
    np.random.seed(42)
    n_normal = 95
    n_anomalies = 5
    
    # Normal products
    normal_data = {
        'product_id': [f'P{i:03d}' for i in range(n_normal)],
        'price': np.random.normal(15, 3, n_normal),  # Normal around $15
        'sales': np.random.normal(100, 20, n_normal),  # Normal around 100 units
        'weight': np.random.normal(2, 0.5, n_normal),  # Normal around 2 lbs
        'rating': np.random.normal(4, 0.3, n_normal),  # Normal around 4 stars
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_normal),
        'description': [f'Standard product {i}' for i in range(n_normal)]
    }
    
    # Clear anomalies with specific characteristics
    anomaly_data = {
        'product_id': ['A001', 'A002', 'A003', 'A004', 'A005'],
        'price': [150.0, 0.01, 500.0, -5.0, 999.99],  # Price anomalies
        'sales': [1000, 1, 5000, 0, 10000],  # Sales anomalies  
        'weight': [50, 0.001, 100, -1, 0],  # Weight anomalies
        'rating': [0.1, 5.5, -1, 10, 0],  # Rating anomalies
        'category': ['UNKNOWN', 'Electronics', 'Home', 'Books', 'Electronics'],
        'description': [
            'Extremely expensive luxury item',
            'Nearly free clearance item',
            'Bulk commercial product',
            'Defective product with negative price',
            'Ultra premium collector edition'
        ]
    }
    
    # Combine data
    all_data = {}
    for key in normal_data:
        if isinstance(normal_data[key], list):
            all_data[key] = normal_data[key] + anomaly_data[key]
        else:
            # Handle numpy arrays
            all_data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
    
    df = pd.DataFrame(all_data)
    
    # Ensure no negative values where they don't make sense
    df.loc[df['price'] < 0, 'price'] = 0.01
    df.loc[df['sales'] < 0, 'sales'] = 0
    df.loc[df['weight'] < 0, 'weight'] = 0.001
    df.loc[df['rating'] < 0, 'rating'] = 0.1
    df.loc[df['rating'] > 5, 'rating'] = 5.0
    
    return df, list(range(95, 100))  # Anomaly indices

def analyze_agent_detailed_results(df, orchestrator):
    """Get detailed results from each agent"""
    options = {
        'business_focus': 'general_retail',
        'max_anomalies': 20,
        'confidence_threshold': 0.5,
        'include_explanations': True
    }
    
    agent_results = {}
    agent_names = orchestrator.list_agents()
    
    print(f"\n🔍 DETAILED AGENT ANALYSIS")
    print(f"={'='*60}")
    print(f"Dataset shape: {df.shape}")
    print(f"Available agents: {agent_names}")
    
    # Test each agent individually
    for agent_name in agent_names:
        print(f"\n--- Analyzing {agent_name} ---")
        try:
            result = orchestrator.analyze_with_agents(df, {**options, 'selected_agents': [agent_name]})
            agent_result = result.get(agent_name, {})
            
            # Extract anomalies with detailed info
            anomalies = []
            if 'anomalies' in agent_result:
                anomalies = agent_result['anomalies']
            elif 'anomalies_detected' in agent_result:
                anomalies = agent_result['anomalies_detected']
            
            # Analyze anomaly details
            anomaly_details = []
            for i, anomaly in enumerate(anomalies):
                if isinstance(anomaly, dict):
                    detail = {
                        'index': i,
                        'type': anomaly.get('anomaly_type', 'Unknown'),
                        'description': anomaly.get('description', 'No description'),
                        'confidence': anomaly.get('confidence_score', anomaly.get('confidence', 0)),
                        'affected_field': anomaly.get('affected_field', 'Unknown'),
                        'value': anomaly.get('anomalous_value', 'Unknown'),
                        'row_index': anomaly.get('row_index', 'Unknown')
                    }
                    anomaly_details.append(detail)
            
            agent_results[agent_name] = {
                'anomalies': anomalies,
                'anomaly_details': anomaly_details,
                'count': len(anomalies),
                'raw_result': agent_result
            }
            
            print(f"  ✅ {agent_name}: {len(anomalies)} anomalies")
            for detail in anomaly_details[:3]:  # Show first 3
                print(f"    - {detail['type']}: {detail['description'][:50]}... (conf: {detail['confidence']:.2f})")
            
        except Exception as e:
            print(f"  ❌ {agent_name}: Error - {e}")
            agent_results[agent_name] = {
                'anomalies': [],
                'anomaly_details': [],
                'count': 0,
                'error': str(e)
            }
    
    return agent_results

def analyze_agent_combinations(df, orchestrator, agent_results):
    """Analyze how different agent combinations perform"""
    print(f"\n🔀 AGENT COMBINATION ANALYSIS")
    print(f"={'='*60}")
    
    options = {
        'business_focus': 'general_retail',
        'max_anomalies': 20,
        'confidence_threshold': 0.5,
        'include_explanations': True
    }
    
    # Define interesting combinations
    combinations = {
        'Top 3 Performers': ['statistical_agent', 'ai_agent', 'visual_agent'],
        'AI Only': ['ai_agent', 'embedding_agent', 'context_agent'],
        'Statistical Only': ['statistical_agent', 'enhanced_statistical_agent'],
        'All Agents': list(agent_results.keys()),
        'Best Statistical + AI': ['statistical_agent', 'ai_agent'],
    }
    
    combination_results = {}
    
    for combo_name, agents in combinations.items():
        print(f"\n--- Testing: {combo_name} ---")
        print(f"    Agents: {agents}")
        
        try:
            result = orchestrator.analyze_with_agents(df, {**options, 'selected_agents': agents})
            
            # Count total anomalies and unique patterns
            total_anomalies = 0
            anomaly_types = set()
            affected_fields = set()
            
            for agent_name in agents:
                agent_result = result.get(agent_name, {})
                if 'error' not in agent_result:
                    anomalies = agent_result.get('anomalies', agent_result.get('anomalies_detected', []))
                    total_anomalies += len(anomalies)
                    
                    for anomaly in anomalies:
                        if isinstance(anomaly, dict):
                            anomaly_types.add(anomaly.get('anomaly_type', 'Unknown'))
                            affected_fields.add(anomaly.get('affected_field', 'Unknown'))
            
            combination_results[combo_name] = {
                'agents': agents,
                'total_anomalies': total_anomalies,
                'unique_types': len(anomaly_types),
                'affected_fields': len(affected_fields),
                'anomaly_types': list(anomaly_types),
                'result': result
            }
            
            print(f"    📊 Total anomalies: {total_anomalies}")
            print(f"    📊 Unique types: {len(anomaly_types)}")
            print(f"    📊 Affected fields: {len(affected_fields)}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            combination_results[combo_name] = {
                'agents': agents,
                'error': str(e)
            }
    
    return combination_results

def analyze_overlap_patterns(agent_results):
    """Analyze overlap patterns between agents"""
    print(f"\n🔄 OVERLAP ANALYSIS")
    print(f"={'='*60}")
    
    # Create overlap matrix
    agent_names = list(agent_results.keys())
    overlap_matrix = np.zeros((len(agent_names), len(agent_names)))
    
    for i, agent1 in enumerate(agent_names):
        for j, agent2 in enumerate(agent_names):
            if i == j:
                overlap_matrix[i][j] = agent_results[agent1]['count']
            else:
                # Simple overlap based on anomaly types (simplified)
                types1 = set()
                types2 = set()
                
                for anomaly in agent_results[agent1]['anomaly_details']:
                    types1.add(anomaly['type'])
                
                for anomaly in agent_results[agent2]['anomaly_details']:
                    types2.add(anomaly['type'])
                
                overlap = len(types1.intersection(types2))
                overlap_matrix[i][j] = overlap
    
    # Print overlap analysis
    print("Agent Overlap Matrix (by anomaly types):")
    print("(Diagonal = total anomalies, off-diagonal = shared types)")
    
    # Create a simple text-based matrix
    header = "Agent".ljust(20) + "".join(f"{name[:8]:>8}" for name in agent_names)
    print(header)
    print("-" * len(header))
    
    for i, agent in enumerate(agent_names):
        row = agent[:20].ljust(20)
        for j in range(len(agent_names)):
            row += f"{int(overlap_matrix[i][j]):>8}"
        print(row)
    
    return overlap_matrix

def investigate_why_all_agents_might_reduce_effectiveness():
    """Investigate the core question: why might all agents together find fewer anomalies?"""
    print(f"\n🧠 INVESTIGATION: Why All Agents Together Might Find Fewer Anomalies")
    print(f"={'='*80}")
    
    reasons = [
        "1. 🎯 DEDUPLICATION: Multiple agents detecting the same anomalies",
        "2. 🔄 INTERFERENCE: Agents contradicting each other's findings",  
        "3. 📊 AGGREGATION: Conservative combination of results",
        "4. 🕒 TIMEOUT: Slower processing with more agents",
        "5. 💾 MEMORY: Resource constraints with multiple agents",
        "6. 🎛️ THRESHOLD: Different confidence thresholds being applied",
        "7. 🏗️ ARCHITECTURE: Orchestrator limiting total results"
    ]
    
    for reason in reasons:
        print(f"   {reason}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    recommendations = [
        "1. Use smart agent selection based on data type",
        "2. Implement weighted voting for anomaly confidence",
        "3. Allow agents to specialize in different anomaly types",
        "4. Use ensemble methods that combine strengths",
        "5. Monitor and tune agent interaction effects",
        "6. Consider sequential rather than parallel agent execution",
        "7. Implement sophisticated deduplication logic"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")

def main():
    """Main analysis function"""
    print("🎯 AGENT EFFECTIVENESS & OVERLAP ANALYSIS")
    print("=" * 80)
    
    # Create test dataset
    df, known_anomaly_indices = create_test_dataset_with_clear_anomalies()
    print(f"Created dataset: {df.shape} ({len(known_anomaly_indices)} known anomalies)")
    print(f"Known anomaly rows: {known_anomaly_indices}")
    
    # Show sample of anomalies
    print(f"\nSample anomalies:")
    for idx in known_anomaly_indices[:3]:
        row = df.iloc[idx]
        print(f"  Row {idx}: {row['product_id']} - Price: ${row['price']:.2f}, Sales: {row['sales']}, Rating: {row['rating']:.1f}")
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    # Detailed agent analysis
    agent_results = analyze_agent_detailed_results(df, orchestrator)
    
    # Analyze combinations
    combination_results = analyze_agent_combinations(df, orchestrator, agent_results)
    
    # Overlap analysis
    overlap_matrix = analyze_overlap_patterns(agent_results)
    
    # Investigation of the core question
    investigate_why_all_agents_might_reduce_effectiveness()
    
    # Summary
    print(f"\n📈 FINAL SUMMARY")
    print(f"={'='*60}")
    
    best_individual = max(agent_results.items(), key=lambda x: x[1]['count'])
    print(f"🏆 Best individual agent: {best_individual[0]} ({best_individual[1]['count']} anomalies)")
    
    if combination_results:
        best_combo = max(combination_results.items(), 
                        key=lambda x: x[1].get('total_anomalies', 0) if 'error' not in x[1] else 0)
        print(f"🏆 Best combination: {best_combo[0]} ({best_combo[1].get('total_anomalies', 0)} total anomalies)")
    
    print(f"\n✅ Analysis complete!")
    print(f"   - Focus on combinations that maximize unique detection")
    print(f"   - Consider agent specialization for different data types")
    print(f"   - Monitor for diminishing returns with more agents")

if __name__ == "__main__":
    main()
