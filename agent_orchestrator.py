"""
Agent Orchestrator for Clean Architecture with Semantic Kernel Integration
Simple interface for coordinating multiple agents including Semantic Kernel-based agents
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Clean orchestrator that coordinates agents with Semantic Kernel integration
    """
    
    def __init__(self):
        self.agents = {}
        self.semantic_kernel_available = False
        self._load_agents()
    
    def _load_agents(self):
        """Load both Semantic Kernel and fallback agents with proper integration"""
        self.semantic_kernel_agents = {}
        self.fallback_agents = {}
        
        # Try to load Semantic Kernel agents first
        try:
            from agents import AgentManager
            from core import get_kernel
            
            kernel = get_kernel()
            if kernel:
                self.semantic_kernel_available = True
                agent_manager = AgentManager()
                init_result = agent_manager.initialize()
                
                if init_result.get("status") == "success":
                    # Get actual SK agents
                    available_agents = agent_manager.list_agents()
                    for agent_name in available_agents:
                        agent = agent_manager.get_agent(agent_name)
                        if agent:
                            self.semantic_kernel_agents[agent_name] = agent
                    
                    logger.info(f"Loaded {len(self.semantic_kernel_agents)} Semantic Kernel agents")
                else:
                    logger.warning(f"Semantic Kernel agents failed to initialize: {init_result}")
            else:
                logger.info("No kernel available - SK agents disabled")
                
        except Exception as e:
            logger.warning(f"Semantic Kernel agents not available: {e}")
        
        # Load fallback agents (always load these as backup)
        try:
            from simple_agents import (
                StatisticalAgent, EnhancedStatisticalAgent
            )
            from enhanced_ai_agent import EnhancedAIAgent
            
            # Create fallback agents with different names to avoid conflicts
            self.fallback_agents = {
                'statistical_agent': StatisticalAgent(),
                'enhanced_statistical_agent': EnhancedStatisticalAgent(), 
                'ai_agent': EnhancedAIAgent()
            }
            
            # Only load simple versions if SK versions not available
            if 'embedding_agent' not in self.semantic_kernel_agents:
                from simple_agents import EmbeddingAgent as SimpleEmbeddingAgent
                self.fallback_agents['embedding_agent'] = SimpleEmbeddingAgent()
            
            if 'context_agent' not in self.semantic_kernel_agents:
                from simple_agents import ContextAgent as SimpleContextAgent  
                self.fallback_agents['context_agent'] = SimpleContextAgent()
                
            # Load other simple agents
            try:
                from simple_agents import MemoryBankAgent, VisualAgent
                self.fallback_agents.update({
                    'memory_bank_agent': MemoryBankAgent(),
                    'visual_agent': VisualAgent()
                })
            except ImportError:
                pass  # These are optional
                
            logger.info(f"Loaded {len(self.fallback_agents)} fallback agents")
            
        except Exception as e:
            logger.error(f"Error loading fallback agents: {e}")
            self.fallback_agents = {}
        
        # Combine agents - SK agents take priority
        self.agents = {}
        self.agents.update(self.fallback_agents)
        self.agents.update(self.semantic_kernel_agents)
        
        logger.info(f"Total agents available: {len(self.agents)} "
                   f"(SK: {len(self.semantic_kernel_agents)}, Fallback: {len(self.fallback_agents)})")
    
    def list_agents(self) -> List[str]:
        """List all available agent names"""
        return list(self.agents.keys())
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about available agents"""
        return {
            'total_agents': len(self.agents),
            'semantic_kernel_agents': list(self.semantic_kernel_agents.keys()),
            'fallback_agents': list(self.fallback_agents.keys()),
            'semantic_kernel_available': self.semantic_kernel_available
        }
    
    def analyze_with_agents(self, df: pd.DataFrame, options: dict) -> dict:
        """
        Main analysis method - coordinates selected agents
        
        Args:
            df: Data to analyze
            options: Analysis configuration including:
                - selected_agents: List of agent names to run
                - sensitivity: Detection sensitivity
                - max_anomalies: Maximum anomalies to find
                - custom_prompt: Custom analysis prompt
        
        Returns:
            Dictionary with results from each agent plus orchestrator summary
        """
        results = {}
        selected_agents = options.get('selected_agents', [])
        custom_prompt = options.get('custom_prompt')
        
        logger.info(f"🚀 Agent Orchestrator starting analysis:")
        logger.info(f"   Agents to run: {selected_agents}")
        logger.info(f"   Custom prompt provided: {bool(custom_prompt)}")
        if custom_prompt:
            logger.info(f"   Custom prompt content: {custom_prompt[:100]}{'...' if len(custom_prompt) > 100 else ''}")
        
        logger.info(f"Starting analysis with agents: {selected_agents}")
        
        # Run each selected agent
        for agent_name in selected_agents:
            if agent_name in self.agents:
                try:
                    logger.info(f"Running {agent_name}")
                    
                    if agent_name == 'statistical_agent':
                        result = self._run_statistical_agent(df, options)
                    elif agent_name == 'enhanced_statistical_agent':
                        result = self._run_enhanced_statistical_agent(df, options)
                    elif agent_name == 'ai_agent':
                        result = self._run_ai_agent(df, options, custom_prompt)
                    elif agent_name == 'embedding_agent':
                        result = self._run_embedding_agent(df, options)
                    elif agent_name == 'context_agent':
                        result = self._run_context_agent(df, options)
                    elif agent_name == 'memory_bank_agent':
                        result = self._run_memory_bank_agent(df, options)
                    elif agent_name == 'visual_agent':
                        result = self._run_visual_agent(df, options)
                    else:
                        result = {'error': f'Unknown agent: {agent_name}'}
                    
                    results[agent_name] = result
                    logger.info(f"Completed {agent_name}")
                    
                except Exception as e:
                    logger.error(f"Error running {agent_name}: {e}")
                    results[agent_name] = {
                        'error': str(e),
                        'status': 'failed'
                    }
            else:
                results[agent_name] = {
                    'error': 'Agent not available',
                    'status': 'unavailable'
                }
        
        # Generate orchestrator summary
        results['orchestrator_summary'] = self._generate_summary(results, df, options)
        
        return results
    
    def _run_statistical_agent(self, df: pd.DataFrame, options: dict) -> dict:
        """Run statistical agent"""
        try:
            agent = self.agents['statistical_agent']
            result = agent.detect_anomalies(df)
            
            # Limit results based on options
            if 'anomalies' in result and isinstance(result['anomalies'], list):
                max_anomalies = options.get('max_anomalies', 10)
                result['anomalies'] = result['anomalies'][:max_anomalies]
            
            return result
        except Exception as e:
            return self._fallback_statistical(df, options)
    
    def _run_enhanced_statistical_agent(self, df: pd.DataFrame, options: dict) -> dict:
        """Run enhanced statistical agent"""
        try:
            agent = self.agents['enhanced_statistical_agent']
            result = agent.detect_anomalies(df)
            
            if 'anomalies' in result and isinstance(result['anomalies'], list):
                max_anomalies = options.get('max_anomalies', 10)
                result['anomalies'] = result['anomalies'][:max_anomalies]
            
            return result
        except Exception as e:
            return self._fallback_statistical(df, options)
    
    def _run_ai_agent(self, df: pd.DataFrame, options: dict, custom_prompt: str = None) -> dict:
        """Run AI agent with optional custom prompt"""
        logger.info(f"🤖 Running AI agent with custom_prompt: {bool(custom_prompt)}")
        if custom_prompt:
            logger.info(f"🤖 AI agent custom_prompt preview: {custom_prompt[:100]}{'...' if len(custom_prompt) > 100 else ''}")
        
        try:
            agent = self.agents['ai_agent']
            
            # Prepare data sample for AI analysis
            sample_size = min(100, len(df))
            df_sample = df.sample(n=sample_size) if len(df) > sample_size else df
            
            # Use the correct method name
            result = agent.analyze_data(df_sample, custom_prompt)
            
            return result
        except Exception as e:
            logger.error(f"AI agent error: {e}")
            return {
                'analysis': f"AI analysis encountered an error: {str(e)}",
                'anomalies': [],
                'status': 'error'
            }
    
    def _run_embedding_agent(self, df: pd.DataFrame, options: dict) -> dict:
        """Run embedding agent"""
        # Use asyncio to run the universal agent runner
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._run_agent_universal('embedding_agent', df, options))
            loop.close()
            return result
        except Exception as e:
            return {
                'analysis': 'Embedding analysis not available',
                'anomalies': [],
                'error': str(e)
            }
    
    def _run_context_agent(self, df: pd.DataFrame, options: dict) -> dict:
        """Run context agent"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._run_agent_universal('context_agent', df, options))
            loop.close()
            return result
        except Exception as e:
            return {
                'analysis': 'Context analysis not available',
                'anomalies': [],
                'error': str(e)
            }
    
    def _run_memory_bank_agent(self, df: pd.DataFrame, options: dict) -> dict:
        """Run memory bank agent"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._run_agent_universal('memory_bank_agent', df, options))
            loop.close()
            return result
        except Exception as e:
            return {
                'analysis': 'Memory bank analysis not available',
                'anomalies': [],
                'error': str(e)
            }
    
    def _run_visual_agent(self, df: pd.DataFrame, options: dict) -> dict:
        """Run visual agent"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._run_agent_universal('visual_agent', df, options))
            loop.close()
            return result
        except Exception as e:
            return {
                'analysis': 'Visual analysis not available',
                'anomalies': [],
                'error': str(e)
            }
    
    def _generate_summary(self, results: dict, df: pd.DataFrame, options: dict) -> str:
        """Generate a summary of all agent results"""
        total_anomalies = 0
        successful_agents = 0
        failed_agents = 0
        
        agent_summaries = []
        
        for agent_name, result in results.items():
            if agent_name == 'orchestrator_summary':
                continue
                
            if isinstance(result, dict):
                if 'error' in result:
                    failed_agents += 1
                    agent_summaries.append(f"❌ {agent_name}: Failed ({result.get('error', 'Unknown error')})")
                else:
                    successful_agents += 1
                    anomaly_count = 0
                    if 'anomalies' in result and isinstance(result['anomalies'], list):
                        anomaly_count = len(result['anomalies'])
                    total_anomalies += anomaly_count
                    agent_summaries.append(f"✅ {agent_name}: Found {anomaly_count} anomalies")
        
        summary = f"""
**Agent Analysis Complete**

📊 **Data**: {len(df)} records analyzed  
🤖 **Agents**: {successful_agents} successful, {failed_agents} failed  
🚨 **Total Anomalies**: {total_anomalies} found  

**Agent Results:**
{chr(10).join(agent_summaries)}

💡 **Recommendation**: Review the anomalies found by each agent. Statistical agents find numerical outliers, while AI agents provide contextual insights.
        """.strip()
        
        return summary
    
    def _fallback_statistical(self, df: pd.DataFrame, options: dict) -> dict:
        """Simple fallback statistical analysis"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            anomalies = []
            
            for col in numeric_cols:
                if df[col].std() > 0:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = df[z_scores > 2.5]
                    
                    for idx, row in outliers.iterrows():
                        anomalies.append({
                            'index': int(idx),
                            'column': col,
                            'value': float(row[col]),
                            'z_score': float(z_scores.loc[idx]),
                            'description': f'Statistical outlier in {col}',
                            'confidence': min(float(z_scores.loc[idx]) / 5.0, 1.0)
                        })
            
            # Limit results
            max_anomalies = options.get('max_anomalies', 10)
            anomalies = sorted(anomalies, key=lambda x: x['confidence'], reverse=True)[:max_anomalies]
            
            return {
                'anomalies': anomalies,
                'summary': f'Found {len(anomalies)} statistical anomalies using fallback analysis',
                'status': 'fallback'
            }
        except Exception as e:
            return {
                'anomalies': [],
                'error': f'Fallback analysis failed: {str(e)}',
                'status': 'failed'
            }
    
    async def _run_agent_universal(self, agent_name: str, df: pd.DataFrame, options: dict) -> dict:
        """Universal agent runner that handles both sync and async agents"""
        if agent_name not in self.agents:
            return {
                'analysis': f'{agent_name} not available',
                'anomalies': [],
                'error': f'Agent {agent_name} not found'
            }
        
        agent = self.agents[agent_name]
        
        try:
            # Check if agent has analyze_async method (Semantic Kernel agents)
            if hasattr(agent, 'analyze_async'):
                result = await agent.analyze_async(df, options)
                # Convert AnalysisResult to dict format
                if hasattr(result, '__dict__'):
                    return {
                        'analysis': result.summary,
                        'anomalies': result.anomalies,
                        'confidence_score': getattr(result, 'confidence_score', 0.0),
                        'agent_name': result.agent_name,
                        'analysis_method': getattr(result, 'analysis_method', 'unknown'),
                        'metadata': getattr(result, 'metadata', {}),
                        'error': getattr(result, 'error', None)
                    }
                else:
                    return result
            
            # Check if agent has detect_anomalies method (Simple agents)
            elif hasattr(agent, 'detect_anomalies'):
                result = agent.detect_anomalies(df)
                
                # Limit results based on options
                if 'anomalies' in result and isinstance(result['anomalies'], list):
                    max_anomalies = options.get('max_anomalies', 10)
                    result['anomalies'] = result['anomalies'][:max_anomalies]
                
                return result
            
            # Check if agent has analyze_data method (AI agents)
            elif hasattr(agent, 'analyze_data'):
                sample_size = min(100, len(df))
                df_sample = df.sample(n=sample_size) if len(df) > sample_size else df
                custom_prompt = options.get('custom_prompt')
                return agent.analyze_data(df_sample, custom_prompt)
            
            else:
                return {
                    'analysis': f'{agent_name} has no supported analysis method',
                    'anomalies': [],
                    'error': f'No analyze_async, detect_anomalies, or analyze_data method found'
                }
                
        except Exception as e:
            logger.error(f"Error running {agent_name}: {e}")
            return {
                'analysis': f'{agent_name} analysis failed',
                'anomalies': [],
                'error': str(e)
            }
