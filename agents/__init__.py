"""
Agents package initialization
Provides centralized access to all anomaly detection agents
"""

import logging
from typing import Dict, List, Optional, Any
from semantic_kernel import Kernel

# Import base classes
from .base import BaseAgent, StatisticalAgent, SemanticKernelAgent, AgentOrchestrator, AnalysisResult

# Import analysis agents
from .analysis import ContextAgent, EmbeddingAgent

# Import core configuration
from core import get_kernel, get_service_status

logger = logging.getLogger(__name__)

class AgentManager:
    """Manages all available agents and their lifecycle"""
    
    def __init__(self):
        self.kernel = None
        self.orchestrator = None
        self._available_agents = {}
        self._initialized = False
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the agent system"""
        try:
            # Get Semantic Kernel
            self.kernel = get_kernel()
            service_status = get_service_status()
            
            # Create orchestrator
            self.orchestrator = AgentOrchestrator(self.kernel)
            
            # Initialize available agents
            self._initialize_agents()
            
            self._initialized = True
            
            return {
                "status": "success",
                "agents_loaded": len(self._available_agents),
                "agents": list(self._available_agents.keys()),
                "semantic_kernel_available": self.kernel is not None,
                "services": service_status
            }
            
        except Exception as e:
            logger.error(f"Agent manager initialization failed: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "agents_loaded": 0
            }
    
    def _initialize_agents(self):
        """Initialize all available agents"""
        agents_to_create = [
            ("statistical", lambda: StatisticalAgent(self.kernel)),
            ("context", lambda: ContextAgent(self.kernel)),
            ("embedding", lambda: EmbeddingAgent(self.kernel)),
        ]
        
        for agent_name, agent_factory in agents_to_create:
            try:
                agent = agent_factory()
                if self.orchestrator.register_agent(agent):
                    self._available_agents[agent_name] = agent
                    logger.info(f"Registered agent: {agent_name}")
                else:
                    logger.warning(f"Failed to register agent: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to create agent {agent_name}: {e}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self._available_agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all available agent names"""
        return list(self._available_agents.keys())
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents"""
        if not self.orchestrator:
            return {}
        return self.orchestrator.get_agent_capabilities()
    
    def analyze_data(self, data, agent_names: List[str] = None, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze data using specified agents"""
        if not self._initialized:
            init_result = self.initialize()
            if init_result["status"] == "error":
                return init_result
        
        if not self.orchestrator:
            return {"error": "Orchestrator not available"}
        
        try:
            # Use all agents if none specified
            if agent_names is None:
                agent_names = self.list_agents()
            
            # Filter to only available agents
            available_agents = [name for name in agent_names if name in self._available_agents]
            
            if not available_agents:
                return {"error": "No available agents found"}
            
            # Run analysis
            results = self.orchestrator.analyze_sync(data, available_agents, options)
            
            # Add metadata
            results["agent_manager_status"] = {
                "total_agents": len(self._available_agents),
                "requested_agents": agent_names,
                "available_agents": available_agents,
                "semantic_kernel_available": self.kernel is not None
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of agent system"""
        status = {
            "initialized": self._initialized,
            "kernel_available": self.kernel is not None,
            "orchestrator_ready": self.orchestrator is not None,
            "agents": {}
        }
        
        if self.orchestrator:
            status.update(self.orchestrator.get_orchestrator_status())
        
        # Get service status
        try:
            service_status = get_service_status()
            if service_status.get("semantic_kernel", False):
                status["services"] = service_status
        except:
            status["services"] = {"error": "Could not get service status"}
        
        return status

# Global agent manager instance
_global_agent_manager = None

def get_agent_manager() -> AgentManager:
    """Get global agent manager instance"""
    global _global_agent_manager
    if _global_agent_manager is None:
        _global_agent_manager = AgentManager()
    return _global_agent_manager

def initialize_agents() -> Dict[str, Any]:
    """Initialize the global agent system"""
    manager = get_agent_manager()
    return manager.initialize()

def analyze_with_agents(data, agent_names: List[str] = None, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze data using the global agent system"""
    manager = get_agent_manager()
    return manager.analyze_data(data, agent_names, options)

def get_available_agents() -> List[str]:
    """Get list of available agent names"""
    manager = get_agent_manager()
    return manager.list_agents()

def get_agent_system_status() -> Dict[str, Any]:
    """Get system status for all agents"""
    manager = get_agent_manager()
    if not manager._initialized:
        return {"status": "not_initialized", "agents": []}
    
    return {
        "status": "initialized" if manager._initialized else "not_initialized",
        "agents": manager.list_agents(),
        "capabilities": manager.get_agent_capabilities(),
        "kernel_available": manager.kernel is not None
    }

__all__ = [
    'BaseAgent',
    'StatisticalAgent', 
    'SemanticKernelAgent',
    'AgentOrchestrator',
    'AnalysisResult',
    'ContextAgent',
    'EmbeddingAgent',
    'AgentManager',
    'get_agent_manager',
    'initialize_agents',
    'analyze_with_agents',
    'get_available_agents',
    'get_agent_system_status'
]
