"""
Base agent classes and interfaces for the anomaly detection system
Provides common functionality and standardized interfaces for all agents
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Standardized result format for all agents"""
    agent_name: str
    anomalies: List[Dict[str, Any]]
    summary: str
    confidence_score: float = 0.0
    analysis_method: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AnomalyRecord:
    """Standardized anomaly record format"""
    index: int
    description: str
    confidence: float
    severity: str = "medium"  # low, medium, high, critical
    category: str = "statistical"
    column: Optional[str] = None
    value: Optional[Any] = None
    z_score: Optional[float] = None
    business_impact: Optional[str] = None
    recommended_actions: Optional[List[str]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'index': self.index,
            'description': self.description,
            'confidence': self.confidence,
            'severity': self.severity,
            'category': self.category,
            'column': self.column,
            'value': self.value,
            'z_score': self.z_score,
            'business_impact': self.business_impact,
            'recommended_actions': self.recommended_actions or [],
            'metadata': self.metadata or {}
        }

class BaseAgent(ABC):
    """Base class for all anomaly detection agents"""
    
    def __init__(self, name: str, kernel: Optional[Kernel] = None):
        self.name = name
        self.kernel = kernel
        self.logger = logging.getLogger(f"agent.{name}")
        self._initialized = False
        self._capabilities = set()
        
    @abstractmethod
    async def analyze_async(self, data: pd.DataFrame, options: Dict[str, Any] = None) -> AnalysisResult:
        """Asynchronous analysis method - must be implemented by all agents"""
        pass
    
    def analyze(self, data: pd.DataFrame, options: Dict[str, Any] = None) -> AnalysisResult:
        """Synchronous wrapper for analysis"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.analyze_async(data, options))
                    return future.result()
            else:
                return asyncio.run(self.analyze_async(data, options))
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                agent_name=self.name,
                anomalies=[],
                summary=f"Analysis failed: {str(e)}",
                error=str(e)
            )
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        pass
    
    def is_capable_of(self, capability: str) -> bool:
        """Check if agent has specific capability"""
        return capability in self.get_capabilities()
    
    def initialize(self) -> bool:
        """Initialize agent - override if needed"""
        try:
            self._initialized = True
            self.logger.info(f"Agent {self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.name}: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if agent is initialized"""
        return self._initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "capabilities": self.get_capabilities(),
            "kernel_available": self.kernel is not None
        }

class StatisticalAgent(BaseAgent):
    """Statistical anomaly detection agent"""
    
    def __init__(self, kernel: Optional[Kernel] = None):
        super().__init__("statistical", kernel)
    
    async def analyze_async(self, data: pd.DataFrame, options: Dict[str, Any] = None) -> AnalysisResult:
        """Perform statistical anomaly detection"""
        try:
            import numpy as np
            from scipy import stats
            
            options = options or {}
            threshold = options.get('threshold', 2.5)
            max_anomalies = options.get('max_anomalies', 10)
            
            anomalies = []
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if data[column].std() > 0:
                    z_scores = np.abs(stats.zscore(data[column]))
                    outlier_indices = np.where(z_scores > threshold)[0]
                    
                    for idx in outlier_indices[:max_anomalies]:
                        anomaly = AnomalyRecord(
                            index=int(idx),
                            description=f"Statistical outlier in {column}",
                            confidence=min(float(z_scores[idx]) / 5.0, 1.0),
                            severity="high" if z_scores[idx] > 4 else "medium",
                            category="statistical",
                            column=column,
                            value=float(data.iloc[idx][column]),
                            z_score=float(z_scores[idx])
                        )
                        anomalies.append(anomaly.to_dict())
            
            # Sort by confidence and limit results
            anomalies = sorted(anomalies, key=lambda x: x['confidence'], reverse=True)[:max_anomalies]
            
            return AnalysisResult(
                agent_name=self.name,
                anomalies=anomalies,
                summary=f"Found {len(anomalies)} statistical anomalies using z-score analysis",
                confidence_score=0.8,
                analysis_method="Z-score statistical analysis"
            )
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            return AnalysisResult(
                agent_name=self.name,
                anomalies=[],
                summary=f"Statistical analysis failed: {str(e)}",
                error=str(e)
            )
    
    def get_capabilities(self) -> List[str]:
        return ["statistical_analysis", "outlier_detection", "z_score_analysis"]

class SemanticKernelAgent(BaseAgent):
    """Base class for agents that use Semantic Kernel functionality"""
    
    def __init__(self, name: str, kernel: Optional[Kernel] = None):
        super().__init__(name, kernel)
        self._kernel_functions = {}
    
    def initialize(self) -> bool:
        """Initialize Semantic Kernel agent"""
        if not self.kernel:
            self.logger.warning(f"No Semantic Kernel available for {self.name}")
            return False
        
        try:
            # Register kernel functions
            self._register_kernel_functions()
            self._initialized = True
            self.logger.info(f"Semantic Kernel agent {self.name} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize SK agent {self.name}: {e}")
            return False
    
    @abstractmethod
    def _register_kernel_functions(self):
        """Register kernel functions - must be implemented by SK agents"""
        pass
    
    async def invoke_kernel_function(self, function_name: str, **kwargs) -> Any:
        """Invoke a registered kernel function"""
        if not self.kernel:
            raise RuntimeError("No Semantic Kernel available")
        
        if function_name not in self._kernel_functions:
            raise ValueError(f"Function {function_name} not registered")
        
        try:
            function = self._kernel_functions[function_name]
            result = await self.kernel.invoke(function, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Kernel function {function_name} failed: {e}")
            raise

class AgentOrchestrator:
    """Orchestrates multiple agents for comprehensive analysis"""
    
    def __init__(self, kernel: Optional[Kernel] = None):
        self.kernel = kernel
        self._agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("orchestrator")
    
    def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent with the orchestrator"""
        try:
            # Initialize agent if not already done
            if not agent.is_initialized:
                if not agent.initialize():
                    self.logger.warning(f"Failed to initialize agent {agent.name}")
                    return False
            
            self._agents[agent.name] = agent
            self.logger.info(f"Registered agent: {agent.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.name}: {e}")
            return False
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return list(self._agents.keys())
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all registered agents"""
        return {name: agent.get_capabilities() for name, agent in self._agents.items()}
    
    async def analyze_with_agents(self, data: pd.DataFrame, agent_names: List[str] = None, 
                                options: Dict[str, Any] = None) -> Dict[str, AnalysisResult]:
        """Run analysis with specified agents"""
        if agent_names is None:
            agent_names = list(self._agents.keys())
        
        results = {}
        options = options or {}
        
        for agent_name in agent_names:
            if agent_name in self._agents:
                try:
                    self.logger.info(f"Running analysis with agent: {agent_name}")
                    result = await self._agents[agent_name].analyze_async(data, options)
                    results[agent_name] = result
                except Exception as e:
                    self.logger.error(f"Agent {agent_name} failed: {e}")
                    results[agent_name] = AnalysisResult(
                        agent_name=agent_name,
                        anomalies=[],
                        summary=f"Agent {agent_name} failed: {str(e)}",
                        error=str(e)
                    )
            else:
                self.logger.warning(f"Agent {agent_name} not found")
                results[agent_name] = AnalysisResult(
                    agent_name=agent_name,
                    anomalies=[],
                    summary=f"Agent {agent_name} not available",
                    error="Agent not found"
                )
        
        return results
    
    def analyze_sync(self, data: pd.DataFrame, agent_names: List[str] = None,
                    options: Dict[str, Any] = None) -> Dict[str, AnalysisResult]:
        """Synchronous wrapper for agent analysis"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in event loop, use thread executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.analyze_with_agents(data, agent_names, options)
                    )
                    return future.result()
            else:
                return asyncio.run(self.analyze_with_agents(data, agent_names, options))
        except Exception as e:
            self.logger.error(f"Orchestrator analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get status of orchestrator and all agents"""
        return {
            "orchestrator_ready": True,
            "kernel_available": self.kernel is not None,
            "agents": {name: agent.get_status() for name, agent in self._agents.items()},
            "total_agents": len(self._agents)
        }
