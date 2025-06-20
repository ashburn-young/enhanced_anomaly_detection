"""
Context Agent using Semantic Kernel
Provides contextual analysis and business-focused explanations for anomalies
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ChatHistory

from agents.base import BaseAgent, SemanticKernelAgent, AnalysisResult, AnomalyRecord

logger = logging.getLogger(__name__)

class ContextAgent(SemanticKernelAgent):
    """
    AI agent that provides contextual analysis and business explanations
    """
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel, "context_agent")
        self.capabilities = [
            "contextual_analysis",
            "business_explanation", 
            "impact_assessment",
            "actionable_recommendations"
        ]
        
    async def analyze_async(self, data: pd.DataFrame, options: Dict[str, Any] = None) -> AnalysisResult:
        """Async analysis method"""
        try:
            if options is None:
                options = {}
                
            # Get anomalies from statistical analysis
            anomalies = self._get_statistical_anomalies(data, options)
            
            # Generate contextual explanations for each anomaly
            contextualized_anomalies = []
            for anomaly in anomalies:
                context = await self._generate_context_explanation(data, anomaly, options)
                anomaly.update(context)
                contextualized_anomalies.append(anomaly)
            
            # Generate overall summary
            summary = await self._generate_summary(data, contextualized_anomalies, options)
            
            return AnalysisResult(
                agent_name=self.name,
                anomalies=contextualized_anomalies,
                summary=summary,
                confidence_score=0.85,
                analysis_method="contextual_ai",
                metadata={
                    "total_anomalies": len(contextualized_anomalies),
                    "data_shape": data.shape,
                    "columns_analyzed": list(data.columns)
                }
            )
            
        except Exception as e:
            logger.error(f"Context agent analysis failed: {e}")
            return AnalysisResult(
                agent_name=self.name,
                anomalies=[],
                summary="Analysis failed due to an error",
                confidence_score=0.0,
                error=str(e)
            )
    
    def _get_statistical_anomalies(self, data: pd.DataFrame, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get statistical anomalies to provide context for"""
        anomalies = []
        
        # Simple Z-score based anomaly detection
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        threshold = options.get('threshold', 2.5)
        
        for col in numeric_columns:
            if data[col].std() == 0:
                continue
                
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            anomaly_indices = data.index[z_scores > threshold].tolist()
            
            for idx in anomaly_indices:
                anomalies.append({
                    'index': int(idx),
                    'column': col,
                    'value': data.loc[idx, col],
                    'z_score': float(z_scores.loc[idx]),
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'severity': self._calculate_severity(z_scores.loc[idx])
                })
        
        return anomalies
    
    def _calculate_severity(self, z_score: float) -> str:
        """Calculate severity based on Z-score"""
        abs_z = abs(z_score)
        if abs_z > 4:
            return "critical"
        elif abs_z > 3:
            return "high"
        elif abs_z > 2.5:
            return "medium"
        else:
            return "low"
    
    async def _generate_context_explanation(self, data: pd.DataFrame, anomaly: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contextual explanation for an anomaly"""
        try:
            # Prepare context data
            row_data = data.iloc[anomaly['index']].to_dict()
            business_context = options.get('business_context', 'retail sales data')
            user_role = options.get('user_role', 'business analyst')
            
            # Create prompt for context explanation
            prompt = self._create_context_prompt(anomaly, row_data, business_context, user_role)
            
            if self.kernel:
                # Use Semantic Kernel for AI-powered explanation
                explanation = await self._get_ai_explanation(prompt)
            else:
                # Fallback to rule-based explanation
                explanation = self._generate_fallback_explanation(anomaly, row_data, business_context)
            
            return {
                'business_explanation': explanation['business_explanation'],
                'potential_causes': explanation['potential_causes'],
                'business_impact': explanation['business_impact'],
                'recommended_actions': explanation['recommended_actions'],
                'context_data': row_data
            }
            
        except Exception as e:
            logger.error(f"Failed to generate context explanation: {e}")
            return {
                'business_explanation': f"Unusual value detected in {anomaly['column']}",
                'potential_causes': ["Data quality issue", "Operational change"],
                'business_impact': "Requires investigation",
                'recommended_actions': ["Verify data source", "Check for process changes"],
                'context_data': {}
            }
    
    def _create_context_prompt(self, anomaly: Dict[str, Any], row_data: Dict[str, Any], business_context: str, user_role: str) -> str:
        """Create prompt for AI explanation"""
        return f"""
As an expert business analyst, explain this anomaly in {business_context} for a {user_role}:

Anomaly Details:
- Column: {anomaly['column']}
- Value: {anomaly['value']}
- Z-score: {anomaly['z_score']:.2f}
- Severity: {anomaly['severity']}
- Mean: {anomaly['mean']:.2f}
- Standard Deviation: {anomaly['std']:.2f}

Row Context: {row_data}

Provide a business-friendly explanation covering:
1. What this anomaly means in business terms
2. Potential causes (2-3 most likely)
3. Business impact assessment
4. Specific recommended actions (2-3 actionable steps)

Keep explanations clear and actionable for business users.
"""
    
    async def _get_ai_explanation(self, prompt: str) -> Dict[str, Any]:
        """Get AI-powered explanation using Semantic Kernel"""
        try:
            # Create a simple function for explanation
            @kernel_function(
                description="Generate business explanation for anomaly",
                name="explain_anomaly"
            )
            def explain_anomaly(prompt: str) -> str:
                return prompt
            
            # Add function to kernel
            self.kernel.add_function(plugin_name="context", function=explain_anomaly)
            
            # Get response
            result = await self.kernel.invoke(function_name="context-explain_anomaly", prompt=prompt)
            response = str(result)
            
            # Parse response (simplified - in production would use better parsing)
            return self._parse_ai_response(response)
            
        except Exception as e:
            logger.error(f"AI explanation failed: {e}")
            return self._generate_fallback_explanation({}, {}, "business data")
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        # Simplified parsing - in production would use better structured output
        lines = response.split('\n')
        
        explanation = {
            'business_explanation': "Significant deviation from normal patterns detected",
            'potential_causes': ["Data anomaly", "Process change", "External factor"],
            'business_impact': "Requires investigation to determine impact",
            'recommended_actions': ["Investigate data source", "Check recent changes", "Monitor trend"]
        }
        
        # Try to extract more specific information from response
        for line in lines:
            if 'explanation' in line.lower() or 'means' in line.lower():
                explanation['business_explanation'] = line.strip()
                break
        
        return explanation
    
    def _generate_fallback_explanation(self, anomaly: Dict[str, Any], row_data: Dict[str, Any], business_context: str) -> Dict[str, Any]:
        """Generate rule-based explanation when AI is not available"""
        column = anomaly.get('column', 'unknown')
        z_score = anomaly.get('z_score', 0)
        severity = anomaly.get('severity', 'medium')
        
        # Rule-based explanations based on column types and severity
        if 'price' in column.lower() or 'cost' in column.lower():
            causes = ["Pricing error", "Market fluctuation", "Promotional pricing"]
            impact = "May affect revenue or profit margins"
            actions = ["Verify pricing strategy", "Check for data entry errors", "Review market conditions"]
        elif 'quantity' in column.lower() or 'sales' in column.lower():
            causes = ["Demand spike", "Inventory issue", "Data recording error"]
            impact = "Could indicate supply/demand imbalance"
            actions = ["Check inventory levels", "Verify sales records", "Analyze demand patterns"]
        else:
            causes = ["Data quality issue", "Process change", "System anomaly"]
            impact = f"Unusual pattern in {column} requires attention"
            actions = ["Investigate data source", "Check for recent changes", "Monitor for patterns"]
        
        explanation = f"Detected {severity} severity anomaly in {column} with Z-score of {z_score:.2f}"
        
        return {
            'business_explanation': explanation,
            'potential_causes': causes,
            'business_impact': impact,
            'recommended_actions': actions
        }
    
    async def _generate_summary(self, data: pd.DataFrame, anomalies: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """Generate overall summary of the analysis"""
        if not anomalies:
            return "No significant anomalies detected in the dataset."
        
        total_anomalies = len(anomalies)
        severity_counts = {}
        columns_affected = set()
        
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'medium')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            columns_affected.add(anomaly.get('column', 'unknown'))
        
        business_context = options.get('business_context', 'your data')
        
        summary = f"Contextual analysis of {business_context} identified {total_anomalies} anomalies across {len(columns_affected)} columns. "
        
        if severity_counts:
            severity_text = []
            for severity, count in severity_counts.items():
                severity_text.append(f"{count} {severity} severity")
            summary += f"Distribution: {', '.join(severity_text)}. "
        
        summary += "Each anomaly includes business context, potential causes, and recommended actions for your review."
        
        return summary
