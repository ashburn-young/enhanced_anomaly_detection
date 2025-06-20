"""
Business Logic Module for Retail Anomaly Detection
Handles core business rules, user profiles, and domain knowledge
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime


class UserProfile(Enum):
    BUSINESS_USER = "Business User"
    DATA_ANALYST = "Data Analyst" 
    TECHNICAL_EXPERT = "Technical Expert"


@dataclass
class BusinessContext:
    """Business context for anomaly analysis"""
    industry: str = "retail"
    focus_areas: List[str] = None
    custom_rules: str = ""
    sensitivity_level: float = 0.5
    max_anomalies: int = 10
    
    def __post_init__(self):
        if self.focus_areas is None:
            self.focus_areas = ["sales", "pricing", "inventory", "customer_behavior"]


@dataclass
class AnomalyInsight:
    """Structured anomaly insight with business context"""
    id: str
    description: str
    confidence: float
    business_impact: str
    recommended_actions: List[str]
    affected_metrics: Dict[str, Any]
    root_cause_analysis: str = ""
    priority: str = "medium"  # low, medium, high, critical
    category: str = "statistical"  # statistical, business, pattern, outlier


class BusinessRulesEngine:
    """Engine for applying business rules and context to anomaly detection"""
    
    def __init__(self):
        self.retail_rules = {
            "sales_anomalies": {
                "high_sales_threshold": 3.0,  # Standard deviations
                "low_sales_threshold": -2.5,
                "seasonal_adjustment": True,
                "business_context": "Unusual sales patterns may indicate promotions, stockouts, or data quality issues"
            },
            "pricing_anomalies": {
                "price_variance_threshold": 2.0,
                "discount_threshold": 0.5,  # 50% discount triggers review
                "business_context": "Pricing anomalies may indicate competitive responses or pricing errors"
            },
            "inventory_anomalies": {
                "stock_level_threshold": 2.5,
                "turnover_threshold": 3.0,
                "business_context": "Inventory anomalies may indicate supply chain issues or demand changes"
            }
        }
    
    def apply_business_context(self, anomaly: Dict, business_context: BusinessContext) -> AnomalyInsight:
        """Apply business context to raw anomaly data"""
        
        # Determine business impact
        impact = self._assess_business_impact(anomaly, business_context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(anomaly, business_context)
        
        # Categorize anomaly
        category = self._categorize_anomaly(anomaly)
        
        # Generate root cause analysis
        root_cause = self._analyze_root_cause(anomaly, business_context)
        
        return AnomalyInsight(
            id=anomaly.get('id', f"anomaly_{datetime.now().timestamp()}"),
            description=anomaly.get('description', 'Anomaly detected'),
            confidence=anomaly.get('confidence', 0.5),
            business_impact=impact,
            recommended_actions=recommendations,
            affected_metrics=anomaly.get('affected_metrics', {}),
            root_cause_analysis=root_cause,
            priority=self._determine_priority(anomaly, impact),
            category=category
        )
    
    def _assess_business_impact(self, anomaly: Dict, context: BusinessContext) -> str:
        """Assess the business impact of an anomaly"""
        confidence = anomaly.get('confidence', 0.5)
        value = anomaly.get('value', 0)
        
        if confidence > 0.8:
            if abs(value) > 1000:  # High value anomaly
                return "HIGH: Significant financial impact potential. Immediate investigation recommended."
            elif abs(value) > 100:
                return "MEDIUM: Moderate impact on operations. Review within 24 hours."
            else:
                return "LOW: Minor operational impact. Monitor for patterns."
        elif confidence > 0.6:
            return "MEDIUM: Moderate confidence anomaly. Consider investigating if pattern emerges."
        else:
            return "LOW: Low confidence detection. Monitor for additional signals."
    
    def _generate_recommendations(self, anomaly: Dict, context: BusinessContext) -> List[str]:
        """Generate business-specific recommendations"""
        recommendations = []
        
        anomaly_type = anomaly.get('column', '').lower()
        
        if 'sales' in anomaly_type or 'revenue' in anomaly_type:
            recommendations.extend([
                "Verify data quality and collection processes",
                "Check for promotional activities or marketing campaigns",
                "Review inventory levels for potential stockouts",
                "Analyze customer feedback and satisfaction metrics"
            ])
        elif 'price' in anomaly_type or 'cost' in anomaly_type:
            recommendations.extend([
                "Review pricing strategy and competitor analysis",
                "Validate pricing rules and discount applications",
                "Check for system errors in price updates",
                "Assess impact on profit margins"
            ])
        elif 'inventory' in anomaly_type or 'stock' in anomaly_type:
            recommendations.extend([
                "Review supply chain and delivery schedules",
                "Check demand forecasting accuracy",
                "Validate inventory counting procedures",
                "Assess seasonal demand patterns"
            ])
        else:
            recommendations.extend([
                "Investigate data source integrity",
                "Review business process changes",
                "Check for system or technical issues",
                "Monitor for trend continuation"
            ])
        
        return recommendations
    
    def _categorize_anomaly(self, anomaly: Dict) -> str:
        """Categorize the type of anomaly"""
        if 'z_score' in anomaly:
            return "statistical"
        elif 'pattern' in str(anomaly.get('description', '')).lower():
            return "pattern"
        elif 'business' in str(anomaly.get('description', '')).lower():
            return "business"
        else:
            return "outlier"
    
    def _analyze_root_cause(self, anomaly: Dict, context: BusinessContext) -> str:
        """Provide root cause analysis"""
        confidence = anomaly.get('confidence', 0.5)
        column = anomaly.get('column', '')
        
        if confidence > 0.8:
            return f"Strong statistical evidence suggests {column} value significantly deviates from normal patterns. Likely systemic cause."
        elif confidence > 0.6:
            return f"Moderate evidence of anomaly in {column}. Could be due to natural variation or emerging trend."
        else:
            return f"Weak evidence of anomaly in {column}. May be normal variation or data quality issue."
    
    def _determine_priority(self, anomaly: Dict, impact: str) -> str:
        """Determine anomaly priority"""
        if "HIGH" in impact:
            return "critical"
        elif "MEDIUM" in impact:
            return "high"
        else:
            return "medium"


class UserExperienceManager:
    """Manages user experience based on profile and preferences"""
    
    def __init__(self):
        self.profile_configs = {
            UserProfile.BUSINESS_USER: {
                "show_technical_details": False,
                "explanation_level": "business",
                "chart_complexity": "simple",
                "default_views": ["summary", "recommendations"],
                "terminology": "business"
            },
            UserProfile.DATA_ANALYST: {
                "show_technical_details": True,
                "explanation_level": "analytical",
                "chart_complexity": "detailed",
                "default_views": ["summary", "statistical", "visualizations"],
                "terminology": "analytical"
            },
            UserProfile.TECHNICAL_EXPERT: {
                "show_technical_details": True,
                "explanation_level": "technical",
                "chart_complexity": "comprehensive",
                "default_views": ["all"],
                "terminology": "technical"
            }
        }
    
    def get_user_config(self, profile: UserProfile) -> Dict:
        """Get configuration for user profile"""
        return self.profile_configs.get(profile, self.profile_configs[UserProfile.BUSINESS_USER])
    
    def customize_content(self, content: str, profile: UserProfile) -> str:
        """Customize content based on user profile"""
        config = self.get_user_config(profile)
        
        if config["terminology"] == "business":
            # Replace technical terms with business terms
            content = content.replace("standard deviation", "typical range")
            content = content.replace("z-score", "anomaly score")
            content = content.replace("outlier", "unusual value")
        
        return content
    
    def get_explanation_depth(self, profile: UserProfile) -> str:
        """Get appropriate explanation depth for user"""
        config = self.get_user_config(profile)
        return config["explanation_level"]


class FeedbackManager:
    """Manages user feedback and learning from human-in-the-loop"""
    
    def __init__(self):
        self.feedback_history = []
        self.analytics = {
            'accuracy_by_method': {},
            'common_false_positives': [],
            'high_value_anomalies': [],
            'user_preferences': {}
        }
    
    def record_feedback(self, anomaly_id: str, approved: bool, anomaly: Dict, 
                       method: str, user_comment: str = "", business_impact: str = "") -> Dict:
        """Record user feedback on anomaly"""
        feedback = {
            'id': anomaly_id,
            'timestamp': datetime.now().isoformat(),
            'approved': approved,
            'method': method,
            'anomaly': anomaly,
            'user_comment': user_comment,
            'business_impact': business_impact,
            'confidence': anomaly.get('confidence', 0.0)
        }
        
        self.feedback_history.append(feedback)
        self._update_analytics(feedback)
        
        return feedback
    
    def _update_analytics(self, feedback: Dict):
        """Update feedback analytics"""
        method = feedback['method']
        approved = feedback['approved']
        
        # Update accuracy by method
        if method not in self.analytics['accuracy_by_method']:
            self.analytics['accuracy_by_method'][method] = {'correct': 0, 'total': 0}
        
        self.analytics['accuracy_by_method'][method]['total'] += 1
        if approved:
            self.analytics['accuracy_by_method'][method]['correct'] += 1
        
        # Track false positives
        if not approved:
            self.analytics['common_false_positives'].append({
                'anomaly': feedback['anomaly'],
                'method': method,
                'comment': feedback.get('user_comment', '')
            })
        
        # Track high-value anomalies
        if approved and feedback.get('business_impact') == 'HIGH':
            self.analytics['high_value_anomalies'].append(feedback)
    
    def get_method_accuracy(self, method: str) -> float:
        """Get accuracy for a specific method"""
        if method in self.analytics['accuracy_by_method']:
            stats = self.analytics['accuracy_by_method'][method]
            if stats['total'] > 0:
                return stats['correct'] / stats['total']
        return 0.0
    
    def get_feedback_summary(self) -> Dict:
        """Get comprehensive feedback summary"""
        total_feedback = len(self.feedback_history)
        approved_count = sum(1 for f in self.feedback_history if f['approved'])
        
        method_accuracies = {}
        for method in self.analytics['accuracy_by_method']:
            method_accuracies[method] = self.get_method_accuracy(method)
        
        return {
            'total_feedback': total_feedback,
            'approval_rate': approved_count / max(total_feedback, 1),
            'method_accuracies': method_accuracies,
            'high_value_count': len(self.analytics['high_value_anomalies']),
            'false_positive_rate': len(self.analytics['common_false_positives']) / max(total_feedback, 1)
        }


class DataQualityAssessment:
    """Assess data quality and provide recommendations"""
    
    @staticmethod
    def assess_data_quality(df: pd.DataFrame) -> Dict:
        """Comprehensive data quality assessment"""
        assessment = {
            'overall_score': 0.0,
            'issues': [],
            'recommendations': [],
            'metrics': {}
        }
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100)
        assessment['metrics']['missing_data'] = missing_pct.to_dict()
        
        if missing_pct.max() > 10:
            assessment['issues'].append(f"High missing data: {missing_pct.idxmax()} ({missing_pct.max():.1f}%)")
            assessment['recommendations'].append("Consider data imputation or collection process review")
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        assessment['metrics']['duplicates'] = duplicate_count
        
        if duplicate_count > 0:
            assessment['issues'].append(f"Found {duplicate_count} duplicate records")
            assessment['recommendations'].append("Remove or investigate duplicate records")
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assessment['metrics']['numeric_columns'] = len(numeric_cols)
        assessment['metrics']['total_columns'] = len(df.columns)
        
        if len(numeric_cols) < len(df.columns) * 0.3:
            assessment['recommendations'].append("Consider converting text fields to numeric where appropriate")
        
        # Calculate overall score
        score = 100
        score -= min(missing_pct.max(), 50)  # Deduct for missing data
        score -= min(duplicate_count / len(df) * 100, 20)  # Deduct for duplicates
        
        assessment['overall_score'] = max(score, 0)
        
        return assessment
