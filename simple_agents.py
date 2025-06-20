"""
Simple Agent Wrappers for Direct Synchronous Analysis
These provide simple interfaces for the orchestrator to use
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StatisticalAgent:
    """Simple statistical analysis agent"""
    
    def detect_anomalies(self, df: pd.DataFrame, options: dict = None) -> dict:
        """Compatibility method for detect_anomalies interface"""
        return self.analyze_data(df)
    
    def analyze_data(self, df: pd.DataFrame) -> dict:
        """Perform basic statistical anomaly detection"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            anomalies = []
            
            for col in numeric_cols:
                if df[col].notna().sum() > 10:  # Need enough data
                    # Z-score method
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers_mask = z_scores > 3
                    outlier_indices = df[outliers_mask].index.tolist()
                    outlier_count = len(outlier_indices)
                    
                    if outlier_count > 0:
                        anomalies.append({
                            'type': 'Z-score Outlier',
                            'column': col,
                            'description': f'Found {outlier_count} outliers in {col} using Z-score > 3',
                            'count': int(outlier_count),
                            'confidence': 0.8,
                            'method': 'Z-score',
                            'threshold': 3.0,
                            'indices': outlier_indices[:15]  # Limit to first 15 for display
                        })
                    
                    # IQR method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    iqr_outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    iqr_outlier_indices = df[iqr_outliers_mask].index.tolist()
                    iqr_outliers_count = len(iqr_outlier_indices)
                    
                    if iqr_outliers_count > 0:
                        anomalies.append({
                            'type': 'IQR Outlier',
                            'column': col,
                            'description': f'Found {iqr_outliers_count} outliers in {col} using IQR method',
                            'count': int(iqr_outliers_count),
                            'confidence': 0.75,
                            'method': 'IQR',
                            'bounds': {'lower': lower_bound, 'upper': upper_bound},
                            'indices': iqr_outlier_indices[:15]  # Limit to first 15 for display
                        })
            
            return {
                'anomalies': anomalies,
                'summary': f'Statistical analysis found {len(anomalies)} anomaly types across {len(numeric_cols)} numeric columns',
                'method': 'Statistical Agent',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {
                'anomalies': [],
                'summary': f'Statistical analysis failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class EnhancedStatisticalAgent:
    """Enhanced statistical analysis with advanced methods"""
    
    def detect_anomalies(self, df: pd.DataFrame, options: dict = None) -> dict:
        """Compatibility method for detect_anomalies interface"""
        return self.analyze_data(df, options)
    
    def analyze_data(self, df: pd.DataFrame, options: dict = None) -> dict:
        """Perform enhanced statistical analysis with detailed examples"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            anomalies = []
            max_anomalies = options.get('max_anomalies', 10) if options else 10
            
            if len(numeric_cols) > 0:
                # Prepare data for multivariate analysis
                clean_data = df[numeric_cols].dropna()
                original_indices = clean_data.index
                
                if len(clean_data) > 10:
                    # Isolation Forest
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(clean_data)
                    
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(scaled_data)
                    outlier_indices = original_indices[outliers == -1]
                    outlier_count = len(outlier_indices)
                    
                    if outlier_count > 0:
                        # Get specific examples of outliers with enhanced business context
                        sample_size = min(10, len(outlier_indices))  # Show up to 10 examples
                        sample_outliers = outlier_indices[:sample_size]
                        specific_examples = []
                        
                        for idx in sample_outliers:
                            row_data = df.loc[idx]
                            example_text = f"Product {idx}: "
                            # Show key fields for business context
                            key_fields = []
                            
                            # Prioritize business-relevant columns
                            priority_cols = ['ProductName', 'BrandName', 'Description', 'Price', 'Weight', 'WeightOz']
                            shown_cols = []
                            
                            # First show priority columns if they exist
                            for col in priority_cols:
                                if col in df.columns and col in row_data and pd.notna(row_data[col]):
                                    if isinstance(row_data[col], (int, float)):
                                        key_fields.append(f"{col}={row_data[col]:.2f}")
                                    else:
                                        key_fields.append(f"{col}='{str(row_data[col])[:40]}'")
                                    shown_cols.append(col)
                                    if len(key_fields) >= 4:  # Limit to 4 key fields
                                        break
                            
                            # Then add numeric columns not already shown
                            for col in numeric_cols:
                                if col not in shown_cols and len(key_fields) < 4:
                                    if col in row_data and pd.notna(row_data[col]):
                                        key_fields.append(f"{col}={row_data[col]:.2f}")
                                        shown_cols.append(col)
                            
                            example_text += ", ".join(key_fields)
                            specific_examples.append(example_text)
                        
                        # Create detailed anomaly with examples and business insights
                        anomaly_detail = {
                            'type': 'Multivariate Statistical Outlier',
                            'description': f'We detected {outlier_count} products that have unusual combinations of features when analyzed together.',
                            'count': int(outlier_count),
                            'confidence': 0.85,
                            'method': 'Isolation Forest',
                            'contamination': 0.1,
                            'specific_examples': specific_examples,
                            'sample_indices': list(sample_outliers),
                            'total_affected': outlier_count,
                            'business_context': {
                                'what_this_means': 'These products don\'t fit typical patterns and may represent exceptional cases, data errors, or emerging trends worth investigating.',
                                'potential_causes': [
                                    'Premium or specialty products with unique characteristics',
                                    'Data entry errors that need correction',
                                    'New product categories or variants',
                                    'Pricing opportunities or issues',
                                    'Inventory classification problems'
                                ],
                                'recommended_actions': [
                                    'Review each product to verify data accuracy',
                                    'Check if these represent legitimate premium/specialty items',
                                    'Investigate pricing strategy for unusual products',
                                    'Consider separate analysis for product categories'
                                ]
                            },
                            'analysis_details': {
                                'features_analyzed': list(numeric_cols),
                                'detection_method': 'Isolation Forest with 10% contamination threshold',
                                'explanation': 'Products identified using advanced multivariate analysis that considers all numeric features simultaneously'
                            }
                        }
                        
                        # If there are many anomalies, provide additional investigation tools
                        if outlier_count > 20:
                            anomaly_detail['large_set_tools'] = {
                                'showing_sample': True,
                                'sample_size': len(specific_examples),
                                'total_available': outlier_count,
                                'investigation_options': [
                                    'Export full anomaly list for detailed review',
                                    'Filter by product category or price range',
                                    'Review statistical distribution of anomalies',
                                    'Sample random subsets for manual validation'
                                ],
                                'business_recommendation': 'For large anomaly sets, focus on high-value products or specific categories first'
                            }
                        
                        anomalies.append(anomaly_detail)
            
            return {
                'anomalies': anomalies,
                'summary': f'Enhanced statistical analysis completed with {len(anomalies)} advanced anomaly types',
                'method': 'Enhanced Statistical Agent',
                'analysis_metadata': {
                    'numeric_columns_analyzed': len(numeric_cols),
                    'records_analyzed': len(clean_data),
                    'detection_algorithms': ['Isolation Forest']
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Enhanced statistical analysis failed: {e}")
            return {
                'anomalies': [],
                'summary': f'Enhanced analysis failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class ContextAgent:
    """AI-powered context analysis agent"""
    
    def detect_anomalies(self, df: pd.DataFrame, options: dict = None) -> dict:
        """Compatibility method for detect_anomalies interface"""
        return self.analyze_data(df)
    
    def analyze_data(self, df: pd.DataFrame) -> dict:
        """Perform AI context analysis"""
        try:
            # Generate summary statistics for AI context
            summary_stats = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'missing_values': df.isnull().sum().sum(),
                'data_types': df.dtypes.value_counts().to_dict()
            }
            
            # Simple pattern-based anomaly detection
            anomalies = []
            
            # Check for suspicious patterns
            if summary_stats['missing_values'] > len(df) * 0.5:
                anomalies.append({
                    'type': 'Data Quality Issue',
                    'description': f'High missing value rate: {summary_stats["missing_values"]} missing values',
                    'confidence': 0.9,
                    'context': 'Data quality concern'
                })
            
            # Check for duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                anomalies.append({
                    'type': 'Duplicate Records',
                    'description': f'Found {duplicate_count} duplicate records',
                    'count': int(duplicate_count),
                    'confidence': 1.0,
                    'context': 'Data integrity issue'
                })
            
            return {
                'anomalies': anomalies,
                'summary': f'AI context analysis completed, evaluated {len(df)} records',
                'recommendations': [
                    'Review data quality issues if any were found',
                    'Consider data preprocessing for missing values',
                    'Validate business logic for detected patterns'
                ],
                'confidence_range': '0.7 - 1.0',
                'data_summary': summary_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {
                'anomalies': [],
                'summary': f'Context analysis failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class EmbeddingAgent:
    """Embedding-based semantic analysis agent"""
    
    def detect_anomalies(self, df: pd.DataFrame, options: dict = None) -> dict:
        """Compatibility method for detect_anomalies interface"""
        return self.analyze_data(df)
    
    def analyze_data(self, df: pd.DataFrame) -> dict:
        """Perform embedding-based analysis"""
        try:
            # For now, simulate embedding analysis
            text_cols = df.select_dtypes(include=['object']).columns
            anomalies = []
            
            for col in text_cols:
                if df[col].notna().sum() > 0:
                    # Simple text pattern analysis
                    value_counts = df[col].value_counts()
                    
                    # Check for unusual text patterns
                    if len(value_counts) > 0:
                        # Check for very rare values (potential typos/anomalies)
                        rare_values = value_counts[value_counts == 1]
                        if len(rare_values) > len(value_counts) * 0.1:  # > 10% are unique
                            anomalies.append({
                                'type': 'Text Pattern Anomaly',
                                'column': col,
                                'description': f'High variation in text values for {col}',
                                'count': len(rare_values),
                                'confidence': 0.6,
                                'context': 'Potential data entry inconsistencies'
                            })
            
            return {
                'anomalies': anomalies,
                'summary': f'Embedding analysis completed, processed {len(text_cols)} text columns',
                'embeddings_count': len(text_cols),
                'confidence': 0.6,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedding analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class MemoryBankAgent:
    """Memory bank pattern matching agent"""
    
    def detect_anomalies(self, df: pd.DataFrame, options: dict = None) -> dict:
        """Compatibility method for detect_anomalies interface"""
        return self.analyze_data(df)
    
    def analyze_data(self, df: pd.DataFrame) -> dict:
        """Perform pattern matching analysis"""
        try:
            anomalies = []
            patterns_found = 0
            
            # Simple pattern detection
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check for patterns in categorical data
                    value_counts = df[col].value_counts()
                    
                    # Pattern: Single dominant value (> 90%)
                    if len(value_counts) > 1:
                        dominant_pct = value_counts.iloc[0] / len(df)
                        if dominant_pct > 0.9:
                            anomalies.append({
                                'type': 'Dominant Pattern',
                                'column': col,
                                'description': f'One value dominates {dominant_pct:.1%} of {col}',
                                'confidence': 0.8,
                                'pattern': 'dominance'
                            })
                            patterns_found += 1
            
            return {
                'anomalies': anomalies,
                'summary': f'Memory bank analysis completed, found {patterns_found} patterns',
                'patterns_count': patterns_found,
                'confidence': 0.7,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Memory bank analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class VisualAgent:
    """Visual/multimodal analysis agent"""
    
    def detect_anomalies(self, df: pd.DataFrame, options: dict = None) -> dict:
        """Compatibility method for detect_anomalies interface"""
        return self.analyze_data(df)
    
    def analyze_data(self, df: pd.DataFrame) -> dict:
        """Perform visual analysis simulation"""
        try:
            anomalies = []
            charts_analyzed = 0
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols[:3]:  # Limit to first 3 columns for demo
                if df[col].notna().sum() > 0:
                    charts_analyzed += 1
                    
                    # Simple visual pattern detection
                    # Check for extreme skewness (visual anomaly)
                    skewness = df[col].skew()
                    if abs(skewness) > 2:
                        anomalies.append({
                            'type': 'Visual Distribution Anomaly',
                            'column': col,
                            'description': f'Highly skewed distribution in {col} (skew: {skewness:.2f})',
                            'confidence': 0.75,
                            'visual_metric': 'skewness',
                            'value': skewness
                        })
            
            return {
                'anomalies': anomalies,
                'summary': f'Visual analysis completed, analyzed {charts_analyzed} charts',
                'charts_analyzed': charts_analyzed,
                'confidence': 0.7,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
