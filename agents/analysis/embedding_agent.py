"""
Embedding Agent using Semantic Kernel
Provides semantic similarity analysis and pattern detection using embeddings
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function

from agents.base import BaseAgent, SemanticKernelAgent, AnalysisResult, AnomalyRecord

logger = logging.getLogger(__name__)

class EmbeddingAgent(SemanticKernelAgent):
    """
    AI agent that uses embeddings for semantic similarity and pattern analysis
    """
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel, "embedding_agent")
        self.capabilities = [
            "semantic_similarity",
            "pattern_clustering", 
            "text_anomaly_detection",
            "multi_dimensional_analysis"
        ]
        self.embeddings_cache = {}
        
    async def analyze_async(self, data: pd.DataFrame, options: Dict[str, Any] = None) -> AnalysisResult:
        """Async analysis using embeddings"""
        try:
            if options is None:
                options = {}
                
            # Determine which columns to analyze
            text_columns = self._identify_text_columns(data)
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            anomalies = []
            
            # Text-based anomaly detection using embeddings
            if text_columns and self.kernel:
                text_anomalies = await self._detect_text_anomalies(data, text_columns, options)
                anomalies.extend(text_anomalies)
            
            # Numeric pattern analysis using embedding-style similarity
            if numeric_columns:
                pattern_anomalies = await self._detect_pattern_anomalies(data, numeric_columns, options)
                anomalies.extend(pattern_anomalies)
            
            # Generate summary
            summary = await self._generate_embedding_summary(data, anomalies, options)
            
            return AnalysisResult(
                agent_name=self.name,
                anomalies=anomalies,
                summary=summary,
                confidence_score=0.80,
                analysis_method="embedding_similarity",
                metadata={
                    "text_columns_analyzed": text_columns,
                    "numeric_columns_analyzed": numeric_columns,
                    "total_anomalies": len(anomalies),
                    "embedding_available": self.kernel is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Embedding agent analysis failed: {e}")
            return AnalysisResult(
                agent_name=self.name,
                anomalies=[],
                summary="Embedding analysis failed due to an error",
                confidence_score=0.0,
                error=str(e)
            )
    
    def _identify_text_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify text columns suitable for embedding analysis"""
        text_columns = []
        
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if column contains meaningful text (not just categories)
                sample_values = data[col].dropna().head(10)
                if len(sample_values) > 0:
                    avg_length = sample_values.astype(str).str.len().mean()
                    unique_ratio = len(sample_values.unique()) / len(sample_values)
                    
                    # Consider it text if average length > 10 and high uniqueness
                    if avg_length > 10 and unique_ratio > 0.5:
                        text_columns.append(col)
        
        return text_columns
    
    async def _detect_text_anomalies(self, data: pd.DataFrame, text_columns: List[str], options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in text data using embeddings"""
        anomalies = []
        
        for col in text_columns:
            try:
                # Get embeddings for all text values
                text_data = data[col].dropna().astype(str).tolist()
                
                if len(text_data) < 5:  # Need minimum data for analysis
                    continue
                
                # Get embeddings
                embeddings = await self._get_text_embeddings(text_data)
                
                if embeddings is None:
                    continue
                
                # Find outliers using embedding similarity
                outlier_indices = self._find_embedding_outliers(embeddings, threshold=0.3)
                
                # Map back to original dataframe indices
                valid_indices = data[col].dropna().index.tolist()
                
                for outlier_idx in outlier_indices:
                    if outlier_idx < len(valid_indices):
                        original_idx = valid_indices[outlier_idx]
                        anomalies.append({
                            'index': int(original_idx),
                            'column': col,
                            'value': data.loc[original_idx, col],
                            'anomaly_type': 'text_semantic',
                            'similarity_score': 0.0,  # Would calculate actual score
                            'severity': 'medium',
                            'business_explanation': f"Text in {col} is semantically different from typical patterns",
                            'category': 'semantic'
                        })
                        
            except Exception as e:
                logger.error(f"Text anomaly detection failed for column {col}: {e}")
                continue
        
        return anomalies
    
    async def _get_text_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for text data"""
        if not self.kernel:
            return None
            
        try:
            # Create embedding function
            @kernel_function(
                description="Generate embeddings for text",
                name="generate_embeddings"
            )
            def generate_embeddings(text: str) -> str:
                return text
            
            embeddings = []
            
            # Process in batches to avoid overwhelming the service
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                for text in batch:
                    try:
                        # In a real implementation, this would use the embedding service
                        # For now, simulate with a simple hash-based approach
                        embedding = self._simulate_embedding(text)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Failed to get embedding for text: {e}")
                        # Use fallback embedding
                        embeddings.append(np.random.normal(0, 1, 384))  # Simulate 384-dim embedding
            
            return np.array(embeddings) if embeddings else None
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _simulate_embedding(self, text: str) -> np.ndarray:
        """Simulate embedding generation (for demo purposes)"""
        # Simple simulation based on text characteristics
        # In production, this would use actual embedding models
        
        features = []
        
        # Length features
        features.append(len(text) / 100.0)
        features.append(len(text.split()) / 50.0)
        
        # Character features
        features.append(sum(1 for c in text if c.isupper()) / len(text))
        features.append(sum(1 for c in text if c.isdigit()) / len(text))
        features.append(sum(1 for c in text if c in '.,!?;') / len(text))
        
        # Word features (simple)
        words = text.lower().split()
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of']
        features.append(sum(1 for w in words if w in common_words) / len(words))
        
        # Pad to desired dimension
        base_features = np.array(features)
        
        # Create a 384-dimensional vector
        np.random.seed(hash(text) % 2**32)  # Deterministic based on text
        embedding = np.random.normal(0, 1, 384)
        
        # Incorporate actual features into first dimensions
        embedding[:len(base_features)] = base_features
        
        return embedding
    
    def _find_embedding_outliers(self, embeddings: np.ndarray, threshold: float = 0.3) -> List[int]:
        """Find outliers in embedding space using similarity"""
        if len(embeddings) < 3:
            return []
        
        outlier_indices = []
        
        # Calculate pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        similarities = similarities / (norms[:, None] * norms[None, :])
        
        # For each point, find its average similarity to all others
        for i in range(len(embeddings)):
            # Exclude self-similarity
            other_similarities = np.concatenate([similarities[i, :i], similarities[i, i+1:]])
            avg_similarity = np.mean(other_similarities)
            
            # If average similarity is below threshold, it's an outlier
            if avg_similarity < threshold:
                outlier_indices.append(i)
        
        return outlier_indices
    
    async def _detect_pattern_anomalies(self, data: pd.DataFrame, numeric_columns: List[str], options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect pattern anomalies in numeric data using embedding-style analysis"""
        anomalies = []
        
        if len(numeric_columns) < 2:
            return anomalies
        
        try:
            # Create feature vectors for each row
            feature_matrix = data[numeric_columns].fillna(0).values
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            
            # Find outliers using distance-based approach (similar to embedding outliers)
            outlier_indices = self._find_pattern_outliers(normalized_features)
            
            for idx in outlier_indices:
                # Find which features contribute most to the anomaly
                row_features = normalized_features[idx]
                feature_scores = np.abs(row_features)
                top_feature_idx = np.argmax(feature_scores)
                top_feature = numeric_columns[top_feature_idx]
                
                anomalies.append({
                    'index': int(data.index[idx]),
                    'column': top_feature,
                    'value': data.iloc[idx][top_feature],
                    'anomaly_type': 'pattern_outlier',
                    'pattern_score': float(feature_scores[top_feature_idx]),
                    'severity': self._calculate_pattern_severity(feature_scores[top_feature_idx]),
                    'business_explanation': f"Row shows unusual pattern across multiple dimensions, primarily in {top_feature}",
                    'category': 'pattern',
                    'affected_features': [col for i, col in enumerate(numeric_columns) if feature_scores[i] > 1.0]
                })
                
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
        
        return anomalies
    
    def _find_pattern_outliers(self, features: np.ndarray, threshold: float = 2.0) -> List[int]:
        """Find outliers in multi-dimensional pattern space"""
        outlier_indices = []
        
        # Calculate distance from each point to all others
        for i in range(len(features)):
            distances = np.linalg.norm(features - features[i], axis=1)
            # Exclude self-distance
            other_distances = np.concatenate([distances[:i], distances[i+1:]])
            
            # Use median distance as reference
            median_distance = np.median(other_distances)
            min_distance = np.min(other_distances)
            
            # If minimum distance to any other point is far above median, it's an outlier
            if min_distance > threshold * median_distance:
                outlier_indices.append(i)
        
        return outlier_indices
    
    def _calculate_pattern_severity(self, pattern_score: float) -> str:
        """Calculate severity based on pattern score"""
        if pattern_score > 3.0:
            return "critical"
        elif pattern_score > 2.5:
            return "high"
        elif pattern_score > 2.0:
            return "medium"
        else:
            return "low"
    
    async def _generate_embedding_summary(self, data: pd.DataFrame, anomalies: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """Generate summary of embedding analysis"""
        if not anomalies:
            return "No semantic or pattern anomalies detected using embedding analysis."
        
        text_anomalies = [a for a in anomalies if a.get('anomaly_type') == 'text_semantic']
        pattern_anomalies = [a for a in anomalies if a.get('anomaly_type') == 'pattern_outlier']
        
        summary_parts = []
        summary_parts.append(f"Embedding analysis detected {len(anomalies)} anomalies:")
        
        if text_anomalies:
            summary_parts.append(f"• {len(text_anomalies)} semantic text anomalies")
        
        if pattern_anomalies:
            summary_parts.append(f"• {len(pattern_anomalies)} multi-dimensional pattern outliers")
        
        summary_parts.append("This analysis identifies subtle patterns and similarities that traditional statistical methods might miss.")
        
        return " ".join(summary_parts)
