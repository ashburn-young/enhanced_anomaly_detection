"""
Data Processing Module for Retail Anomaly Detection
Handles data loading, preprocessing, and sample data generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataProcessor:
    """Core data processing functionality"""
    
    @staticmethod
    def load_sample_data() -> pd.DataFrame:
        """Load or generate sample retail data"""
        sample_path = Path("enhanced_sample_data.csv")
        
        if sample_path.exists():
            try:
                df = pd.read_csv(sample_path)
                logger.info(f"Loaded existing sample data: {len(df)} records")
                return df
            except Exception as e:
                logger.warning(f"Failed to load existing sample data: {e}")
        
        # Generate new sample data
        logger.info("Generating new sample data")
        df = DataProcessor._generate_enhanced_sample_data()
        
        # Save for future use
        try:
            df.to_csv(sample_path, index=False)
            logger.info(f"Saved sample data to {sample_path}")
        except Exception as e:
            logger.warning(f"Failed to save sample data: {e}")
        
        return df
    
    @staticmethod
    def _generate_enhanced_sample_data(num_records: int = 1000) -> pd.DataFrame:
        """Generate realistic retail sample data with injected anomalies"""
        np.random.seed(42)
        
        # Generate base data
        data = {
            'product_id': [f"PROD_{i:04d}" for i in range(num_records)],
            'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], num_records),
            'sales_amount': np.random.normal(100, 30, num_records),
            'quantity_sold': np.random.poisson(5, num_records),
            'price_per_unit': np.random.normal(25, 8, num_records),
            'discount_percentage': np.random.uniform(0, 0.3, num_records),
            'customer_rating': np.random.normal(4.0, 0.8, num_records),
            'inventory_level': np.random.normal(50, 15, num_records),
            'days_since_launch': np.random.randint(1, 365, num_records),
            'supplier_id': np.random.choice([f"SUP_{i:03d}" for i in range(20)], num_records),
            'store_location': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], num_records),
            'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], num_records)
        }
        
        df = pd.DataFrame(data)
        
        # Add timestamps
        start_date = datetime.now() - timedelta(days=90)
        df['transaction_date'] = [
            start_date + timedelta(days=np.random.randint(0, 90))
            for _ in range(num_records)
        ]
        
        # Ensure positive values where needed
        df['sales_amount'] = np.abs(df['sales_amount'])
        df['price_per_unit'] = np.abs(df['price_per_unit'])
        df['inventory_level'] = np.abs(df['inventory_level'])
        df['customer_rating'] = np.clip(df['customer_rating'], 1, 5)
        
        # Calculate derived fields
        df['total_revenue'] = df['sales_amount'] * df['quantity_sold']
        df['profit_margin'] = (df['price_per_unit'] - (df['price_per_unit'] * 0.6)) / df['price_per_unit']
        df['inventory_turnover'] = df['quantity_sold'] / np.maximum(df['inventory_level'], 1)
        
        # Inject controlled anomalies
        df = DataProcessor._inject_anomalies(df)
        
        return df
    
    @staticmethod
    def _inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
        """Inject realistic anomalies into the dataset"""
        anomaly_indices = []
        
        # High sales anomaly
        high_sales_idx = np.random.choice(df.index, size=5, replace=False)
        df.loc[high_sales_idx, 'sales_amount'] = df['sales_amount'].mean() + 5 * df['sales_amount'].std()
        df.loc[high_sales_idx, 'total_revenue'] = df.loc[high_sales_idx, 'sales_amount'] * df.loc[high_sales_idx, 'quantity_sold']
        anomaly_indices.extend(high_sales_idx)
        
        # Low price anomaly (potential pricing error)
        low_price_idx = np.random.choice(df.index, size=3, replace=False)
        df.loc[low_price_idx, 'price_per_unit'] = df['price_per_unit'].mean() * 0.1  # 90% discount
        df.loc[low_price_idx, 'discount_percentage'] = 0.9
        anomaly_indices.extend(low_price_idx)
        
        # High inventory with low sales (dead stock)
        dead_stock_idx = np.random.choice(df.index, size=4, replace=False)
        df.loc[dead_stock_idx, 'inventory_level'] = df['inventory_level'].mean() + 3 * df['inventory_level'].std()
        df.loc[dead_stock_idx, 'quantity_sold'] = 0
        df.loc[dead_stock_idx, 'sales_amount'] = 0
        anomaly_indices.extend(dead_stock_idx)
        
        # Unusual customer rating pattern
        rating_anomaly_idx = np.random.choice(df.index, size=2, replace=False)
        df.loc[rating_anomaly_idx, 'customer_rating'] = 1.0  # Very low rating
        anomaly_indices.extend(rating_anomaly_idx)
        
        # Extreme discount anomaly
        discount_anomaly_idx = np.random.choice(df.index, size=3, replace=False)
        df.loc[discount_anomaly_idx, 'discount_percentage'] = 0.8  # 80% discount
        anomaly_indices.extend(discount_anomaly_idx)
        
        # Mark anomalies for validation (in real scenario, this wouldn't exist)
        df['is_injected_anomaly'] = False
        df.loc[anomaly_indices, 'is_injected_anomaly'] = True
        
        return df
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate uploaded data and provide recommendations"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'summary': {}
        }
        
        # Check basic structure
        if len(df) == 0:
            validation_results['errors'].append("Dataset is empty")
            validation_results['is_valid'] = False
            return validation_results
        
        if len(df.columns) == 0:
            validation_results['errors'].append("Dataset has no columns")
            validation_results['is_valid'] = False
            return validation_results
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            validation_results['warnings'].append("No numeric columns found - statistical analysis may be limited")
        
        # Check data quality issues
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            high_missing_cols = missing_data[missing_data > len(df) * 0.1].index.tolist()
            if high_missing_cols:
                validation_results['warnings'].append(f"High missing data in columns: {high_missing_cols}")
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(f"Found {duplicate_count} duplicate records")
            validation_results['recommendations'].append("Consider removing or investigating duplicate records")
        
        # Check data types
        validation_results['summary'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'missing_values': missing_data.sum(),
            'duplicate_records': duplicate_count,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Size recommendations
        if len(df) > 10000:
            validation_results['recommendations'].append("Large dataset detected - consider sampling for faster processing")
        elif len(df) < 100:
            validation_results['recommendations'].append("Small dataset - anomaly detection may be less reliable")
        
        return validation_results
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame, options: Dict = None) -> Tuple[pd.DataFrame, Dict]:
        """Preprocess data for anomaly detection"""
        if options is None:
            options = {}
        
        preprocessing_log = {
            'steps_applied': [],
            'original_shape': df.shape,
            'final_shape': None,
            'warnings': []
        }
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Handle missing values
        if options.get('handle_missing', True):
            initial_missing = processed_df.isnull().sum().sum()
            if initial_missing > 0:
                # Numeric columns: fill with median
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if processed_df[col].isnull().any():
                        median_val = processed_df[col].median()
                        processed_df[col].fillna(median_val, inplace=True)
                
                # Categorical columns: fill with mode
                categorical_cols = processed_df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if processed_df[col].isnull().any():
                        mode_val = processed_df[col].mode()[0] if len(processed_df[col].mode()) > 0 else 'Unknown'
                        processed_df[col].fillna(mode_val, inplace=True)
                
                preprocessing_log['steps_applied'].append(f"Filled {initial_missing} missing values")
        
        # Remove duplicates
        if options.get('remove_duplicates', False):
            initial_size = len(processed_df)
            processed_df.drop_duplicates(inplace=True)
            removed_count = initial_size - len(processed_df)
            if removed_count > 0:
                preprocessing_log['steps_applied'].append(f"Removed {removed_count} duplicate records")
        
        # Convert datetime columns
        datetime_columns = processed_df.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            preprocessing_log['steps_applied'].append(f"Identified {len(datetime_columns)} datetime columns")
        
        # Outlier capping (optional)
        if options.get('cap_outliers', False):
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            capped_cols = []
            
            for col in numeric_cols:
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                initial_outliers = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
                if initial_outliers > 0:
                    processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
                    capped_cols.append(col)
            
            if capped_cols:
                preprocessing_log['steps_applied'].append(f"Capped outliers in columns: {capped_cols}")
        
        preprocessing_log['final_shape'] = processed_df.shape
        
        return processed_df, preprocessing_log
    
    @staticmethod
    def get_data_insights(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data insights"""
        insights = {
            'basic_stats': {},
            'column_types': {},
            'correlations': {},
            'distributions': {},
            'recommendations': []
        }
        
        # Basic statistics
        insights['basic_stats'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum()
        }
        
        # Column type analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        insights['column_types'] = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
        
        # Correlation analysis (for numeric columns)
        if len(numeric_cols) > 1:
            try:
                correlation_matrix = df[numeric_cols].corr()
                # Find high correlations
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # High correlation threshold
                            high_corr_pairs.append({
                                'column1': correlation_matrix.columns[i],
                                'column2': correlation_matrix.columns[j],
                                'correlation': corr_val
                            })
                
                insights['correlations'] = {
                    'high_correlations': high_corr_pairs,
                    'matrix': correlation_matrix.to_dict()
                }
            except Exception as e:
                insights['correlations'] = {'error': str(e)}
        
        # Distribution analysis
        if len(numeric_cols) > 0:
            distribution_stats = {}
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                try:
                    col_data = df[col].dropna()
                    distribution_stats[col] = {
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'skewness': float(col_data.skew()),
                        'outlier_count': int(((col_data < col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))) |
                                             (col_data > col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)))).sum())
                    }
                except Exception as e:
                    distribution_stats[col] = {'error': str(e)}
            
            insights['distributions'] = distribution_stats
        
        # Generate recommendations
        recommendations = []
        
        if insights['basic_stats']['missing_values']:
            max_missing = max(insights['basic_stats']['missing_values'].values())
            if max_missing > len(df) * 0.1:
                recommendations.append("Consider addressing high missing data before analysis")
        
        if insights['basic_stats']['duplicate_records'] > 0:
            recommendations.append("Remove duplicate records for cleaner analysis")
        
        if len(numeric_cols) < 3:
            recommendations.append("Limited numeric columns may reduce anomaly detection effectiveness")
        
        if len(df) > 50000:
            recommendations.append("Large dataset - consider sampling for faster processing")
        
        insights['recommendations'] = recommendations
        
        return insights


class SampleDataGenerator:
    """Advanced sample data generator for testing and demos"""
    
    @staticmethod
    def generate_retail_scenarios() -> Dict[str, pd.DataFrame]:
        """Generate different retail scenarios for testing"""
        scenarios = {}
        
        # Scenario 1: Normal retail data
        scenarios['normal_retail'] = SampleDataGenerator._generate_normal_retail_data()
        
        # Scenario 2: Fraud detection scenario
        scenarios['fraud_detection'] = SampleDataGenerator._generate_fraud_scenario()
        
        # Scenario 3: Inventory management scenario
        scenarios['inventory_management'] = SampleDataGenerator._generate_inventory_scenario()
        
        # Scenario 4: Pricing anomalies scenario
        scenarios['pricing_anomalies'] = SampleDataGenerator._generate_pricing_scenario()
        
        return scenarios
    
    @staticmethod
    def _generate_normal_retail_data(num_records: int = 500) -> pd.DataFrame:
        """Generate normal retail data without anomalies"""
        np.random.seed(123)
        
        data = {
            'product_id': [f"PROD_{i:04d}" for i in range(num_records)],
            'sales_amount': np.random.normal(100, 25, num_records),
            'quantity': np.random.poisson(3, num_records),
            'price': np.random.normal(30, 5, num_records),
            'category': np.random.choice(['A', 'B', 'C', 'D'], num_records),
            'store_id': np.random.choice([f"STORE_{i}" for i in range(10)], num_records)
        }
        
        df = pd.DataFrame(data)
        df['sales_amount'] = np.abs(df['sales_amount'])
        df['price'] = np.abs(df['price'])
        df['revenue'] = df['sales_amount'] * df['quantity']
        
        return df
    
    @staticmethod
    def _generate_fraud_scenario(num_records: int = 500) -> pd.DataFrame:
        """Generate data with fraud-like anomalies"""
        df = SampleDataGenerator._generate_normal_retail_data(num_records)
        
        # Inject fraud patterns
        fraud_indices = np.random.choice(df.index, size=10, replace=False)
        
        # Unusual high-value transactions
        df.loc[fraud_indices[:5], 'sales_amount'] = df['sales_amount'].mean() + 10 * df['sales_amount'].std()
        
        # Unusual patterns (multiple small transactions)
        df.loc[fraud_indices[5:], 'quantity'] = 1
        df.loc[fraud_indices[5:], 'sales_amount'] = df['sales_amount'].mean() * 0.1
        
        return df
    
    @staticmethod
    def _generate_inventory_scenario(num_records: int = 500) -> pd.DataFrame:
        """Generate data with inventory-related anomalies"""
        df = SampleDataGenerator._generate_normal_retail_data(num_records)
        
        # Add inventory columns
        df['inventory_level'] = np.random.normal(100, 30, num_records)
        df['reorder_point'] = np.random.normal(20, 5, num_records)
        df['days_out_of_stock'] = np.random.poisson(0.5, num_records)
        
        # Inject inventory anomalies
        anomaly_indices = np.random.choice(df.index, size=15, replace=False)
        
        # Overstocking
        df.loc[anomaly_indices[:5], 'inventory_level'] = df['inventory_level'].mean() + 5 * df['inventory_level'].std()
        df.loc[anomaly_indices[:5], 'quantity'] = 0  # No sales despite high inventory
        
        # Stockouts
        df.loc[anomaly_indices[5:10], 'inventory_level'] = 0
        df.loc[anomaly_indices[5:10], 'days_out_of_stock'] = np.random.randint(5, 30, 5)
        
        # Fast-moving items
        df.loc[anomaly_indices[10:], 'quantity'] = df['quantity'].mean() + 3 * df['quantity'].std()
        
        return df
    
    @staticmethod
    def _generate_pricing_scenario(num_records: int = 500) -> pd.DataFrame:
        """Generate data with pricing anomalies"""
        df = SampleDataGenerator._generate_normal_retail_data(num_records)
        
        # Add pricing columns
        df['cost'] = df['price'] * np.random.uniform(0.5, 0.8, num_records)
        df['margin'] = (df['price'] - df['cost']) / df['price']
        df['competitor_price'] = df['price'] * np.random.uniform(0.9, 1.1, num_records)
        
        # Inject pricing anomalies
        anomaly_indices = np.random.choice(df.index, size=12, replace=False)
        
        # Pricing errors (extremely low prices)
        df.loc[anomaly_indices[:4], 'price'] = df['price'].mean() * 0.1
        df.loc[anomaly_indices[:4], 'margin'] = (df.loc[anomaly_indices[:4], 'price'] - df.loc[anomaly_indices[:4], 'cost']) / df.loc[anomaly_indices[:4], 'price']
        
        # Premium pricing (very high prices)
        df.loc[anomaly_indices[4:8], 'price'] = df['price'].mean() + 5 * df['price'].std()
        
        # Negative margins
        df.loc[anomaly_indices[8:], 'price'] = df.loc[anomaly_indices[8:], 'cost'] * 0.8
        df.loc[anomaly_indices[8:], 'margin'] = (df.loc[anomaly_indices[8:], 'price'] - df.loc[anomaly_indices[8:], 'cost']) / df.loc[anomaly_indices[8:], 'price']
        
        return df
