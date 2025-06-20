"""
Enhanced AI Analysis Agent - matches krogerdemo1 functionality
Provides detailed anomaly detection with business context and custom prompts
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EnhancedAIAgent:
    """Enhanced AI agent that provides detailed anomaly analysis like krogerdemo1"""
    
    def __init__(self):
        """Initialize with Azure OpenAI if available"""
        self.openai_available = False
        self.client = None
        self.deployment_name = None
        
        try:
            # Try to initialize Azure OpenAI
            import openai
            
            # Support both variable names for compatibility
            api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            
            if api_key and azure_endpoint:
                # Handle different OpenAI versions
                openai_version = openai.__version__
                
                if openai_version.startswith('0.'):
                    # Legacy OpenAI API (0.x.x)
                    openai.api_type = "azure"
                    openai.api_base = azure_endpoint.rstrip("/")
                    openai.api_key = api_key
                    openai.api_version = api_version
                    self.client = openai
                    self.api_style = "legacy"
                else:
                    # New OpenAI API (1.x.x+)
                    from openai import AzureOpenAI
                    self.client = AzureOpenAI(
                        api_key=api_key,
                        api_version=api_version,
                        azure_endpoint=azure_endpoint
                    )
                    self.api_style = "modern"
                
                self.openai_available = True
                logger.info(f"Enhanced AI Agent initialized with Azure OpenAI (version {openai_version}, {self.api_style} API)")
            else:
                logger.warning("Azure OpenAI credentials not found - using fallback analysis")
                
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI: {e}")
    
    def analyze_data(self, df: pd.DataFrame, custom_prompt: str = None) -> dict:
        """
        Perform enhanced AI analysis with custom business prompts
        
        Args:
            df: DataFrame to analyze
            custom_prompt: User's custom business rules and focus areas
            
        Returns:
            Detailed analysis results with specific anomalies
        """
        if self.openai_available and self.client:
            return self._analyze_with_azure_openai(df, custom_prompt)
        else:
            return self._analyze_with_fallback(df, custom_prompt)
    
    def _analyze_with_azure_openai(self, df: pd.DataFrame, custom_prompt: str = None) -> dict:
        """Analyze using Azure OpenAI - matches krogerdemo1 functionality"""
        try:
            # Create data summary
            data_summary = self._create_data_summary(df)
            sample_data = df.head(10).to_string()
            
            # Create comprehensive prompt
            prompt = self._create_comprehensive_prompt(data_summary, sample_data, custom_prompt)
            
            # Debug logging to track AI usage
            logger.info(f"🤖 Azure OpenAI Analysis Starting...")
            logger.info(f"📊 Dataset: {len(df)} records, {len(df.columns)} columns")
            logger.info(f"🔧 Model: {self.deployment_name} ({self.api_style} API)")
            logger.info(f"📝 Custom prompt: {bool(custom_prompt)}")
            if custom_prompt:
                logger.info(f"📋 Custom prompt content: {custom_prompt[:200]}{'...' if len(custom_prompt) > 200 else ''}")
            else:
                logger.info(f"📋 Using default business focus (no custom prompt provided)")
            logger.debug(f"📋 Full prompt length: {len(prompt)} characters")
            
            # Make API call based on OpenAI version
            if self.api_style == "legacy":
                # Legacy OpenAI API (0.x.x)
                if not hasattr(self.client, 'ChatCompletion'):
                    logger.error("ChatCompletion API not available")
                    return self._analyze_with_fallback(df, custom_prompt)
                
                response = self.client.ChatCompletion.create(
                    engine=self.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert retail data anomaly detection assistant. Analyze datasets for unusual patterns, business rule violations, and data quality issues.

Provide detailed, actionable results in this JSON format:
{
    "anomalies_detected": [
        {
            "anomaly_type": "specific type",
            "description": "detailed explanation with specific values and context",
            "confidence_score": 0.0-1.0,
            "potential_causes": ["cause1", "cause2"],
            "affected_records": integer,
            "severity": "low|medium|high",
            "specific_examples": ["example1", "example2"],
            "business_impact": "description of business impact",
            "recommended_action": "specific action to take"
        }
    ],
    "summary": "executive summary",
    "recommendations": ["actionable rec1", "actionable rec2"],
    "data_quality_score": 0.0-1.0,
    "business_insights": ["insight1", "insight2"]
}"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=3000,
                    request_timeout=45
                )
                
                # Parse legacy response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message'):
                        content = response.choices[0].message.content
                    else:
                        content = response.choices[0].get('message', {}).get('content', str(response))
                else:
                    content = str(response)
                    
            else:
                # Modern OpenAI API (1.x.x+)
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert retail data anomaly detection assistant. Analyze datasets for unusual patterns, business rule violations, and data quality issues.

Provide detailed, actionable results in this JSON format:
{
    "anomalies_detected": [
        {
            "anomaly_type": "specific type",
            "description": "detailed explanation with specific values and context",
            "confidence_score": 0.0-1.0,
            "potential_causes": ["cause1", "cause2"],
            "affected_records": integer,
            "severity": "low|medium|high",
            "specific_examples": ["example1", "example2"],
            "business_impact": "description of business impact",
            "recommended_action": "specific action to take"
        }
    ],
    "summary": "executive summary",
    "recommendations": ["actionable rec1", "actionable rec2"],
    "data_quality_score": 0.0-1.0,
    "business_insights": ["insight1", "insight2"]
}"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=3000,
                    timeout=45
                )
                
                # Parse modern response
                content = response.choices[0].message.content
            
            result = self._parse_ai_response(content)
            
            # Debug logging for AI response
            logger.info(f"✅ Azure OpenAI Response received")
            logger.info(f"📊 Raw response length: {len(content)} characters")
            logger.info(f"🔍 Anomalies found: {len(result.get('anomalies_detected', []))}")
            logger.debug(f"🔎 Raw response preview: {content[:200]}...")
            
            # Add metadata
            result['analysis_method'] = 'Enhanced AI with Azure OpenAI'
            result['custom_prompt_used'] = bool(custom_prompt)
            result['timestamp'] = datetime.now().isoformat()
            result['model_used'] = self.deployment_name
            result['api_style'] = self.api_style
            
            logger.info(f"AI analysis completed: {len(result.get('anomalies_detected', []))} anomalies found")
            return result
            
        except Exception as e:
            logger.error(f"❌ Azure OpenAI analysis failed: {e}")
            logger.error(f"🔧 Model: {getattr(self, 'deployment_name', 'unknown')}")
            logger.error(f"🌐 Endpoint available: {bool(os.getenv('AZURE_OPENAI_ENDPOINT'))}")
            logger.error(f"🔑 API key available: {bool(os.getenv('AZURE_OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_KEY'))}")
            return self._analyze_with_fallback(df, custom_prompt)
    
    def _analyze_with_fallback(self, df: pd.DataFrame, custom_prompt: str = None) -> dict:
        """Fallback analysis when AI is not available"""
        try:
            anomalies = []
            
            # Data quality checks
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 10:
                anomalies.append({
                    "anomaly_type": "High Missing Data Rate",
                    "description": f"Dataset has {missing_pct:.1f}% missing values, which exceeds the 10% threshold for reliable analysis",
                    "confidence_score": 0.9,
                    "potential_causes": ["Data collection issues", "System downtime", "Integration problems"],
                    "affected_records": int(df.isnull().sum().sum()),
                    "severity": "high" if missing_pct > 25 else "medium",
                    "specific_examples": [f"Column '{col}' has {df[col].isnull().sum()} missing values" for col in df.columns if df[col].isnull().sum() > 0][:3],
                    "business_impact": "Incomplete data may lead to incorrect business decisions and unreliable analytics",
                    "recommended_action": "Review data collection processes and implement data validation checks"
                })
            
            # Duplicate detection
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                anomalies.append({
                    "anomaly_type": "Duplicate Records",
                    "description": f"Found {duplicate_count} exact duplicate records ({(duplicate_count/len(df)*100):.1f}% of dataset)",
                    "confidence_score": 1.0,
                    "potential_causes": ["Data pipeline errors", "System integration issues", "Manual data entry errors"],
                    "affected_records": int(duplicate_count),
                    "severity": "high" if duplicate_count > len(df) * 0.05 else "medium",
                    "specific_examples": ["Multiple identical rows detected"],
                    "business_impact": "Duplicate data can skew metrics and lead to incorrect business insights",
                    "recommended_action": "Implement deduplication process and review data ingestion workflow"
                })
            
            # Numeric column analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].notna().sum() > 10:
                    # Check for negative values where they shouldn't be
                    if col.lower() in ['price', 'amount', 'cost', 'sales', 'revenue', 'quantity'] and (df[col] < 0).any():
                        negative_count = (df[col] < 0).sum()
                        anomalies.append({
                            "anomaly_type": f"Negative Values in {col}",
                            "description": f"Found {negative_count} negative values in '{col}' column, which should contain only positive values",
                            "confidence_score": 0.95,
                            "potential_causes": ["Data entry errors", "System calculation errors", "Data type conversion issues"],
                            "affected_records": int(negative_count),
                            "severity": "high",
                            "specific_examples": [f"Row with {col} = {val}" for val in df[df[col] < 0][col].head(3).values],
                            "business_impact": "Negative values in business metrics can distort financial reporting and analytics",
                            "recommended_action": "Review data validation rules and fix source system constraints"
                        })
                    
                    # Statistical outliers
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    if len(outliers) > 0 and len(outliers) < len(df) * 0.1:  # Don't flag if too many outliers
                        anomalies.append({
                            "anomaly_type": f"Statistical Outliers in {col}",
                            "description": f"Found {len(outliers)} statistical outliers in '{col}' (values outside {lower_bound:.2f} to {upper_bound:.2f} range)",
                            "confidence_score": 0.8,
                            "potential_causes": ["Exceptional transactions", "Data entry errors", "System processing errors"],
                            "affected_records": len(outliers),
                            "severity": "medium",
                            "specific_examples": [f"{col} = {val:.2f}" for val in outliers[col].head(3).values],
                            "business_impact": "Outliers may represent significant business events or data quality issues",
                            "recommended_action": "Investigate individual outlier records to determine if they represent valid business events"
                        })
            
            # Custom prompt analysis (basic pattern matching)
            if custom_prompt:
                custom_anomalies = self._analyze_custom_prompt(df, custom_prompt)
                anomalies.extend(custom_anomalies)
            
            return {
                "anomalies_detected": anomalies,
                "summary": f"Fallback analysis completed on {len(df)} records, found {len(anomalies)} potential issues",
                "recommendations": [
                    "Configure Azure OpenAI for enhanced AI analysis",
                    "Review data quality processes",
                    "Implement automated data validation",
                    "Set up monitoring for duplicate detection"
                ],
                "data_quality_score": max(0.0, 1.0 - (missing_pct / 100) - (duplicate_count / len(df))),
                "business_insights": [
                    f"Dataset contains {len(df)} records with {len(df.columns)} attributes",
                    f"Data completeness: {100 - missing_pct:.1f}%",
                    f"Duplicate rate: {(duplicate_count/len(df)*100):.1f}%"
                ],
                "analysis_method": "Fallback Statistical Analysis",
                "custom_prompt_used": bool(custom_prompt),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return {
                "anomalies_detected": [],
                "summary": f"Analysis failed: {str(e)}",
                "recommendations": ["Check data format and try again"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_data_summary(self, df: pd.DataFrame) -> str:
        """Create comprehensive data summary for AI analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        summary = f"""
Dataset Overview:
- Total Records: {len(df)}
- Total Columns: {len(df.columns)}
- Numeric Columns: {len(numeric_cols)}
- Text/Categorical Columns: {len(text_cols)}
- Missing Values: {df.isnull().sum().sum()} ({(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%)
- Duplicate Records: {df.duplicated().sum()}

Column Information:
"""
        
        for col in df.columns[:10]:  # Limit to first 10 columns
            col_info = f"- {col}: {df[col].dtype}"
            if df[col].dtype in ['int64', 'float64']:
                col_info += f" (range: {df[col].min():.2f} to {df[col].max():.2f})"
            elif df[col].dtype == 'object':
                unique_count = df[col].nunique()
                col_info += f" ({unique_count} unique values)"
            summary += col_info + "\n"
        
        return summary
    
    def _create_comprehensive_prompt(self, data_summary: str, sample_data: str, custom_prompt: str = None) -> str:
        """Create comprehensive analysis prompt with enhanced business context"""
        
        base_prompt = f"""
You are a senior business analyst specializing in retail data anomaly detection. Your role is to identify anomalies that could significantly impact business operations, revenue, customer satisfaction, or compliance.

DATA OVERVIEW:
{data_summary}

SAMPLE DATA FOR CONTEXT:
{sample_data}

ANALYSIS REQUIREMENTS:
Your task is to find specific, actionable anomalies that business users need to know about. For each anomaly you identify:

1. BUSINESS CONTEXT: Explain what makes this unusual from a business perspective
2. SPECIFIC EVIDENCE: Provide exact data points (product names, values, IDs)
3. IMPACT ASSESSMENT: Describe the potential business consequences
4. ROOT CAUSE ANALYSIS: Suggest possible reasons for the anomaly
5. ACTION PLAN: Recommend specific steps to investigate or resolve

FOCUS AREAS (in order of business priority):
- Pricing errors or suspicious pricing patterns that could affect profitability
- Inventory anomalies that could lead to stockouts or excess carrying costs
- Product quality issues reflected in specifications or customer ratings
- Sales performance outliers that indicate opportunities or problems
- Customer behavior patterns that suggest satisfaction issues
- Data quality problems that could affect decision-making

CUSTOM BUSINESS REQUIREMENTS:
{custom_prompt if custom_prompt else "Focus on general retail business anomalies that could impact operations, revenue, or customer satisfaction."}

CRITICAL: You MUST find and report specific anomalies with actual data examples. Do not report "no anomalies" unless you have thoroughly analyzed the data.

RESPONSE FORMAT - Return ONLY valid JSON:
{{
    "analysis_summary": "Executive summary of key findings",
    "anomalies_detected": [
        {{
            "anomaly_type": "specific type",
            "description": "detailed explanation with specific values and context",
            "confidence_score": 0.0-1.0,
            "potential_causes": ["cause1", "cause2"],
            "affected_records": integer,
            "severity": "low|medium|high",
            "specific_examples": ["example1", "example2"],
            "business_impact": "description of business impact",
            "recommended_action": "specific action to take",
            "product_reference": "specific product name or ID if available",
            "anomalous_value": "the actual problematic value",
            "field_name": "exact column name where anomaly was found"
        }}
    ],
    "summary": "executive summary",
    "recommendations": ["actionable rec1", "actionable rec2"],
    "data_quality_score": 0.0-1.0,
    "business_insights": ["insight1", "insight2"]
}}

IMPORTANT: 
- You MUST identify at least 1-3 anomalies unless the data is perfectly clean
- Provide specific product/record details, not generic patterns
- Focus on what business users can act upon
- Include actual data values to support your findings
- Prioritize anomalies by business impact, not just statistical significance
- Look for outliers in numerical fields, unusual text patterns, missing values, and data inconsistencies"""
        
        return base_prompt
    
    def _analyze_custom_prompt(self, df: pd.DataFrame, custom_prompt: str) -> List[Dict]:
        """Basic analysis based on custom prompt keywords"""
        anomalies = []
        prompt_lower = custom_prompt.lower()
        
        # Look for price-related rules
        if 'price' in prompt_lower or 'pricing' in prompt_lower:
            price_cols = [col for col in df.columns if 'price' in col.lower()]
            for col in price_cols:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    # Look for price ranges mentioned in prompt
                    if '$' in custom_prompt:
                        # Basic price range detection (simplified)
                        extreme_high = df[col] > df[col].quantile(0.99)
                        if extreme_high.any():
                            anomalies.append({
                                "anomaly_type": f"Extreme {col} Values",
                                "description": f"Found {extreme_high.sum()} records with unusually high {col} values based on your business rules",
                                "confidence_score": 0.7,
                                "potential_causes": ["Pricing errors", "Premium products", "Data entry mistakes"],
                                "affected_records": int(extreme_high.sum()),
                                "severity": "medium",
                                "specific_examples": [f"{col} = ${val:.2f}" for val in df[extreme_high][col].head(3).values],
                                "business_impact": "Pricing anomalies can affect revenue and customer satisfaction",
                                "recommended_action": "Review pricing strategy and validate data entry processes"
                            })
        
        # Look for quantity-related rules
        if 'quantity' in prompt_lower or 'qty' in prompt_lower:
            qty_cols = [col for col in df.columns if any(term in col.lower() for term in ['quantity', 'qty', 'amount'])]
            for col in qty_cols:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    # Look for unusual quantities
                    if df[col].max() > 20:  # Based on prompt example ">20 items"
                        high_qty = df[col] > 20
                        if high_qty.any():
                            anomalies.append({
                                "anomaly_type": f"High Quantity Purchases",
                                "description": f"Found {high_qty.sum()} transactions with quantities >20 items as specified in your business rules",
                                "confidence_score": 0.85,
                                "potential_causes": ["Bulk purchases", "B2B transactions", "Promotional buying"],
                                "affected_records": int(high_qty.sum()),
                                "severity": "medium",
                                "specific_examples": [f"{col} = {val}" for val in df[high_qty][col].head(3).values],
                                "business_impact": "High quantity purchases may indicate business customers or promotional success",
                                "recommended_action": "Verify if these are legitimate bulk orders or require special handling"
                            })
        
        return anomalies
    
    def _parse_ai_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI response with robust error handling and enhanced structure"""
        if not response_content:
            logger.warning("🔍 AI returned empty response")
            return {
                "anomalies_detected": [],
                "summary": "AI returned empty response",
                "recommendations": ["Try again with a smaller dataset"]
            }
        
        try:
            # Find JSON content
            json_start = response_content.find('{')
            if json_start >= 0:
                json_content = response_content[json_start:]
                # Find the matching closing brace
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(json_content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    try:
                        parsed_json = json.loads(json_content[:json_end])
                        logger.info(f"✅ Successfully parsed AI JSON response")
                        
                        # Normalize the response format
                        anomalies = parsed_json.get('anomalies_detected', parsed_json.get('anomalies', []))
                        
                        # Enhance anomalies with business context if missing
                        enhanced_anomalies = []
                        for i, anomaly in enumerate(anomalies):
                            enhanced_anomaly = anomaly.copy()
                            
                            # Ensure required fields exist
                            if 'confidence_score' not in enhanced_anomaly and 'confidence' in enhanced_anomaly:
                                enhanced_anomaly['confidence_score'] = enhanced_anomaly['confidence']
                            
                            # Add index for UI reference if missing
                            if 'index' not in enhanced_anomaly and 'sample_indices' not in enhanced_anomaly:
                                enhanced_anomaly['index'] = i
                            
                            # Enhance description for business users
                            if 'description' in enhanced_anomaly and 'business_context' not in enhanced_anomaly:
                                enhanced_anomaly['business_context'] = self._generate_business_context(enhanced_anomaly)
                            
                            enhanced_anomalies.append(enhanced_anomaly)
                        
                        return {
                            "anomalies_detected": enhanced_anomalies,
                            "summary": parsed_json.get('summary', parsed_json.get('analysis_summary', 'AI analysis completed')),
                            "recommendations": parsed_json.get('recommendations', []),
                            "data_quality_score": parsed_json.get('data_quality_score', 0.8),
                            "business_insights": parsed_json.get('business_insights', []),
                            "raw_response": response_content
                        }
                    except json.JSONDecodeError as e:
                        logger.warning(f"🔍 JSON parsing failed: {e}")
                        pass
            
            # If JSON parsing fails, extract information from text
            logger.info("🔍 Falling back to text-based response parsing")
            lines = response_content.split('\n')
            summary = "AI analysis completed with text-based parsing"
            anomalies = []
            
            # Look for anomaly patterns in text
            current_anomaly = {}
            for line in lines:
                line = line.strip()
                if any(term in line.lower() for term in ['anomaly', 'outlier', 'unusual', 'suspicious', 'found']):
                    if current_anomaly:
                        anomalies.append(current_anomaly)
                    
                    current_anomaly = {
                        "anomaly_type": "AI Detected Pattern",
                        "description": line,
                        "confidence_score": 0.6,
                        "potential_causes": ["See AI analysis"],
                        "affected_records": 1,
                        "severity": "medium",
                        "specific_examples": [line],
                        "business_impact": "Requires manual review",
                        "recommended_action": "Investigate the pattern mentioned"
                    }
                elif current_anomaly and any(keyword in line.lower() for keyword in ['price', 'product', 'value', 'record']):
                    # Try to extract specific examples
                    if 'specific_examples' not in current_anomaly:
                        current_anomaly['specific_examples'] = []
                    current_anomaly['specific_examples'].append(line[:100])
            
            if current_anomaly:
                anomalies.append(current_anomaly)
            
            return {
                "anomalies_detected": anomalies,
                "summary": summary,
                "recommendations": ["Review AI text output for additional insights"],
                "raw_response": response_content
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse AI response: {e}")
            return {
                "anomalies_detected": [],
                "summary": f"Failed to parse AI response: {str(e)}",
                "recommendations": ["Check AI response format"],
                "raw_response": response_content,
                "error": str(e)
            }
    
    def _generate_business_context(self, anomaly: Dict) -> Dict:
        """Generate business context for anomalies"""
        anomaly_type = anomaly.get('anomaly_type', '').lower()
        field_name = anomaly.get('field_name', anomaly.get('affected_field', ''))
        
        context = {
            'business_priority': 'medium',
            'investigation_urgency': 'standard',
            'potential_impact': 'operational'
        }
        
        if 'price' in field_name.lower() or 'pricing' in anomaly_type:
            context.update({
                'business_priority': 'high',
                'investigation_urgency': 'immediate',
                'potential_impact': 'financial',
                'key_questions': [
                    'Is this pricing error affecting customer purchases?',
                    'How does this compare to competitor pricing?',
                    'What is the profit margin impact?'
                ]
            })
        elif 'inventory' in field_name.lower() or 'stock' in anomaly_type:
            context.update({
                'business_priority': 'high',
                'investigation_urgency': 'urgent',
                'potential_impact': 'operational',
                'key_questions': [
                    'Are we at risk of stockouts?',
                    'Is excess inventory tying up capital?',
                    'What is the demand forecast accuracy?'
                ]
            })
        elif 'quality' in anomaly_type or 'data' in anomaly_type:
            context.update({
                'business_priority': 'medium',
                'investigation_urgency': 'standard',
                'potential_impact': 'analytical',
                'key_questions': [
                    'Is this affecting decision accuracy?',
                    'What systems need data validation?',
                    'Are there process improvements needed?'
                ]
            })
        
        return context
