"""
UI Components Module for Retail Anomaly Detection
Modular UI components for clean separation of presentation and logic
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any
import re
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

try:
    from business_logic import UserProfile, BusinessContext, AnomalyInsight, FeedbackManager, DataQualityAssessment
except ImportError:
    # Fallback if business_logic module isn't available
    from enum import Enum
    class UserProfile(Enum):
        BUSINESS_USER = "Business User"
        DATA_ANALYST = "Data Analyst" 
        TECHNICAL_EXPERT = "Technical Expert"


class UIStyleManager:
    """Manages consistent UI styling across the application - Theme-compatible version"""
    
    @staticmethod
    def apply_minimal_enhancements():
        """Apply minimal styling enhancements that work with Streamlit's theme system"""
        st.markdown("""
        <style>
            /* Custom styled components that enhance the theme without overriding it */
            .main-header {
                background: linear-gradient(90deg, var(--primary-color) 0%, #1e40af 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            
            .metric-card {
                background: linear-gradient(135deg, var(--secondary-background-color) 0%, #374151 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid var(--primary-color);
                margin: 1rem 0;
                color: var(--text-color);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .anomaly-card {
                background: linear-gradient(135deg, var(--secondary-background-color) 0%, #374151 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid #ef4444;
                margin: 1rem 0;
                color: var(--text-color);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .success-card {
                background: linear-gradient(135deg, var(--secondary-background-color) 0%, #374151 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid var(--primary-color);
                margin: 1rem 0;
                color: var(--text-color);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .warning-card {
                background: linear-gradient(135deg, var(--secondary-background-color) 0%, #374151 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid #f59e0b;
                margin: 1rem 0;
                color: var(--text-color);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .insight-card {
                background: linear-gradient(135deg, var(--secondary-background-color) 0%, #4b5563 100%);
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                border: 1px solid #6b7280;
                color: var(--text-color);
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            /* Enhanced button styling that works with the theme */
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 20px rgba(29, 185, 84, 0.4);
                transition: all 0.3s ease;
            }
            
            /* Role dashboard specific styling */
            .role-dashboard-card {
                background: linear-gradient(135deg, var(--secondary-background-color) 0%, #4b5563 100%);
                padding: 1.5rem;
                border-radius: 12px;
                margin: 1rem;
                color: var(--text-color);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                border: 1px solid #6b7280;
            }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def apply_dark_theme():
        """Legacy method - now just calls minimal enhancements to avoid breaking existing code"""
        UIStyleManager.apply_minimal_enhancements()
    
    @staticmethod
    def create_header(title: str, subtitle: str = ""):
        """Create styled header"""
        header_html = f"""
        <div class='main-header'>
            <h1>🤖 {title}</h1>
            {f"<p>{subtitle}</p>" if subtitle else ""}
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    @staticmethod
    def create_metric_card(title: str, value: str, description: str = "", card_type: str = "metric"):
        """Create styled metric card"""
        card_class = f"{card_type}-card"
        card_html = f"""
        <div class='{card_class}'>
            <h3>{title}</h3>
            <h2>{value}</h2>
            {f"<p>{description}</p>" if description else ""}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


class WelcomeScreenComponent:
    """Welcome screen with onboarding and quick tour"""
    
    @staticmethod
    def display():
        """Display welcome screen"""
        st.markdown("""
        ## 👋 Welcome to AI-Powered Anomaly Detection
        
        This intelligent system uses multiple AI agents to analyze your retail data and identify anomalies 
        that matter to your business.
        
        ### 🎯 What This System Does
        - **Statistical Analysis**: Finds numerical outliers and patterns
        - **AI-Powered Insights**: Provides business context and explanations  
        - **Multi-Modal Detection**: Uses various techniques for comprehensive coverage
        - **Human-in-the-Loop**: Learns from your feedback to improve accuracy
        
        ### 🚀 Quick Start
        1. **Choose your profile** in the sidebar (Business User, Analyst, Expert)
        2. **Upload your data** or use our sample dataset
        3. **Configure analysis** with custom prompts and agent selection
        4. **Review results** and provide feedback to improve the system
        
        ### 💡 Pro Tips
        - Start with sample data to explore capabilities
        - Use custom prompts to focus on specific business concerns
        - Provide feedback on results to train the system for your needs
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            UIStyleManager.create_metric_card("🤖 AI Agents", "7", "Specialized detection agents", "success")
        with col2:
            UIStyleManager.create_metric_card("📊 Methods", "Multiple", "Statistical + AI analysis", "metric")
        with col3:
            UIStyleManager.create_metric_card("🎯 Focus", "Business", "Actionable insights", "warning")


class DataPreviewComponent:
    """Data preview and quality assessment component"""
    
    @staticmethod
    def display_data_overview(df: pd.DataFrame, show_quality: bool = True):
        """Display comprehensive data overview"""
        st.subheader("📊 Data Overview")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        with col4:
            missing_pct = (df.isnull().sum().sum() / df.size) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        # Data preview
        with st.expander("🔍 Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Data quality assessment
        if show_quality:
            try:
                quality_assessment = DataQualityAssessment.assess_data_quality(df)
                DataPreviewComponent._display_quality_assessment(quality_assessment)
            except Exception as e:
                st.warning(f"Data quality assessment unavailable: {e}")
    
    @staticmethod
    def _display_quality_assessment(assessment: Dict):
        """Display data quality assessment"""
        with st.expander("⚡ Data Quality Assessment", expanded=False):
            score = assessment['overall_score']
            
            # Quality score
            if score >= 80:
                st.success(f"Data Quality Score: {score:.1f}/100 - Excellent")
            elif score >= 60:
                st.warning(f"Data Quality Score: {score:.1f}/100 - Good")
            else:
                st.error(f"Data Quality Score: {score:.1f}/100 - Needs Attention")
            
            # Issues and recommendations
            if assessment['issues']:
                st.write("**Issues Found:**")
                for issue in assessment['issues']:
                    st.write(f"⚠️ {issue}")
            
            if assessment['recommendations']:
                st.write("**Recommendations:**")
                for rec in assessment['recommendations']:
                    st.write(f"💡 {rec}")


class SidebarComponent:
    """Comprehensive sidebar with user controls"""
    
    @staticmethod
    def display() -> Dict:
        """Display sidebar and return configuration"""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # User profile selection with dynamic descriptions
            profile_descriptions = {
                "Business User": "🎯 Focus on business insights and actionable recommendations",
                "Data Analyst": "📊 Emphasis on statistical methods and data quality",
                "Technical Expert": "⚙️ Advanced algorithms and technical deep-dives"
            }
            
            profile = st.selectbox(
                "👤 Your Role:",
                ["Business User", "Data Analyst", "Technical Expert"],
                help="Choose your role to customize the interface and analysis approach"
            )
            
            # Show role-specific description
            if profile in profile_descriptions:
                st.caption(profile_descriptions[profile])
            
            # Data source selection
            st.subheader("📁 Data Source")
            data_source = st.radio(
                "Choose data source:",
                ["Upload your data", "Use sample data"],
                help="Upload CSV file or use our sample retail data"
            )
            
            # Custom analysis section
            st.subheader("🎯 Custom Analysis")
            
            # Role-specific focus area templates
            focus_templates = SidebarComponent._get_role_specific_templates(profile)
            
            selected_template = st.selectbox(
                "Quick Templates:",
                list(focus_templates.keys()),
                help="Choose a predefined business focus tailored to your role"
            )
            
            # Auto-populate if template selected
            template_text = focus_templates.get(selected_template, "")
            
            custom_prompt = st.text_area(
                "Business Focus Areas:",
                value=template_text,
                placeholder="e.g., Find sales anomalies that might indicate fraud, unusual customer behavior, or pricing errors...",
                height=100,
                help="Tell the AI what business concerns to focus on"
            )
            
            # Smart agent suggestion based on role and focus area
            suggested_agents = SidebarComponent._suggest_agents_for_role_and_focus(profile, custom_prompt, selected_template)
            if suggested_agents:
                agent_explanation = SidebarComponent._get_agent_explanation_for_role(profile)
                st.info(f"💡 **Recommended agents for {profile}:** {', '.join(suggested_agents)}")
                if agent_explanation:
                    st.caption(agent_explanation)
            
            # Agent selection
            st.subheader("🤖 AI Agent Selection")
            available_agents = [
                "statistical_agent",
                "enhanced_statistical_agent", 
                "ai_agent",
                "embedding_agent",
                "memory_bank_agent",
                "visual_agent"
            ]
            
            # Use suggested agents as default if available
            default_agents = suggested_agents if suggested_agents else ["statistical_agent", "ai_agent"]
            
            selected_agents = st.multiselect(
                "Select agents to run:",
                available_agents,
                default=default_agents,
                help="Choose which AI agents to include in the analysis"
            )
            
            # Analysis configuration
            st.subheader("⚙️ Configuration")
            sensitivity = st.slider(
                "Detection Sensitivity", 
                0.1, 1.0, 0.5,
                help="Higher values detect more anomalies"
            )
            max_anomalies = st.number_input(
                "Max Anomalies to Find", 
                1, 100, 10,
                help="Limit the number of anomalies returned"
            )
            
            # Quick actions
            st.subheader("🔧 Quick Actions")
            show_tutorial = st.button("📚 Show Tutorial")
            show_help = st.button("❓ Help & Tips")
            
            # Map string profile to enum
            profile_mapping = {
                "Business User": UserProfile.BUSINESS_USER,
                "Data Analyst": UserProfile.DATA_ANALYST,
                "Technical Expert": UserProfile.TECHNICAL_EXPERT
            }
            
            return {
                'profile': profile,  # Keep as string for dashboard
                'profile_enum': profile_mapping.get(profile, UserProfile.BUSINESS_USER),  # Enum for business logic
                'data_source': data_source,
                'custom_prompt': custom_prompt,
                'selected_agents': selected_agents,
                'sensitivity': sensitivity,
                'max_anomalies': max_anomalies,
                'show_tutorial': show_tutorial,
                'show_help': show_help
            }
    
    @staticmethod
    def _suggest_agents_for_focus(custom_prompt: str, selected_template: str) -> List[str]:
        """Suggest agents based on business focus area"""
        focus_text = (custom_prompt + " " + selected_template).lower()
        suggested = []
        
        # Always include basic statistical analysis
        suggested.append("statistical_agent")
        
        # AI agent for business context and insights
        if any(word in focus_text for word in ['fraud', 'behavior', 'trend', 'insight', 'business', 'customer', 'market']):
            suggested.append("ai_agent")
        
        # Enhanced statistical for complex patterns
        if any(word in focus_text for word in ['complex', 'pattern', 'multivariate', 'advanced', 'sophisticated']):
            suggested.append("enhanced_statistical_agent")
        
        # Embedding agent for similarity analysis
        if any(word in focus_text for word in ['similar', 'cluster', 'group', 'category', 'segment', 'classification']):
            suggested.append("embedding_agent")
        
        # Memory bank for historical patterns
        if any(word in focus_text for word in ['historical', 'past', 'trend', 'seasonal', 'memory', 'previous']):
            suggested.append("memory_bank_agent")
        
        # Visual agent for image/visual analysis
        if any(word in focus_text for word in ['image', 'visual', 'picture', 'appearance', 'look']):
            suggested.append("visual_agent")
        
        # Template-based suggestions
        if "retail operations" in selected_template.lower():
            suggested.extend(["ai_agent", "enhanced_statistical_agent"])
        elif "supply chain" in selected_template.lower():
            suggested.extend(["enhanced_statistical_agent", "memory_bank_agent"])
        elif "customer experience" in selected_template.lower():
            suggested.extend(["ai_agent", "embedding_agent"])
        elif "financial performance" in selected_template.lower():
            suggested.extend(["ai_agent", "enhanced_statistical_agent"])
        elif "data quality" in selected_template.lower():
            suggested.extend(["enhanced_statistical_agent"])
        elif "growth opportunities" in selected_template.lower():
            suggested.extend(["ai_agent", "embedding_agent", "memory_bank_agent"])
        elif "risk management" in selected_template.lower():
            suggested.extend(["ai_agent", "enhanced_statistical_agent", "memory_bank_agent"])
        
        # Remove duplicates and return
        return list(dict.fromkeys(suggested))  # Preserves order while removing duplicates

    @staticmethod
    def _get_role_specific_templates(profile):
        """Get role-specific quick templates for business focus areas"""
        templates = {
            "Business User": {
                "General Business Review": "Review all metrics for unusual patterns that could impact business performance",
                "Revenue Impact Analysis": "Focus on sales anomalies, pricing errors, and revenue-affecting patterns",
                "Customer Experience Issues": "Identify patterns that might indicate poor customer experience or satisfaction issues",
                "Operational Efficiency": "Find inefficiencies in processes, inventory, or resource utilization",
                "Risk Assessment": "Detect potential risks in financial metrics, compliance, or operational areas",
                "Large-Scale Product Catalog Audit": "Comprehensive anomaly detection for massive product datasets (10K+ products) from retail operations. Identify products with wrong descriptions, incorrect weights, inappropriate images, pricing errors, missing UPCs, inconsistent brand names, and other data quality issues that impact customer trust and operational efficiency. Optimized for high-volume retail product catalogs with advanced outlier detection algorithms.",
                "Data Quality Assessment": "Comprehensive data quality evaluation focused on retail product datasets. Identify inconsistencies, missing values, duplicate records, pricing anomalies, and data integrity issues that could impact business operations. Perfect for large product catalogs from retail operations.",
                "Retail Product Data Quality": "Identify anomalies in product information including incorrect descriptions, inconsistent weights, missing or inappropriate product images, pricing errors, and data quality issues across large retail product catalogs. Focus on detecting outliers that could impact customer experience, inventory management, and sales performance in retail environments."
            },
            "Data Analyst": {
                "Statistical Outliers": "Identify statistical outliers and data quality issues across all metrics",
                "Trend Analysis": "Analyze time-series patterns and detect trend anomalies",
                "Correlation Analysis": "Find unusual correlations and relationships between variables",
                "Data Quality Assessment": "Check for data integrity issues, missing patterns, and quality problems",
                "Predictive Insights": "Identify patterns that could predict future anomalies or trends"
            },
            "Technical Expert": {
                "System Performance": "Analyze system metrics, performance indicators, and technical anomalies",
                "Data Pipeline Issues": "Detect data processing errors, pipeline failures, or integration problems",
                "Security Patterns": "Identify potential security threats or unusual access patterns",
                "Infrastructure Monitoring": "Monitor system health, resource usage, and capacity issues",
                "Advanced Analytics": "Perform deep technical analysis using multiple algorithms and approaches"
            }
        }
        return templates.get(profile, templates["Business User"])

    @staticmethod
    def _suggest_agents_for_role_and_focus(profile, custom_prompt, selected_template):
        """Suggest agents based on user role and focus area"""
        suggested = []
        
        # Base suggestions by role
        if profile == "Business User":
            suggested = ["ai_agent", "enhanced_statistical_agent"]
        elif profile == "Data Analyst":
            suggested = ["statistical_agent", "enhanced_statistical_agent", "embedding_agent"]
        elif profile == "Technical Expert":
            suggested = ["enhanced_statistical_agent", "memory_bank_agent", "visual_agent"]
        
        # Add suggestions based on focus area keywords
        focus_text = (custom_prompt + " " + selected_template).lower()
        
        if any(keyword in focus_text for keyword in ["fraud", "security", "risk"]):
            suggested.extend(["ai_agent", "memory_bank_agent"])
        if any(keyword in focus_text for keyword in ["visual", "chart", "pattern"]):
            suggested.append("visual_agent")
        if any(keyword in focus_text for keyword in ["similarity", "cluster", "embedding"]):
            suggested.append("embedding_agent")
        if any(keyword in focus_text for keyword in ["trend", "time", "historical"]):
            suggested.append("memory_bank_agent")
        if any(keyword in focus_text for keyword in ["product", "retail", "catalog", "quality", "description", "pricing", "inventory"]):
            suggested.extend(["ai_agent", "enhanced_statistical_agent", "visual_agent"])
        if any(keyword in focus_text for keyword in ["data quality", "inconsistent", "missing", "incorrect", "duplicate", "integrity", "evaluation"]):
            suggested.extend(["statistical_agent", "enhanced_statistical_agent"])
        if any(keyword in focus_text for keyword in ["comprehensive"]):
            suggested.extend(["ai_agent", "enhanced_statistical_agent", "visual_agent"])
        
        # Remove duplicates and return
        return list(dict.fromkeys(suggested))

    @staticmethod
    def _get_agent_explanation_for_role(profile):
        """Get role-specific explanation for agent recommendations"""
        explanations = {
            "Business User": "These agents provide business-focused insights with clear explanations and actionable recommendations.",
            "Data Analyst": "These agents offer statistical rigor and detailed analytical capabilities for thorough data exploration.",
            "Technical Expert": "These agents provide advanced technical analysis and system-level insights for comprehensive monitoring."
        }
        return explanations.get(profile, explanations["Business User"])
    

class AnomalyDisplayComponent:
    """Advanced anomaly display with business context"""
    
    @staticmethod
    def display_agent_results(results: Dict, df: pd.DataFrame = None, user_profile: UserProfile = UserProfile.BUSINESS_USER):
        """Display comprehensive agent results"""
        if not results:
            st.warning("No analysis results to display")
            return
        
        # Display orchestrator summary
        if 'orchestrator_summary' in results:
            st.subheader("🤖 Agent Analysis Summary")
            st.info(results['orchestrator_summary'])
        
        # Display individual agent results
        for agent_name, agent_results in results.items():
            if agent_name == 'orchestrator_summary':
                continue
            
            AnomalyDisplayComponent._display_single_agent_results(
                agent_name, agent_results, df, user_profile
            )
    
    @staticmethod
    def _display_single_agent_results(agent_name: str, results: Dict, df: pd.DataFrame, user_profile: UserProfile):
        """Display results from a single agent with enhanced handling for large anomaly sets"""
        agent_display_name = agent_name.replace('_', ' ').title()
        
        with st.expander(f"📊 {agent_display_name} Results", expanded=True):
            if isinstance(results, dict):
                if 'error' in results:
                    st.error(f"❌ {agent_display_name} failed: {results['error']}")
                    return
                
                if 'anomalies' in results:
                    anomalies = results['anomalies']
                    if anomalies:
                        anomaly_count = len(anomalies)
                        st.write(f"**Found {anomaly_count} anomalies**")
                        
                        # Handle large anomaly sets differently
                        if anomaly_count > 20:
                            AnomalyDisplayComponent._display_large_anomaly_set(
                                anomalies, agent_name, df, user_profile
                            )
                        else:
                            # Display all anomalies for smaller sets
                            for i, anomaly in enumerate(anomalies, 1):
                                AnomalyDisplayComponent._display_single_anomaly(
                                    anomaly, i, agent_name, df, user_profile
                                )
                        
                        # For Statistical Agent, add a combined outliers table at the end
                        if agent_name == 'statistical_agent' and df is not None:
                            AnomalyDisplayComponent._display_statistical_agent_outliers_summary(anomalies, df)
                            
                    else:
                        st.success("✅ No anomalies detected - data appears normal")
                
                elif 'anomalies_detected' in results:
                    anomalies = results['anomalies_detected']
                    if anomalies:
                        anomaly_count = len(anomalies)
                        st.write(f"**Found {anomaly_count} anomalies**")
                        
                        # Handle large anomaly sets differently
                        if anomaly_count > 20:
                            AnomalyDisplayComponent._display_large_anomaly_set(
                                anomalies, agent_name, df, user_profile
                            )
                        else:
                            # Display all anomalies for smaller sets
                            for i, anomaly in enumerate(anomalies, 1):
                                AnomalyDisplayComponent._display_single_anomaly(
                                    anomaly, i, agent_name, df, user_profile
                                )
                                
                        # For Statistical Agent, add a combined outliers table at the end
                        if agent_name == 'statistical_agent' and df is not None:
                            AnomalyDisplayComponent._display_statistical_agent_outliers_summary(anomalies, df)
                            
                    else:
                        st.success("✅ No anomalies detected - data appears normal")
                
                elif 'analysis' in results:
                    st.write("**Analysis Results:**")
                    st.write(results['analysis'])
                
                elif 'summary' in results:
                    st.write("**Summary:**")
                    st.write(results['summary'])
                
                else:
                    st.json(results)
            else:
                st.write(str(results))
    
    @staticmethod
    def _display_single_anomaly(anomaly: Dict, anomaly_num: int, agent_name: str, 
                               df: pd.DataFrame, user_profile: UserProfile):
        """Display a single anomaly with rich context"""
        anomaly_id = f"{agent_name}_{anomaly_num}_{datetime.now().timestamp()}"
        
        # Create anomaly container
        with st.container():
            # Anomaly header
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"### 🚨 Anomaly {anomaly_num}")
                
                # Display anomaly details based on user profile
                if user_profile == UserProfile.BUSINESS_USER:
                    AnomalyDisplayComponent._display_business_friendly_anomaly(anomaly, df)
                elif user_profile == UserProfile.DATA_ANALYST:
                    AnomalyDisplayComponent._display_analytical_anomaly(anomaly, df)
                else:  # Technical Expert
                    AnomalyDisplayComponent._display_technical_anomaly(anomaly, df)
                
                # Data comparison if available
                if df is not None and 'index' in anomaly:
                    AnomalyDisplayComponent._display_data_comparison(anomaly, df)
            
            with col2:
                # Feedback buttons
                st.write("**Your Assessment:**")
                if st.button("👍 Confirm", key=f"confirm_{anomaly_id}"):
                    st.success("Confirmed!")
                    # Record feedback in session state
                    if 'feedback_history' not in st.session_state:
                        st.session_state.feedback_history = []
                    
                    feedback = {
                        'anomaly_id': anomaly_id,
                        'approved': True,
                        'anomaly': anomaly,
                        'agent': agent_name,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.feedback_history.append(feedback)
                
                if st.button("👎 Reject", key=f"reject_{anomaly_id}"):
                    st.warning("Rejected!")
                    # Record feedback in session state
                    if 'feedback_history' not in st.session_state:
                        st.session_state.feedback_history = []
                    
                    feedback = {
                        'anomaly_id': anomaly_id,
                        'approved': False,
                        'anomaly': anomaly,
                        'agent': agent_name,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.feedback_history.append(feedback)
                
                # Confidence indicator
                confidence = anomaly.get('confidence', 0.0)
                if confidence > 0.8:
                    st.markdown("🔴 **High Confidence**")
                elif confidence > 0.6:
                    st.markdown("🟡 **Medium Confidence**")
                else:
                    st.markdown("⚪ **Low Confidence**")
            
            st.divider()
    
    @staticmethod
    def _display_business_friendly_anomaly(anomaly: Dict, df: pd.DataFrame = None):
        """Display anomaly in business-friendly terms with rich context"""
        # Extract key information
        column = anomaly.get('column', 'unknown field')
        value = anomaly.get('value', 'N/A')
        z_score = anomaly.get('z_score', 0)
        confidence = anomaly.get('confidence', 0.0)
        
        # Generate human-readable explanation
        human_explanation = AnomalyDisplayComponent._generate_human_explanation(anomaly)
        st.markdown(f"**🔍 What we found:** {human_explanation}")
        
        # Show the specific problem with context
        if 'value' in anomaly and 'column' in anomaly:
            problem_context = AnomalyDisplayComponent._generate_problem_context(anomaly)
            st.markdown(f"**⚠️ The Issue:** {problem_context}")
        elif 'description' in anomaly:
            # For anomalies that don't have detailed structure, enhance the description
            enhanced_description = AnomalyDisplayComponent._enhance_anomaly_description(anomaly)
            st.markdown(f"**⚠️ What This Means:** {enhanced_description}")
        
        # Business impact assessment
        business_impact = AnomalyDisplayComponent._assess_business_impact(anomaly)
        st.markdown(f"**💼 Business Impact:** {business_impact}")
        
        # Show actual data evidence
        data_evidence = AnomalyDisplayComponent._generate_data_evidence(anomaly)
        if data_evidence:
            st.markdown("**📊 Data Evidence:**")
            st.markdown(data_evidence)
        
        # Recommended actions with specific context
        actions = AnomalyDisplayComponent._generate_contextual_actions(anomaly)
        if actions:
            st.markdown("**✅ Recommended Actions:**")
            for action in actions:
                st.markdown(f"• {action}")
        
        # --- New Section: Identified Outliers Table ---
        AnomalyDisplayComponent._display_outliers_table(anomaly, df)

    @staticmethod
    def _display_outliers_table(anomaly: Dict, df: pd.DataFrame = None):
        """Display outliers table for any anomaly - reusable across all display methods"""
        # Handle different anomaly data structures
        sample_indices = anomaly.get('sample_indices') or anomaly.get('indices')
        column = anomaly.get('column', None)
        
        # If no sample_indices found, check if anomaly has 'index' (for Statistical Agent)
        if not sample_indices and 'index' in anomaly:
            sample_indices = [anomaly['index']]
        
        if sample_indices and df is not None and column and column in df.columns:
            st.markdown("**🧪 Identified Outliers for Review:**")
            # Show a table of the outlier rows for the relevant column
            outlier_rows = df.loc[sample_indices]
            # Only show the relevant column and index for clarity
            display_df = outlier_rows[[column]].copy()
            display_df.reset_index(inplace=True)
            display_df.rename(columns={column: f"Outlier Value ({column})", 'index': 'Row Index'}, inplace=True)
            st.dataframe(display_df, use_container_width=True)
        elif sample_indices and df is not None:
            # If column is not available, show all columns for those indices
            st.markdown("**🧪 Identified Outliers for Review:**")
            outlier_rows = df.loc[sample_indices]
            st.dataframe(outlier_rows, use_container_width=True)
        elif sample_indices:
            # If no DataFrame, just show indices
            st.markdown("**🧪 Identified Outlier Indices:** " + ", ".join(str(idx) for idx in sample_indices))

    @staticmethod
    def _display_statistical_agent_outliers_summary(anomalies: List[Dict], df: pd.DataFrame):
        """Display a combined outliers table for Statistical Agent results"""
        if not anomalies or df is None:
            return
        
        # Collect all outlier indices from all anomalies
        all_outlier_indices = set()
        anomaly_details = []
        
        for anomaly in anomalies:
            indices = anomaly.get('indices', [])
            column = anomaly.get('column')
            method = anomaly.get('method', 'Statistical')
            
            if indices and column:
                all_outlier_indices.update(indices)
                anomaly_details.append({
                    'column': column,
                    'method': method,
                    'count': len(indices)
                })
        
        if all_outlier_indices:
            st.markdown("---")
            st.markdown("### 📋 Statistical Agent - Sample Outliers Summary")
            st.markdown(f"**Found {len(all_outlier_indices)} unique outlier rows across all analyzed columns:**")
            
            # Show a summary of which columns had outliers
            for detail in anomaly_details:
                st.markdown(f"• **{detail['column']}**: {detail['count']} outliers using {detail['method']} method")
            
            # Show a sample of the outlier rows
            sample_size = min(10, len(all_outlier_indices))
            sample_indices = list(all_outlier_indices)[:sample_size]
            
            if sample_indices:
                st.markdown(f"**🧪 Sample Outlier Data (showing {len(sample_indices)} of {len(all_outlier_indices)} total):**")
                
                outlier_rows = df.loc[sample_indices].copy()
                
                # Show key business columns first
                priority_columns = ['product_name', 'category', 'price', 'sales', 'quantity']
                display_columns = [col for col in priority_columns if col in df.columns]
                
                # Add flagged columns
                flagged_columns = [detail['column'] for detail in anomaly_details]
                for col in flagged_columns:
                    if col not in display_columns:
                        display_columns.append(col)
                
                # Show available columns
                if display_columns:
                    display_df = outlier_rows[display_columns].copy()
                    display_df.reset_index(inplace=True)
                    display_df.rename(columns={'index': 'Row Index'}, inplace=True)
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.dataframe(outlier_rows.iloc[:, :6], use_container_width=True)
                
                if len(all_outlier_indices) > sample_size:
                    st.caption(f"Note: Showing sample of {sample_size} outliers. {len(all_outlier_indices) - sample_size} more outliers detected.")

    @staticmethod
    def _enhance_anomaly_description(anomaly: Dict) -> str:
        """Enhance anomaly descriptions for better business understanding"""
        description = anomaly.get('description', '')
        anomaly_type = anomaly.get('type', '')
        count = anomaly.get('count', 0)
        confidence = anomaly.get('confidence', 0.0)
        
        if 'statistical outlier' in description.lower():
            # Try to extract specific details from the description
            import re
            field_match = re.search(r'in (\w+)', description)
            field = field_match.group(1) if field_match else 'unknown field'
            
            return f"""We found a product with unusual **{field}** that stands out significantly from your typical product data. 
            
**Why this matters:** Products with unusual characteristics can indicate:
• Data quality issues that need fixing
• Exceptional products worth highlighting  
• Process changes that need documentation
• Pricing or inventory decisions that need review

**What makes it unusual:** This product's {field} value doesn't follow the normal patterns we see in your data, making it worth investigating."""
        
        elif 'data quality' in description.lower():
            return f"""**Data Quality Alert:** {description}
            
**Why this matters:** Poor data quality can lead to:
• Incorrect business decisions
• Customer dissatisfaction
• Operational inefficiencies
• Compliance issues

**Immediate action needed:** Review and clean this data to ensure accurate analysis."""
        
        elif 'multivariate' in description.lower() or 'isolation forest' in description.lower():
            products_text = "products" if count > 1 else "product"
            return f"""We found **{count} {products_text}** that have unusual combinations of features when analyzed together.
            
**Why this matters:** These products don't fit typical patterns and could represent:
• Premium or specialty products worth highlighting
• Data entry errors that need correction
• New product categories emerging
• Pricing opportunities or issues

**What makes them unusual:** While individual features might look normal, the combination of features is rare in your product catalog."""
        
        else:
            return f"""**Business Alert:** {description}
            
**Why this matters:** This anomaly indicates a deviation from normal business patterns that could represent opportunities or issues requiring attention.

**Next steps:** Investigation recommended to understand the business implications and determine appropriate action."""
    
    @staticmethod
    def _generate_human_explanation(anomaly: Dict) -> str:
        """Generate human-readable explanation of the anomaly"""
        column = anomaly.get('column', 'unknown field')
        value = anomaly.get('value', 'N/A')
        z_score = anomaly.get('z_score', 0)
        
        # Check if this is just a simple description fallback
        description = anomaly.get('description', '')
        if description and not any(key in anomaly for key in ['column', 'value', 'z_score']):
            # This is likely a simple anomaly from an agent, enhance it
            return AnomalyDisplayComponent._enhance_simple_description(description, anomaly)
        
        # Create field-specific explanations
        field_context = {
            'weight': 'product weight',
            'weightz': 'product weight', 
            'weightoz': 'product weight',
            'price': 'pricing',
            'sales': 'sales amount',
            'quantity': 'quantity sold',
            'inventory': 'inventory level',
            'cost': 'cost structure',
            'rating': 'customer rating',
            'discount': 'discount percentage'
        }
        
        field_type = 'data point'
        for key, desc in field_context.items():
            if key.lower() in column.lower():
                field_type = desc
                break
        
        # Determine severity based on z-score
        if abs(z_score) > 4:
            severity = "extremely unusual"
            probability = "less than 1 in 10,000"
        elif abs(z_score) > 3:
            severity = "very unusual"
            probability = "less than 3 in 1,000"
        elif abs(z_score) > 2:
            severity = "notably unusual"
            probability = "less than 5 in 100"
        else:
            severity = "somewhat unusual"
            probability = "outside normal ranges"
        
        # Direction of anomaly
        direction = "much higher than" if z_score > 0 else "much lower than"
        
        # Create richer explanation
        base_explanation = f"This product has a {severity} {field_type} of **{value}**, which is {direction} what we typically see."
        context_explanation = f" This type of deviation occurs in {probability} similar products, making it worth investigating."
        
        return base_explanation + context_explanation
    
    @staticmethod
    def _enhance_simple_description(description: str, anomaly: Dict) -> str:
        """Enhance simple anomaly descriptions to be more business-friendly"""
        description_lower = description.lower()
        
        if 'statistical outlier' in description_lower:
            # Extract field name
            import re
            field_match = re.search(r'in (\w+)', description)
            field = field_match.group(1) if field_match else 'unknown field'
            
            # Get confidence if available
            confidence = anomaly.get('confidence', 0.0)
            confidence_text = f" (confidence: {confidence:.0%})" if confidence > 0 else ""
            
            # Map field to business context
            field_context = {
                'weight': 'product weight',
                'weightz': 'product weight', 
                'weightoz': 'product weight',
                'price': 'pricing',
                'sales': 'sales amount',
                'quantity': 'quantity sold',
                'inventory': 'inventory level',
                'cost': 'cost structure',
                'rating': 'customer rating',
                'discount': 'discount percentage'
            }
            
            field_type = 'data point'
            for key, desc in field_context.items():
                if key.lower() in field.lower():
                    field_type = desc
                    break
            
            return f"We found a product with unusual {field_type} that doesn't match typical patterns in your data{confidence_text}. This kind of anomaly can indicate data quality issues, exceptional products, or business opportunities worth investigating."
        
        elif 'data quality' in description_lower:
            return f"**Data Quality Alert:** {description}. This type of issue can affect the reliability of your analysis and should be addressed to ensure accurate business insights."
        
        elif 'multivariate' in description_lower or 'isolation forest' in description_lower:
            count = anomaly.get('count', 'some')
            return f"We detected {count} products that have unusual combinations of features when compared together. These products don't fit the typical patterns and may represent exceptional cases, data errors, or emerging trends worth investigating."
        
        else:
            # Generic enhancement
            return f"**Business Alert:** {description}. This anomaly represents a deviation from normal patterns that could impact your business operations or reveal hidden opportunities."
    
    
    @staticmethod
    def _generate_problem_context(anomaly: Dict) -> str:
        """Generate specific problem context with data comparison"""
        column = anomaly.get('column', '').lower()
        value = anomaly.get('value', 'N/A')
        z_score = anomaly.get('z_score', 0)
        
        # Get additional context if available
        confidence = anomaly.get('confidence', 0.0)
        confidence_text = f" (confidence: {confidence:.0%})" if confidence > 0 else ""
        
        if 'weight' in column:
            if z_score > 0:
                severity = "significantly heavier" if z_score > 3 else "heavier"
                context = f"This product weighs **{value} oz**, which is {severity} than normal products{confidence_text}."
                business_impact = " This could indicate:\n• **Packaging Error:** Wrong packaging or extra components included\n• **Product Classification:** May be categorized incorrectly (e.g., family size vs. individual)\n• **Data Entry Mistake:** Weight may have been entered incorrectly\n• **Manufacturing Variation:** Could be a legitimate product variant"
                recommendation = f"\n\n**💡 Immediate Action:** Verify this product's actual weight and check if it matches manufacturer specifications. Review similar products in the same category."
            else:
                severity = "much lighter" if z_score < -3 else "lighter"
                context = f"This product weighs only **{value} oz**, which is {severity} than expected{confidence_text}."
                business_impact = " This might suggest:\n• **Incomplete Information:** Missing product details or components\n• **Sample/Travel Size:** Could be a smaller variant not properly categorized\n• **Data Quality Issue:** Weight measurement or entry error\n• **Cost Optimization:** Manufacturer may have reduced product size"
                recommendation = f"\n\n**💡 Immediate Action:** Confirm this is the correct weight and verify product category. Check if this affects pricing per unit."
            
            return context + business_impact + recommendation
        
        elif 'price' in column:
            if z_score > 0:
                severity = "much higher" if z_score > 3 else "higher"
                context = f"This product is priced at **${value}**, which is {severity} than similar items{confidence_text}."
                business_impact = " This could indicate:\n• **Premium Product:** Luxury or premium variant with enhanced features\n• **Pricing Error:** Accidentally set too high, potentially losing customers\n• **Market Positioning:** Intentional premium pricing strategy\n• **Competitor Analysis Needed:** Price may be out of line with market"
                recommendation = f"\n\n**💡 Immediate Action:** Review competitor pricing and analyze if this premium is justified by features or quality."
            else:
                severity = "unusually low" if z_score < -3 else "lower"
                context = f"This product is priced at only **${value}**, which is {severity}{confidence_text}."
                business_impact = " This might suggest:\n• **Clearance/Liquidation:** Product being sold at cost or below\n• **Pricing Error:** May be priced too low, losing profit margin\n• **Loss Leader Strategy:** Intentionally low price to drive traffic\n• **Cost Reduction:** Manufacturer may have reduced production costs"
                recommendation = f"\n\n**💡 Immediate Action:** Verify this price is intentional and analyze impact on profit margins."
            
            return context + business_impact + recommendation
        
        elif 'sales' in column:
            if z_score > 0:
                severity = "exceptionally high" if z_score > 3 else "high"
                context = f"This product generated **${value}** in sales, which is {severity}{confidence_text}."
                business_impact = " This could indicate:\n• **Viral Success:** Product has gained unexpected popularity\n• **Bulk Purchase:** Large order from single customer or event\n• **Seasonal Spike:** Product may be experiencing seasonal demand\n• **Marketing Success:** Promotional campaign may have been very effective\n• **Data Aggregation:** Sales may be incorrectly aggregated"
                recommendation = f"\n\n**💡 Immediate Action:** Investigate what drove this success and see if it can be replicated for other products."
            else:
                severity = "much lower" if z_score < -3 else "lower"
                context = f"This product only generated **${value}** in sales, which is {severity} than expected{confidence_text}."
                business_impact = " This might suggest:\n• **Poor Performance:** Product not resonating with customers\n• **Inventory Issues:** Stockouts preventing sales\n• **Competition:** Competitors may have better alternatives\n• **Marketing Gap:** Product may lack proper promotion\n• **Market Shift:** Customer preferences may have changed"
                recommendation = f"\n\n**💡 Immediate Action:** Analyze barriers to sales and consider product positioning or promotional strategies."
            
            return context + business_impact + recommendation
        
        elif 'inventory' in column:
            if z_score > 0:
                severity = "much higher" if z_score > 3 else "higher"
                context = f"This product has **{value}** units in inventory, which is {severity} than normal{confidence_text}."
                business_impact = " This could indicate:\n• **Overstocking:** Poor demand forecasting or overordering\n• **Slow Moving:** Product not selling as expected\n• **Seasonal Build-up:** Preparing for anticipated demand\n• **Supply Chain Issue:** Deliveries arrived faster than expected"
                recommendation = f"\n\n**💡 Immediate Action:** Review demand forecasting accuracy and consider promotional strategies to move excess inventory."
            else:
                severity = "critically low" if z_score < -3 else "low"
                context = f"This product has only **{value}** units in inventory, which is {severity}{confidence_text}."
                business_impact = " This might suggest:\n• **High Demand:** Product selling faster than expected\n• **Supply Chain Delay:** Shipments may be delayed\n• **Understocking:** Initial order quantity was too low\n• **Stockout Risk:** May soon be unable to fulfill orders"
                recommendation = f"\n\n**💡 Immediate Action:** Immediately reorder and check supplier lead times to prevent stockouts."
            
            return context + business_impact + recommendation
        
        else:
            # Generic context with z-score explanation
            severity_desc = "extreme" if abs(z_score) > 3 else "significant" if abs(z_score) > 2 else "notable"
            direction = "higher" if z_score > 0 else "lower"
            
            context = f"The value **{value}** for {column} is {abs(z_score):.1f} standard deviations {direction} than normal{confidence_text}."
            explanation = f" This represents a {severity_desc} deviation that occurs in less than {5 if abs(z_score) > 2 else 16}% of typical cases."
            
            business_impact = f"\n\n**Business Impact:** This anomaly indicates an unusual pattern that could represent:\n• Data quality issues requiring investigation\n• Exceptional business performance (positive or negative)\n• Process changes that need documentation\n• Opportunities for improvement or replication"
            
            recommendation = f"\n\n**💡 Recommended Action:** Investigate the root cause of this deviation and determine if it represents an issue to fix or a success to replicate."
            
            return context + explanation + business_impact + recommendation
    
    @staticmethod
    def _assess_business_impact(anomaly: Dict) -> str:
        """Assess business impact with specific reasoning"""
        column = anomaly.get('column', '').lower()
        z_score = anomaly.get('z_score', 0)
        confidence = anomaly.get('confidence', 0.0)
        
        impact_level = "Low"
        if confidence > 0.8 and abs(z_score) > 3:
            impact_level = "High"
        elif confidence > 0.6 and abs(z_score) > 2:
            impact_level = "Medium"
        
        if 'weight' in column:
            return f"**{impact_level} Impact** - Weight anomalies can affect shipping costs, product categorization, and customer expectations. May require product verification and potential corrections."
        
        elif 'price' in column:
            return f"**{impact_level} Impact** - Pricing anomalies directly affect revenue and profit margins. Could impact competitiveness and customer perception. Immediate review recommended."
        
        elif 'sales' in column:
            return f"**{impact_level} Impact** - Sales anomalies indicate performance outliers that could signal opportunities or problems. May affect inventory planning and marketing strategies."
        
        elif 'inventory' in column:
            return f"**{impact_level} Impact** - Inventory anomalies affect customer satisfaction and working capital. Could lead to stockouts or excess carrying costs."
        
        else:
            return f"**{impact_level} Impact** - This anomaly represents a significant deviation that warrants investigation to understand root causes and potential business implications."
    
    @staticmethod
    def _generate_data_evidence(anomaly: Dict) -> str:
        """Generate comprehensive data evidence explanation"""
        z_score = anomaly.get('z_score', 0)
        value = anomaly.get('value', 'N/A')
        column = anomaly.get('column', 'field')
        confidence = anomaly.get('confidence', 0.0)
        method = anomaly.get('method', '')
        count = anomaly.get('count', 0)
        anomaly_type = anomaly.get('type', '')
        
        # Check if this is a Statistical Agent anomaly with aggregate results
        if method and count > 0 and 'Outlier' in anomaly_type:
            evidence = f"**📊 Statistical Analysis ({method}):**\n"
            evidence += f"• **Detection Method:** {method}\n"
            evidence += f"• **Outliers Found:** {count} data points\n"
            evidence += f"• **Column Analyzed:** {column}\n"
            
            confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
            evidence += f"• **Detection Confidence:** {confidence:.0%} ({confidence_level})\n"
            
            # Method-specific details
            if 'Z-score' in anomaly_type:
                threshold = anomaly.get('threshold', 3.0)
                evidence += f"• **Threshold:** Values beyond {threshold} standard deviations\n"
                evidence += f"• **Meaning:** These points are statistically unusual\n"
            elif 'IQR' in anomaly_type:
                bounds = anomaly.get('bounds', {})
                if bounds:
                    evidence += f"• **Normal Range:** {bounds.get('lower', 'N/A'):.2f} to {bounds.get('upper', 'N/A'):.2f}\n"
                evidence += f"• **Meaning:** Values outside the typical interquartile range\n"
            
            evidence += f"\n**🎯 Business Context:**\n"
            evidence += f"• {count} products show unusual {column} values requiring review\n"
            evidence += f"• Statistical analysis identified these as significant outliers\n"
            evidence += f"• These findings warrant business investigation\n"
            
            return evidence
        
        # Original detailed z-score analysis for other agents
        elif abs(z_score) > 0:
            evidence = f"**📊 Statistical Analysis:**\n"
            evidence += f"• **Anomaly Score:** {abs(z_score):.2f} (higher scores = more unusual)\n"
            evidence += f"• **Actual Value:** {value}\n"
            
            # Probability interpretation
            if abs(z_score) > 4:
                probability = "less than 0.01%"
                rarity = "extremely rare"
            elif abs(z_score) > 3:
                probability = "less than 0.3%"
                rarity = "very rare"
            elif abs(z_score) > 2:
                probability = "less than 5%"
                rarity = "uncommon"
            else:
                probability = "less than 16%"
                rarity = "outside normal range"
            
            evidence += f"• **Probability:** This value appears in {probability} of normal data\n"
            evidence += f"• **Classification:** {rarity.title()} occurrence\n"
            
            # Confidence level
            confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
            evidence += f"• **Detection Confidence:** {confidence:.0%} ({confidence_level})\n"
            
            # Direction and magnitude
            direction = "above" if z_score > 0 else "below"
            magnitude_desc = "extremely" if abs(z_score) > 4 else "very" if abs(z_score) > 3 else "notably" if abs(z_score) > 2 else "somewhat"
            evidence += f"• **Deviation:** {magnitude_desc.title()} {direction} typical values\n"
            
            # Business context
            evidence += f"\n**🎯 Business Context:**\n"
            
            if abs(z_score) > 3:
                evidence += f"• This level of deviation typically requires immediate investigation\n"
                evidence += f"• Similar anomalies often reveal critical business insights\n"
            elif abs(z_score) > 2:
                evidence += f"• This deviation warrants business review and analysis\n"
                evidence += f"• May indicate process improvements or issues\n"
            else:
                evidence += f"• This deviation is worth monitoring for patterns\n"
                evidence += f"• Could be normal variation or emerging trend\n"
            
            # Field-specific evidence context
            field_context = AnomalyDisplayComponent._get_field_specific_evidence(column, value, z_score)
            if field_context:
                evidence += f"\n**📈 Field-Specific Analysis:**\n{field_context}"
            
            return evidence
        
        return "Limited statistical data available for this anomaly."
    
    @staticmethod
    def _get_field_specific_evidence(column: str, value: any, z_score: float) -> str:
        """Get field-specific evidence context"""
        column_lower = column.lower()
        
        if 'weight' in column_lower:
            if z_score > 0:
                return f"• Products this heavy may affect shipping costs and storage\n• Could indicate premium or bulk product variants\n• May require different handling procedures"
            else:
                return f"• Unusually light products may indicate travel/sample sizes\n• Could suggest cost optimization or material changes\n• May affect customer value perception"
        
        elif 'price' in column_lower:
            if z_score > 0:
                return f"• High prices may indicate premium market positioning\n• Could affect sales volume and market competitiveness\n• May require justification through enhanced features"
            else:
                return f"• Low prices may indicate promotional pricing or cost leadership\n• Could impact profit margins and brand perception\n• May drive higher sales volume"
        
        elif 'sales' in column_lower:
            if z_score > 0:
                return f"• High sales performance indicates strong market demand\n• Success factors should be analyzed and replicated\n• May require inventory planning adjustments"
            else:
                return f"• Low sales may indicate market challenges or opportunities\n• Requires analysis of pricing, positioning, or promotion\n• May need marketing strategy review"
        
        elif 'inventory' in column_lower:
            if z_score > 0:
                return f"• High inventory levels may indicate overstock situations\n• Could lead to carrying costs and potential markdowns\n• May require demand forecasting review"
            else:
                return f"• Low inventory levels may indicate strong demand or stockouts\n• Could lead to lost sales and customer satisfaction issues\n• May require supply chain review"
        
        return f"• Unusual values in {column} can indicate data quality issues or business opportunities\n• Further investigation recommended to understand root causes"
    
    @staticmethod
    def _generate_contextual_actions(anomaly: Dict) -> List[str]:
        """Generate context-specific recommended actions"""
        column = anomaly.get('column', '').lower()
        z_score = anomaly.get('z_score', 0)
        
        actions = []
        
        if 'weight' in column:
            actions.extend([
                "Verify product weight using physical measurement",
                "Check if packaging or product specifications have changed",
                "Review data entry processes for this product",
                "Compare with manufacturer specifications"
            ])
        
        elif 'price' in column:
            actions.extend([
                "Review pricing strategy and competitor analysis",
                "Verify no pricing errors in the system",
                "Check if special promotions or discounts apply",
                "Assess impact on profit margins and sales volume"
            ])
        
        elif 'sales' in column:
            if z_score > 0:
                actions.extend([
                    "Investigate what drove exceptional sales performance",
                    "Check if this can be replicated for other products",
                    "Ensure sufficient inventory to meet potential demand",
                    "Analyze customer feedback and reviews"
                ])
            else:
                actions.extend([
                    "Investigate barriers to sales (pricing, availability, marketing)",
                    "Review product positioning and promotion strategies",
                    "Check for negative customer feedback or reviews",
                    "Consider product discontinuation if consistently poor"
                ])
        
        elif 'inventory' in column:
            if z_score > 0:
                actions.extend([
                    "Review demand forecasting accuracy",
                    "Consider promotional strategies to move excess inventory",
                    "Evaluate storage costs and potential markdowns",
                    "Adjust future procurement plans"
                ])
            else:
                actions.extend([
                    "Immediately reorder to prevent stockouts",
                    "Check supplier lead times and delivery status",
                    "Consider temporary substitutes or alternatives",
                    "Review minimum stock level policies"
                ])
        
        else:
            actions.extend([
                "Investigate the root cause of this data anomaly",
                "Verify data accuracy and collection processes",
                "Check for system errors or data quality issues",
                "Monitor for similar patterns in related products"
            ])
        
        return actions
    
    @staticmethod
    def _display_analytical_anomaly(anomaly: Dict, df: pd.DataFrame = None):
        """Display anomaly with analytical details"""
        for key, value in anomaly.items():
            if key not in ['index', 'raw_data', 'recommended_actions']:
                st.write(f"**{key.title()}:** {value}")
        
        if 'recommended_actions' in anomaly:
            st.write("**Analysis Recommendations:**")
            for action in anomaly['recommended_actions']:
                st.write(f"• {action}")
        
        # Add outliers table for analytical view
        AnomalyDisplayComponent._display_outliers_table(anomaly, df)
    
    @staticmethod
    def _display_technical_anomaly(anomaly: Dict, df: pd.DataFrame = None):
        """Display anomaly with full technical details"""
        st.json(anomaly)
        
        # Add outliers table for technical view
        AnomalyDisplayComponent._display_outliers_table(anomaly, df)
    
    @staticmethod
    def _display_data_comparison(anomaly: Dict, df: pd.DataFrame):
        """Display enhanced data comparison for the anomaly"""
        if 'index' not in anomaly:
            return
        
        try:
            anomaly_index = anomaly['index']
            if anomaly_index >= len(df):
                return
            
            with st.expander("📊 Data Comparison - See the Evidence", expanded=False):
                # Get the anomalous record
                anomaly_record = df.iloc[anomaly_index]
                
                # Get relevant comparison data
                column = anomaly.get('column')
                if column and column in df.columns:
                    # Calculate statistics for the specific column
                    col_data = df[column].dropna()
                    mean_val = col_data.mean()
                    median_val = col_data.median()
                    std_val = col_data.std()
                    
                    # Business-friendly comparison
                    st.markdown("### 🎯 The Anomaly vs Normal Range")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label=f"This Product's {column}",
                            value=f"{anomaly_record[column]:.2f}" if pd.notnull(anomaly_record[column]) else "N/A",
                            delta=f"{(anomaly_record[column] - mean_val):.2f} vs average" if pd.notnull(anomaly_record[column]) else None
                        )
                    
                    with col2:
                        st.metric(
                            label=f"Typical {column}",
                            value=f"{mean_val:.2f}",
                            help="Average value for similar products"
                        )
                    
                    with col3:
                        percentage_diff = ((anomaly_record[column] - mean_val) / mean_val * 100) if mean_val != 0 else 0
                        st.metric(
                            label="Difference",
                            value=f"{percentage_diff:+.1f}%",
                            help="How much this product differs from typical products"
                        )
                    
                    # Range context
                    st.markdown("### 📏 Where This Fits in Our Data")
                    normal_range_low = mean_val - 2 * std_val
                    normal_range_high = mean_val + 2 * std_val
                    
                    if anomaly_record[column] > normal_range_high:
                        range_explanation = f"This value ({anomaly_record[column]:.2f}) is **above** the normal range of {normal_range_low:.2f} - {normal_range_high:.2f}"
                        st.error(f"🔴 {range_explanation}")
                    elif anomaly_record[column] < normal_range_low:
                        range_explanation = f"This value ({anomaly_record[column]:.2f}) is **below** the normal range of {normal_range_low:.2f} - {normal_range_high:.2f}"
                        st.error(f"🔴 {range_explanation}")
                    else:
                        range_explanation = f"This value ({anomaly_record[column]:.2f}) is within the normal range of {normal_range_low:.2f} - {normal_range_high:.2f}"
                        st.success(f"🟢 {range_explanation}")
                
                # Full product details comparison
                st.markdown("### 🏷️ Complete Product Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**This Unusual Product:**")
                    
                    # Create a clean product profile
                    product_info = {}
                    for col_name, value in anomaly_record.items():
                        # Format the display nicely
                        if pd.notnull(value):
                            if isinstance(value, float):
                                if col_name.lower() in ['price', 'cost', 'sales', 'revenue']:
                                    product_info[col_name] = f"${value:.2f}"
                                elif col_name.lower() in ['weight', 'weightoz']:
                                    product_info[col_name] = f"{value:.1f} oz"
                                elif col_name.lower() in ['rating']:
                                    product_info[col_name] = f"{value:.1f}/5.0"
                                else:
                                    product_info[col_name] = f"{value:.2f}"
                            else:
                                product_info[col_name] = str(value)
                        else:
                            product_info[col_name] = "N/A"
                    
                    # Display as formatted text instead of JSON
                    for key, value in product_info.items():
                        if key.lower() == anomaly.get('column', '').lower():
                            st.markdown(f"**{key}:** <span style='color: #ff6b6b; font-weight: bold;'>{value}</span> ⚠️", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{key}:** {value}")
                
                with col2:
                    st.markdown("**Typical Products (Sample):**")
                    
                    # Get a sample of normal products (excluding the anomaly)
                    normal_sample = df[df.index != anomaly_index].sample(min(3, len(df)-1))
                    
                    if len(normal_sample) > 0:
                        # Show key metrics for comparison
                        comparison_metrics = ['ProductName', 'BrandName', column] if column else ['ProductName', 'BrandName']
                        
                        for col_name in comparison_metrics:
                            if col_name in normal_sample.columns:
                                st.markdown(f"**{col_name} Examples:**")
                                for idx, row in normal_sample.iterrows():
                                    value = row[col_name]
                                    if pd.notnull(value):
                                        if isinstance(value, float):
                                            if col_name.lower() in ['price', 'cost', 'sales', 'revenue']:
                                                formatted_value = f"${value:.2f}"
                                            elif col_name.lower() in ['weight', 'weightoz']:
                                                formatted_value = f"{value:.1f} oz"
                                            else:
                                                formatted_value = f"{value:.2f}"
                                        else:
                                            formatted_value = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
                                        
                                        st.write(f"• {formatted_value}")
                                st.write("")
                
                # Statistical summary
                if column and column in df.columns:
                    st.markdown("### 📊 Statistical Context")
                    
                    # Create a simple distribution visualization
                    col_data = df[column].dropna()
                    
                    # Percentile ranking
                    percentile = (col_data < anomaly_record[column]).mean() * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Minimum", f"{col_data.min():.2f}")
                    with col2:
                        st.metric("Average", f"{col_data.mean():.2f}")
                    with col3:
                        st.metric("Maximum", f"{col_data.max():.2f}")
                    with col4:
                        st.metric("This Product's Rank", f"{percentile:.0f}th percentile")
                    
                    # Simple explanation
                    if percentile > 95:
                        st.success(f"📈 This product is in the top 5% for {column} - very high value")
                    elif percentile < 5:
                        st.warning(f"📉 This product is in the bottom 5% for {column} - very low value")
                    elif percentile > 75:
                        st.info(f"📊 This product is above average for {column}")
                    elif percentile < 25:
                        st.info(f"📊 This product is below average for {column}")
                    else:
                        st.info(f"📊 This product is around average for {column}")
                
        except Exception as e:
            st.error(f"Error displaying data comparison: {e}")
            # Fallback to simple display
            st.write("**Anomalous Record:**")
            try:
                anomaly_record = df.iloc[anomaly_index]
                st.json(anomaly_record.to_dict())
            except:
                st.write("Unable to display record details")
    
    @staticmethod
    def _display_large_anomaly_set(anomalies: List[Dict], agent_name: str, df: pd.DataFrame, user_profile: UserProfile):
        """Handle display of large anomaly sets with pagination and business insights"""
        anomaly_count = len(anomalies)
        
        # Show summary first
        st.info(f"🔍 **Large Anomaly Set Detected:** Found {anomaly_count:,} anomalies. Showing key insights and examples below.")
        
        # Business summary for large sets
        if user_profile == UserProfile.BUSINESS_USER:
            st.markdown("### 📈 **Business Impact Summary**")
            
            # Analyze patterns in the large set
            pattern_summary = AnomalyDisplayComponent._analyze_large_set_patterns(anomalies, df)
            if pattern_summary:
                st.markdown(pattern_summary)
            
            # Show specific examples with business context
            st.markdown("### 🎯 **Key Examples to Review**")
            
            # Get representative examples (first few, last few, and some random)
            sample_indices = AnomalyDisplayComponent._get_representative_samples(anomalies, sample_size=10)
            
            for idx in sample_indices:
                if idx < len(anomalies):
                    anomaly = anomalies[idx]
                    
                    # Add sample information to the anomaly
                    enhanced_anomaly = anomaly.copy()
                    enhanced_anomaly['sample_info'] = f"Example {idx + 1} of {anomaly_count:,} anomalies"
                    
                    AnomalyDisplayComponent._display_single_anomaly(
                        enhanced_anomaly, idx + 1, agent_name, df, user_profile
                    )
            
            # Add investigation tools
            st.markdown("### 🔧 **Investigation Tools**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📊 Export Anomaly List", key=f"export_{agent_name}"):
                    AnomalyDisplayComponent._export_anomaly_list(anomalies, agent_name)
            
            with col2:
                if st.button(f"📈 Show Distribution", key=f"dist_{agent_name}"):
                    AnomalyDisplayComponent._show_anomaly_distribution(anomalies, df)
            
            with col3:
                if st.button(f"🔍 Show More Examples", key=f"more_{agent_name}"):
                    st.session_state[f'show_more_{agent_name}'] = True
            
            # Show additional examples if requested
            if st.session_state.get(f'show_more_{agent_name}', False):
                st.markdown("### 📋 **Additional Examples**")
                
                # Pagination for additional examples
                examples_per_page = 5
                total_pages = min(10, (anomaly_count - 10) // examples_per_page + 1)  # Limit to 10 pages
                
                if total_pages > 0:
                    page = st.selectbox(
                        "Select page:", 
                        range(1, total_pages + 1), 
                        key=f"page_{agent_name}"
                    )
                    
                    start_idx = 10 + (page - 1) * examples_per_page  # Start after the first 10 examples
                    end_idx = min(start_idx + examples_per_page, anomaly_count)
                    
                    for i in range(start_idx, end_idx):
                        anomaly = anomalies[i]
                        enhanced_anomaly = anomaly.copy()
                        enhanced_anomaly['sample_info'] = f"Example {i + 1} of {anomaly_count:,} anomalies"
                        
                        AnomalyDisplayComponent._display_single_anomaly(
                            enhanced_anomaly, i + 1, agent_name, df, user_profile
                        )
        else:
            # For data analysts and technical experts, show more technical summary
            st.markdown("### 📊 **Statistical Summary**")
            
            # Show statistical breakdown
            stats_summary = AnomalyDisplayComponent._get_statistical_summary(anomalies)
            st.markdown(stats_summary)
            
            # Show sample of anomalies
            st.markdown("### 🔍 **Sample Anomalies**")
            sample_size = 15 if user_profile == UserProfile.DATA_ANALYST else 20
            
            for i, anomaly in enumerate(anomalies[:sample_size], 1):
                enhanced_anomaly = anomaly.copy()
                enhanced_anomaly['sample_info'] = f"Sample {i} of {anomaly_count:,} total"
                
                AnomalyDisplayComponent._display_single_anomaly(
                    enhanced_anomaly, i, agent_name, df, user_profile
                )
            
            if anomaly_count > sample_size:
                st.info(f"📝 **Note:** Showing {sample_size} of {anomaly_count:,} anomalies. Use the investigation tools above to explore more.")
    
    @staticmethod
    def _analyze_large_set_patterns(anomalies: List[Dict], df: pd.DataFrame) -> str:
        """Analyze patterns in large anomaly sets for business insights"""
        if not anomalies:
            return ""
        
        # Count anomalies by type, field, or confidence level
        patterns = {
            'fields_affected': {},
            'confidence_levels': {'high': 0, 'medium': 0, 'low': 0},
            'value_ranges': {}
        }
        
        for anomaly in anomalies:
            # Count by field
            field = anomaly.get('column', anomaly.get('field', 'Unknown'))
            patterns['fields_affected'][field] = patterns['fields_affected'].get(field, 0) + 1
            
            # Count by confidence level
            confidence = anomaly.get('confidence', 0.0)
            if confidence > 0.8:
                patterns['confidence_levels']['high'] += 1
            elif confidence > 0.6:
                patterns['confidence_levels']['medium'] += 1
            else:
                patterns['confidence_levels']['low'] += 1
        
        # Generate business summary
        summary_parts = []
        
        # Most affected fields
        if patterns['fields_affected']:
            top_fields = sorted(patterns['fields_affected'].items(), key=lambda x: x[1], reverse=True)[:3]
            field_summary = ", ".join([f"**{field}** ({count:,} anomalies)" for field, count in top_fields])
            summary_parts.append(f"🎯 **Most affected areas:** {field_summary}")
        
        # Confidence distribution
        high_conf = patterns['confidence_levels']['high']
        total = len(anomalies)
        if high_conf > 0:
            high_percentage = (high_conf / total) * 100
            summary_parts.append(f"🔴 **{high_conf:,} high-confidence anomalies** ({high_percentage:.1f}%) require immediate attention")
        
        # Business recommendations
        if total > 1000:
            summary_parts.append("⚠️ **Large-scale data quality issue detected** - consider systematic review of data collection processes")
        elif total > 100:
            summary_parts.append("📋 **Significant anomaly cluster** - may indicate process changes or data entry issues")
        
        return "\n\n".join(summary_parts) if summary_parts else "Data analysis in progress..."
    
    @staticmethod
    def _get_representative_samples(anomalies: List[Dict], sample_size: int = 10) -> List[int]:
        """Get representative sample indices from a large anomaly set"""
        import random
        
        total = len(anomalies)
        if total <= sample_size:
            return list(range(total))
        
        # Get a mix of first, last, and random samples
        samples = []
        
        # First few (to show immediate examples)
        samples.extend(range(min(3, total)))
        
        # Last few (to show if patterns change)
        if total > 6:
            samples.extend(range(max(total - 3, 3), total))
        
        # Random samples from the middle
        remaining_slots = sample_size - len(samples)
        if remaining_slots > 0 and total > 6:
            middle_start = 3
            middle_end = total - 3
            if middle_end > middle_start:
                middle_samples = random.sample(range(middle_start, middle_end), 
                                             min(remaining_slots, middle_end - middle_start))
                samples.extend(middle_samples)
        
        return sorted(list(set(samples)))
    
    @staticmethod
    def _export_anomaly_list(anomalies: List[Dict], agent_name: str):
        """Export anomaly list for further analysis"""
        try:
            import pandas as pd
            from io import StringIO
            
            # Convert anomalies to DataFrame
            export_data = []
            for i, anomaly in enumerate(anomalies, 1):
                row = {
                    'anomaly_id': i,
                    'agent': agent_name,
                    'column': anomaly.get('column', anomaly.get('field', 'Unknown')),
                    'value': anomaly.get('value', 'N/A'),
                    'confidence': anomaly.get('confidence', 0.0),
                    'z_score': anomaly.get('z_score', 0.0),
                    'description': anomaly.get('description', 'No description')
                }
                export_data.append(row)
            
            df_export = pd.DataFrame(export_data)
            
            # Create downloadable CSV
            csv_buffer = StringIO()
            df_export.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label=f"📥 Download {agent_name} Anomalies CSV",
                data=csv_data,
                file_name=f"{agent_name}_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_{agent_name}_{datetime.now().timestamp()}"
            )
            st.success("✅ Export ready! Click the download button above.")
            
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    @staticmethod
    def _show_anomaly_distribution(anomalies: List[Dict], df: pd.DataFrame):
        """Show distribution analysis of anomalies"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Analyze anomaly distribution by field
            field_counts = {}
            confidence_values = []
            
            for anomaly in anomalies:
                field = anomaly.get('column', anomaly.get('field', 'Unknown'))
                field_counts[field] = field_counts.get(field, 0) + 1
                
                confidence = anomaly.get('confidence', 0.0)
                if confidence > 0:
                    confidence_values.append(confidence)
            
            # Create distribution charts
            col1, col2 = st.columns(2)
            
            with col1:
                if field_counts:
                    # Field distribution chart
                    fields = list(field_counts.keys())[:10]  # Top 10 fields
                    counts = [field_counts[field] for field in fields]
                    
                    fig_fields = px.pie(
                        values=counts,
                        names=fields,
                        title="Anomalies by Field"
                    )
                    fig_fields.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_fields, use_container_width=True)
                else:
                    st.info("No anomalies detected by any agent")
            
            with col2:
                if confidence_values:
                    # Confidence distribution
                    fig_conf = px.histogram(
                        x=confidence_values,
                        nbins=20,
                        title="Confidence Score Distribution",
                        labels={'x': 'Confidence Score', 'y': 'Count'}
                    )
                    fig_conf.update_layout(height=400)
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📊 Summary Statistics")
            if field_counts:
                total_anomalies = sum(field_counts.values())
                top_field = max(field_counts.items(), key=lambda x: x[1])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Anomalies", total_anomalies)
                with col2:
                    st.metric("Fields Affected", len(field_counts))
                with col3:
                    st.metric("Top Affected Field", f"{top_field[0]} ({top_field[1]})")
            
            if confidence_values:
                avg_confidence = sum(confidence_values) / len(confidence_values)
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
                
        except Exception as e:
            st.error(f"Distribution analysis failed: {str(e)}")
            st.write("Unable to generate distribution charts")
    
    @staticmethod
    def _get_statistical_summary(anomalies: List[Dict]) -> str:
        """Get statistical summary for large anomaly sets"""
        if not anomalies:
            return "No anomalies to analyze"
        
        # Analyze confidence distribution
        confidences = [a.get('confidence', a.get('confidence_score', 0.0)) for a in anomalies]
        high_conf = sum(1 for c in confidences if c > 0.8)
        medium_conf = sum(1 for c in confidences if 0.5 < c <= 0.8)
        low_conf = sum(1 for c in confidences if c <= 0.5)
        
        # Analyze by type
        types = {}
        for anomaly in anomalies:
            anom_type = anomaly.get('type', anomaly.get('anomaly_type', 'Unknown'))
            types[anom_type] = types.get(anom_type, 0) + 1
        
        summary = f"""
**Statistical Breakdown:**
- **Total Anomalies:** {len(anomalies):,}
- **High Confidence (>80%):** {high_conf:,} ({high_conf/len(anomalies)*100:.1f}%)
- **Medium Confidence (50-80%):** {medium_conf:,}
- **Low Confidence (<50%):** {low_conf:,} ({low_conf/len(anomalies)*100:.1f}%)

**Top Anomaly Types:**
"""
        
        # Add top 5 types
        for anom_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / len(anomalies)) * 100
            summary += f"- **{anom_type}:** {count:,} ({percentage:.1f}%)\n"
        
        return summary


class VisualizationComponent:
    """Component for data visualization and charting"""
    
    @staticmethod
    def display_analysis_charts(results: Dict):
        """Display analysis results with charts and visualizations"""
        if not results:
            return
        
        st.subheader("📊 Analysis Visualizations")
        
        # Create tabs for different chart types
        chart_tabs = st.tabs(["📈 Anomaly Overview", "🔍 Agent Performance", "📊 Data Insights"])
        
        with chart_tabs[0]:
            VisualizationComponent._display_anomaly_overview_charts(results)
        
        with chart_tabs[1]:
            VisualizationComponent._display_agent_performance_charts(results)
        
        with chart_tabs[2]:
            VisualizationComponent._display_data_insight_charts(results)
    
    @staticmethod
    def _display_anomaly_overview_charts(results: Dict):
        """Display overview charts of anomalies found"""
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Count anomalies by agent
        agent_anomaly_counts = {}
        total_anomalies = 0
        
        for agent_name, agent_results in results.items():
            if agent_name == 'orchestrator_summary':
                continue
            
            if isinstance(agent_results, dict):
                anomaly_count = 0
                if 'anomalies' in agent_results and isinstance(agent_results['anomalies'], list):
                    anomaly_count = len(agent_results['anomalies'])
                elif 'anomalies_detected' in agent_results and isinstance(agent_results['anomalies_detected'], list):
                    anomaly_count = len(agent_results['anomalies_detected'])
                
                agent_anomaly_counts[agent_name.replace('_', ' ').title()] = anomaly_count
                total_anomalies += anomaly_count
        
        if agent_anomaly_counts:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of anomalies by agent
                if total_anomalies > 0:
                    fig_pie = px.pie(
                        values=list(agent_anomaly_counts.values()),
                        names=list(agent_anomaly_counts.keys()),
                        title="Anomalies Found by Agent"
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No anomalies detected by any agent")
            
            with col2:
                # Bar chart of anomalies by agent
                fig_bar = px.bar(
                    x=list(agent_anomaly_counts.keys()),
                    y=list(agent_anomaly_counts.values()),
                    title="Anomaly Count by Agent",
                    labels={'x': 'Agent', 'y': 'Number of Anomalies'}
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No anomaly data available for visualization")
    
    @staticmethod
    def _display_agent_performance_charts(results: Dict):
        """Display agent performance metrics"""
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Agent status overview
        agent_status = {}
        agent_execution_times = {}
        
        for agent_name, agent_results in results.items():
            if agent_name == 'orchestrator_summary':
                continue
            
            if isinstance(agent_results, dict):
                if 'error' in agent_results:
                    agent_status[agent_name.replace('_', ' ').title()] = 'Failed'
                elif 'status' in agent_results and agent_results['status'] == 'failed':
                    agent_status[agent_name.replace('_', ' ').title()] = 'Failed'
                else:
                    agent_status[agent_name.replace('_', ' ').title()] = 'Success'
                
                # Mock execution time (in a real implementation, you'd track this)
                agent_execution_times[agent_name.replace('_', ' ').title()] = np.random.uniform(0.5, 3.0)
        
        if agent_status:
            col1, col2 = st.columns(2)
            
            with col1:
                # Agent success/failure chart
                status_counts = {'Success': 0, 'Failed': 0}
                for status in agent_status.values():
                    status_counts[status] += 1
                
                fig_status = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    title="Agent Execution Status",
                    color_discrete_map={'Success': '#00CC96', 'Failed': '#EF553B'}
                )
                st.plotly_chart(fig_status, use_container_width=True)
            
            with col2:
                # Mock execution time chart
                fig_time = px.bar(
                    x=list(agent_execution_times.keys()),
                    y=list(agent_execution_times.values()),
                    title="Agent Execution Time (seconds)",
                    labels={'x': 'Agent', 'y': 'Execution Time (s)'}
                )
                fig_time.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No agent performance data available")
    
    @staticmethod
    def _display_data_insight_charts(results: Dict):
        """Display data insight visualizations"""
        st.info("Data insight charts would be displayed here based on the analysis results")
        
        # Show summary statistics if available
        if 'orchestrator_summary' in results:
            st.write("**Analysis Summary:**")
            st.write(results['orchestrator_summary'])
        
        # Add any additional insights from individual agents
        insights = []
        for agent_name, agent_results in results.items():
            if agent_name == 'orchestrator_summary':
                continue
            
            if isinstance(agent_results, dict):
                if 'summary' in agent_results:
                    insights.append(f"**{agent_name.replace('_', ' ').title()}:** {agent_results['summary']}")
                elif 'analysis' in agent_results:
                    insights.append(f"**{agent_name.replace('_', ' ').title()}:** {agent_results['analysis']}")
        
        if insights:
            st.write("**Key Insights:**")
            for insight in insights:
                st.write(f"• {insight}")


class FeedbackComponent:
    """Component for collecting user feedback on analysis results"""
    
    @staticmethod
    def display_feedback_section():
        """Display feedback collection interface"""
        st.subheader("💬 Feedback & Improvement")
        
        with st.expander("📝 Provide Feedback", expanded=False):
            feedback_type = st.selectbox(
                "What would you like to feedback on?",
                ["Analysis Quality", "UI/UX Experience", "Performance", "Feature Request", "Bug Report"]
            )
            
            rating = st.slider(
                "Overall satisfaction (1-5):",
                1, 5, 3,
                help="Rate your overall experience with the analysis"
            )
            
            feedback_text = st.text_area(
                "Your feedback:",
                placeholder="Tell us what worked well and what could be improved...",
                height=100
            )
            
            if st.button("Submit Feedback"):
                if feedback_text.strip():
                    # Store feedback in session state
                    if 'user_feedback' not in st.session_state:
                        st.session_state.user_feedback = []
                    
                    feedback_entry = {
                        'type': feedback_type,
                        'rating': rating,
                        'feedback': feedback_text,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.user_feedback.append(feedback_entry)
                    
                    st.success("Thank you for your feedback! It helps us improve the system.")
                else:
                    st.warning("Please provide some feedback text before submitting.")
        
        # Display feedback history if available
        if hasattr(st.session_state, 'user_feedback') and st.session_state.user_feedback:
            with st.expander("📊 Your Feedback History", expanded=False):
                for i, feedback in enumerate(st.session_state.user_feedback, 1):
                    st.write(f"**Feedback {i}** ({feedback['type']}) - Rating: {feedback['rating']}/5")
                    st.write(f"*{feedback['feedback']}*")
                    st.caption(f"Submitted: {feedback['timestamp']}")
                    st.divider()


class RoleDashboardComponent:
    """Role-specific dashboard component that adapts based on user profile"""
    
    @staticmethod
    def display_role_dashboard(profile: str, config: dict):
        """Display role-specific dashboard content"""
        st.subheader(f"📊 {profile} Dashboard")
        
        if profile == "Business User":
            RoleDashboardComponent._display_business_dashboard(config)
        elif profile == "Data Analyst":
            RoleDashboardComponent._display_analyst_dashboard(config)
        elif profile == "Technical Expert":
            RoleDashboardComponent._display_technical_dashboard(config)
        else:
            RoleDashboardComponent._display_default_dashboard(config)
    
    @staticmethod
    def _display_business_dashboard(config: dict):
        """Display business user focused dashboard"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            UIStyleManager.create_metric_card(
                "🎯 Business Focus", 
                "Revenue & Operations", 
                "Optimized for business insights",
                "success"
            )
        
        with col2:
            UIStyleManager.create_metric_card(
                "🤖 Recommended Agents", 
                "AI + Statistical", 
                "Business-context analysis",
                "metric"
            )
        
        with col3:
            UIStyleManager.create_metric_card(
                "📈 Analysis Type", 
                "Actionable Insights", 
                "Clear recommendations",
                "warning"
            )
        
        st.markdown("""
        **🎯 Business User Focus Areas:**
        - Revenue impact analysis and pricing optimization
        - Customer experience and satisfaction patterns  
        - Operational efficiency and cost reduction opportunities
        - Risk assessment and compliance monitoring
        - Data quality issues affecting business decisions
        """)
    
    @staticmethod
    def _display_analyst_dashboard(config: dict):
        """Display data analyst focused dashboard"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            UIStyleManager.create_metric_card(
                "📊 Analysis Focus", 
                "Statistical Methods", 
                "Data-driven insights",
                "success"
            )
        
        with col2:
            UIStyleManager.create_metric_card(
                "🔍 Detection Methods", 
                "Multi-Algorithm", 
                "Statistical + Embedding",
                "metric"
            )
        
        with col3:
            UIStyleManager.create_metric_card(
                "📈 Output Type", 
                "Detailed Analysis", 
                "Statistical evidence",
                "warning"
            )
        
        st.markdown("""
        **📊 Data Analyst Focus Areas:**
        - Statistical outlier detection and significance testing
        - Trend analysis and time-series anomaly detection
        - Correlation analysis and feature relationships
        - Data quality assessment and integrity checks
        - Predictive insights and pattern recognition
        """)
    
    @staticmethod
    def _display_technical_dashboard(config: dict):
        """Display technical expert focused dashboard"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            UIStyleManager.create_metric_card(
                "⚙️ Technical Focus", 
                "Advanced Analytics", 
                "Deep algorithmic analysis",
                "success"
            )
        
        with col2:
            UIStyleManager.create_metric_card(
                "🧠 Agent Types", 
                "Memory + Visual", 
                "Complex pattern detection",
                "metric"
            )
        
        with col3:
            UIStyleManager.create_metric_card(
                "🔧 Analysis Depth", 
                "Multi-Modal", 
                "Comprehensive coverage",
                "warning"
            )
        
        st.markdown("""
        **⚙️ Technical Expert Focus Areas:**
        - System performance monitoring and optimization
        - Data pipeline integrity and processing errors
        - Security pattern analysis and threat detection
        - Infrastructure monitoring and capacity planning
        - Advanced multi-modal anomaly detection
        """)
    
    @staticmethod
    def _display_default_dashboard(config: dict):
        """Display default dashboard for unknown profiles"""
        st.info("Select a user profile to see customized dashboard content")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            UIStyleManager.create_metric_card("🎯 Flexibility", "All Roles", "Adaptable interface", "metric")
        with col2:
            UIStyleManager.create_metric_card("🤖 Agents", "7 Available", "Multiple detection methods", "success")
        with col3:
            UIStyleManager.create_metric_card("📊 Analysis", "Comprehensive", "Business + Technical", "warning")