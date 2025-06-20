"""
Enhanced Modular Retail Anomaly Detection App
Clean architecture with separated concerns and modular components
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="Enhanced Anomaly Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular components
try:
    from agent_orchestrator import AgentOrchestrator
    from business_logic import UserProfile, BusinessContext, FeedbackManager
    from ui_components import (
        UIStyleManager, WelcomeScreenComponent, DataPreviewComponent,
        SidebarComponent, AnomalyDisplayComponent, VisualizationComponent,
        FeedbackComponent, RoleDashboardComponent
    )
    from data_processing import DataProcessor, SampleDataGenerator
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = UserProfile.BUSINESS_USER
    if 'feedback_manager' not in st.session_state:
        st.session_state.feedback_manager = FeedbackManager()
    if 'data_insights' not in st.session_state:
        st.session_state.data_insights = None
    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True

def load_orchestrator():
    """Load the agent orchestrator with error handling"""
    try:
        orchestrator = AgentOrchestrator()
        return orchestrator
    except Exception as e:
        st.error(f"Failed to load agent orchestrator: {e}")
        logger.error(f"Orchestrator loading error: {e}", exc_info=True)
        return None

def handle_file_upload():
    """Handle file upload with enhanced validation and error handling"""
    
    # Display file size info
    st.info("📁 **File Upload Guidelines:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Max Size:** 200MB")
    with col2:
        st.write("**Format:** CSV files only")
    with col3:
        st.write("**Encoding:** UTF-8 recommended")
    
    # Add debug info
    if st.checkbox("🔍 Show Upload Debug Info", value=False, help="Show technical details for troubleshooting"):
        st.code(f"""
Upload Configuration:
- Max File Size: 200MB
- Accepted Types: CSV
- Server CORS: Enabled
- XSRF Protection: Disabled
        """)
    
    # Test CSV download option
    with st.expander("🧪 Test Upload with Sample CSV", expanded=False):
        st.write("**Download a test CSV file to verify upload functionality:**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Generate Test CSV", type="secondary", help="Create a small test file for upload testing"):
                try:
                    test_csv_data, test_df = generate_test_csv()
                    st.download_button(
                        label="⬇️ Download Test CSV",
                        data=test_csv_data,
                        file_name="test_anomaly_data.csv",
                        mime="text/csv",
                        help="Download this file and try uploading it to test the upload functionality"
                    )
                    st.success(f"✅ Test CSV generated: {len(test_df)} rows, {len(test_df.columns)} columns")
                    
                    with st.expander("👀 Preview Test Data", expanded=False):
                        st.dataframe(test_df.head(10), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Failed to generate test CSV: {e}")
        
        with col2:
            st.info("**How to test:**")
            st.write("1. Click 'Generate Test CSV'")
            st.write("2. Download the test file")
            st.write("3. Upload it using the uploader below")
            st.write("4. Verify no errors occur")
    
    uploaded_file = st.file_uploader(
        "Upload your CSV data",
        type=['csv'],
        help="Upload a CSV file with your retail/business data (Max 200MB)",
        accept_multiple_files=False,
        key="csv_uploader"  # Add explicit key for better state management
    )
    
    if uploaded_file is not None:
        try:
            # Log the upload attempt
            logger.info(f"File upload started: {uploaded_file.name}, size: {uploaded_file.size} bytes")
            
            # Check file size (in bytes)
            file_size = uploaded_file.size
            file_size_mb = file_size / (1024 * 1024)
            
            st.info(f"📊 **File Info:** {uploaded_file.name} ({file_size_mb:.2f} MB)")
            
            if file_size_mb > 200:
                st.error("❌ File is too large (>200MB). Please use a smaller file or sample your data.")
                return None
            
            if file_size == 0:
                st.error("❌ File appears to be empty (0 bytes).")
                return None
            
            # Show progress for large files
            if file_size_mb > 10:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"Reading {uploaded_file.name} ({file_size_mb:.1f} MB)...")
                progress_bar.progress(25)
                
                # Read file with explicit encoding detection
                try:
                    # Try UTF-8 first
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                    progress_bar.progress(75)
                except UnicodeDecodeError:
                    # Fallback to other encodings
                    uploaded_file.seek(0)  # Reset file pointer
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                        st.warning("⚠️ File was read using latin-1 encoding. Consider saving as UTF-8 for better compatibility.")
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
                        st.warning("⚠️ File was read using cp1252 encoding. Consider saving as UTF-8 for better compatibility.")
                
                progress_bar.progress(100)
                status_text.text("File loaded successfully!")
                
                # Clear progress indicators after a moment
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
            else:
                # For smaller files, read directly
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                        st.warning("⚠️ File was read using latin-1 encoding.")
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
                        st.warning("⚠️ File was read using cp1252 encoding.")
            
            # Quick data checks before validation
            if df.empty:
                st.error("❌ The uploaded file is empty.")
                return None
                
            if len(df.columns) == 0:
                st.error("❌ The uploaded file has no columns.")
                return None
            
            # Log successful read
            logger.info(f"File read successfully: {len(df)} rows, {len(df.columns)} columns")
            
            # Display basic file info
            with st.expander("📋 File Information", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Records", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                    st.metric("Numeric Columns", numeric_cols)
                with col4:
                    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                    st.metric("Memory Usage", f"{memory_mb:.1f} MB")
                
                # Show column names and types
                st.write("**Column Information:**")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': [str(dtype) for dtype in df.dtypes],
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True, height=200)
            
            # Validate data
            with st.spinner("Validating data quality..."):
                validation_results = DataProcessor.validate_data(df)
            
            if not validation_results['is_valid']:
                st.error("❌ **Data validation failed:**")
                for error in validation_results['errors']:
                    st.error(f"• {error}")
                return None
            
            # Show warnings if any
            if validation_results['warnings']:
                with st.expander("⚠️ Data Quality Warnings", expanded=False):
                    for warning in validation_results['warnings']:
                        st.warning(f"• {warning}")
            
            # Show recommendations
            if validation_results['recommendations']:
                with st.expander("💡 Data Recommendations", expanded=False):
                    for rec in validation_results['recommendations']:
                        st.info(f"• {rec}")
            
            # Save to session state
            st.session_state.uploaded_data = df
            st.success(f"✅ **Data loaded successfully!** {len(df):,} records with {len(df.columns)} columns")
            
            return df
            
        except UnicodeDecodeError as e:
            st.error("❌ **Encoding Error:** The file encoding is not supported. Please save your CSV file with UTF-8 encoding.")
            st.info("💡 **Tip:** In Excel, use 'Save As' and select 'CSV UTF-8 (Comma delimited)'")
            logger.error(f"Unicode decode error: {e}")
            return None
            
        except pd.errors.EmptyDataError:
            st.error("❌ **Empty File:** The uploaded file appears to be empty or corrupted.")
            return None
            
        except pd.errors.ParserError as e:
            st.error("❌ **CSV Parse Error:** Unable to parse the CSV file. Please check the file format.")
            st.error(f"Details: {str(e)}")
            st.info("💡 **Tip:** Ensure your file is a valid CSV with consistent column separators")
            logger.error(f"CSV parser error: {e}")
            return None
            
        except MemoryError:
            st.error("❌ **Memory Error:** The file is too large to process. Please try a smaller file or sample your data.")
            return None
            
        except Exception as e:
            error_msg = str(e).lower()
            st.error(f"❌ **Upload Error:** {str(e)}")
            
            # Specific guidance for common errors
            if "400" in error_msg or "bad request" in error_msg:
                st.error("🔍 **HTTP 400 Error Detected**")
                st.info("💡 **Troubleshooting Steps:**")
                st.info("1. **File Format:** Ensure your file is a valid CSV with proper headers")
                st.info("2. **File Size:** Try with a smaller file (< 10MB) to test")
                st.info("3. **Browser:** Try refreshing the page or using a different browser")
                st.info("4. **File Content:** Check for special characters or unusual formatting")
                st.info("5. **Network:** If on a corporate network, check firewall/proxy settings")
                
                # Provide sample data option
                st.info("🎯 **Quick Test:** Try the sample data below to verify the system works")
                
            elif "cors" in error_msg or "origin" in error_msg:
                st.error("🔍 **CORS Error Detected**")
                st.info("💡 **Solution:** Try refreshing the page. If the problem persists, contact support.")
                
            elif "timeout" in error_msg:
                st.error("🔍 **Timeout Error Detected**")
                st.info("💡 **Solution:** Your file may be too large or your connection too slow. Try a smaller file.")
                
            else:
                st.info("💡 **General Troubleshooting:**")
                st.info("• Check that your file is a valid CSV")
                st.info("• Ensure the file is not corrupted") 
                st.info("• Try with a smaller file to test")
                st.info("• Refresh the page and try again")
                st.info("• Contact support if the problem persists")
            
            logger.error(f"File upload error: {e}", exc_info=True)
            return None
    
    return None

def handle_sample_data():
    """Handle sample data loading with scenario selection"""
    col1, col2 = st.columns(2)
    
    with col1:
        scenario = st.selectbox(
            "Choose sample data scenario:",
            ["Enhanced Retail Data", "Normal Retail", "Fraud Detection", "Inventory Issues", "Pricing Anomalies"],
            help="Different scenarios to test various anomaly detection capabilities"
        )
    
    with col2:
        if st.button("🔄 Load Sample Data", type="secondary"):
            try:
                if scenario == "Enhanced Retail Data":
                    df = DataProcessor.load_sample_data()
                else:
                    scenarios = SampleDataGenerator.generate_retail_scenarios()
                    scenario_mapping = {
                        "Normal Retail": "normal_retail",
                        "Fraud Detection": "fraud_detection", 
                        "Inventory Issues": "inventory_management",
                        "Pricing Anomalies": "pricing_anomalies"
                    }
                    df = scenarios[scenario_mapping[scenario]]
                
                st.session_state.uploaded_data = df
                st.success(f"✅ Sample data loaded: {len(df)} records")
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to load sample data: {e}")
                return None
    
    return st.session_state.uploaded_data

def run_analysis(df: pd.DataFrame, config: dict, orchestrator: AgentOrchestrator):
    """Run the agent analysis with progress tracking"""
    
    # Create business context
    business_context = BusinessContext(
        industry="retail",
        focus_areas=config.get('custom_prompt', '').split(',') if config.get('custom_prompt') else [],
        custom_rules=config.get('custom_prompt', ''),
        sensitivity_level=config.get('sensitivity', 0.5),
        max_anomalies=config.get('max_anomalies', 10)
    )
    
    # Prepare analysis options
    analysis_options = {
        'selected_agents': config['selected_agents'],
        'sensitivity': config['sensitivity'],
        'max_anomalies': config['max_anomalies'],
        'custom_prompt': config.get('custom_prompt'),
        'user_profile': config.get('profile_enum', UserProfile.BUSINESS_USER)
    }
    
    # Debug logging to verify prompt flow
    logger.info(f"🔍 Analysis Configuration:")
    logger.info(f"   Selected agents: {config['selected_agents']}")
    logger.info(f"   User profile: {config.get('profile', 'Unknown')}")
    logger.info(f"   Custom prompt provided: {bool(config.get('custom_prompt'))}")
    if config.get('custom_prompt'):
        logger.info(f"   Custom prompt preview: {config.get('custom_prompt')[:150]}{'...' if len(config.get('custom_prompt', '')) > 150 else ''}")
    
    # Run analysis with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🤖 Initializing agents...")
        progress_bar.progress(20)
        
        status_text.text("📊 Analyzing data...")
        progress_bar.progress(50)
        
        # Run the actual analysis
        results = orchestrator.analyze_with_agents(df, analysis_options)
        
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Analysis failed: {e}")
        logger.error(f"Analysis error: {e}", exc_info=True)
        return None

def display_data_insights(df: pd.DataFrame):
    """Display comprehensive data insights"""
    with st.expander("📊 Data Intelligence", expanded=False):
        if st.session_state.data_insights is None:
            with st.spinner("Analyzing data characteristics..."):
                st.session_state.data_insights = DataProcessor.get_data_insights(df)
        
        insights = st.session_state.data_insights
        
        # Basic stats in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{insights['basic_stats']['total_records']:,}")
        with col2:
            st.metric("Columns", insights['basic_stats']['total_columns'])
        with col3:
            st.metric("Numeric Columns", len(insights['column_types']['numeric']))
        with col4:
            memory_mb = insights['basic_stats']['memory_usage_mb']
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        # Recommendations
        if insights['recommendations']:
            st.subheader("💡 Data Recommendations")
            for rec in insights['recommendations']:
                st.info(rec)
        
        # High correlations
        if insights['correlations'].get('high_correlations'):
            st.subheader("🔗 High Correlations Detected")
            for corr in insights['correlations']['high_correlations']:
                st.write(f"• **{corr['column1']}** ↔ **{corr['column2']}**: {corr['correlation']:.3f}")

def display_tutorial():
    """Display tutorial and help information"""
    st.markdown("""
    ## 📚 How to Use This System
    
    ### 🎯 Getting Started
    1. **Select your role** in the sidebar to customize the interface
    2. **Upload data** or use sample data to explore capabilities
    3. **Configure analysis** by selecting agents and setting parameters
    4. **Add custom prompts** to focus on specific business concerns
    5. **Review results** and provide feedback to improve accuracy
    
    ### 🤖 Understanding the Agents
    
    **Statistical Agent**: Finds numerical outliers using traditional statistical methods
    - Best for: Detecting extreme values and statistical anomalies
    - Outputs: Z-scores, outlier indices, statistical confidence
    
    **Enhanced Statistical Agent**: Advanced statistical analysis with context
    - Best for: More sophisticated pattern detection
    - Outputs: Multi-dimensional outliers, trend analysis
    
    **AI Agent**: Business-aware contextual analysis
    - Best for: Understanding business implications of anomalies
    - Outputs: Business explanations, impact assessment, recommendations
    
    **Embedding Agent**: Uses AI embeddings for similarity analysis
    - Best for: Finding unusual patterns in high-dimensional data
    - Outputs: Similarity scores, cluster analysis
    
    **Memory Bank Agent**: Learns from historical patterns
    - Best for: Detecting deviations from learned normal behavior
    - Outputs: Historical comparisons, trend analysis
    
    **Visual Agent**: Analyzes visual patterns in data
    - Best for: Finding visual anomalies in charts and patterns
    - Outputs: Visual pattern analysis, chart-based insights
    
    ### 💡 Pro Tips
    - Start with statistical and AI agents for comprehensive coverage
    - Use custom prompts to focus on specific business scenarios
    - Provide feedback on results to improve future analysis
    - Compare results across multiple agents for validation
    
    ### 🎯 Custom Prompts Examples
    - "Find sales anomalies that might indicate fraud"
    - "Detect inventory issues that could affect customer satisfaction"
    - "Identify pricing errors that could impact profitability"
    - "Look for customer behavior patterns that seem unusual"
    """)

def generate_test_csv():
    """Generate a test CSV file for upload testing"""
    import io
    
    # Create sample data
    test_data = {
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'sales': np.random.normal(1000, 200, 100),
        'profit': np.random.normal(150, 50, 100),
        'customers': np.random.poisson(50, 100),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    }
    
    df = pd.DataFrame(test_data)
    
    # Add some anomalies
    anomaly_indices = np.random.choice(len(df), 5, replace=False)
    df.loc[anomaly_indices, 'sales'] = df.loc[anomaly_indices, 'sales'] * 3  # Sales spikes
    df.loc[anomaly_indices[:2], 'profit'] = df.loc[anomaly_indices[:2], 'profit'] * -1  # Losses
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    return csv_data, df

def main():
    """Main application function"""
    
    # Apply minimal theme-compatible styling enhancements
    UIStyleManager.apply_minimal_enhancements()
    
    # Apply minimal theme-compatible styling
    st.markdown("""
    <style>
        /* Remove default margins and ensure proper spacing */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        
        /* Enhance readability */
        .stSelectbox > div > div {
            background-color: var(--secondary-background-color);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Create header
    UIStyleManager.create_header(
        "Enhanced Anomaly Detection System",
        "Modular AI-driven analysis with business intelligence"
    )
    
    # Load orchestrator
    orchestrator = load_orchestrator()
    if not orchestrator:
        st.error("Cannot proceed without agent orchestrator. Please check your setup.")
        return
    
    # Sidebar configuration
    config = SidebarComponent.display()
    st.session_state.user_profile = config.get('profile_enum', UserProfile.BUSINESS_USER)
    
    # Role-specific dashboard
    RoleDashboardComponent.display_role_dashboard(config['profile'], config)
    
    # Handle tutorial and help
    if config['show_tutorial']:
        display_tutorial()
        return
    
    if config['show_help']:
        st.info("💡 Use the sidebar to configure your analysis. Upload data or use sample data to get started!")
        return
    
    # Show welcome screen for new users
    if st.session_state.show_welcome and not st.session_state.uploaded_data:
        WelcomeScreenComponent.display()
        
        if st.button("🚀 Get Started", type="primary"):
            st.session_state.show_welcome = False
            st.rerun()
        return
    
    # Data loading section
    st.header("📁 Data Source")
    
    df = None
    if config['data_source'] == "Upload your data":
        df = handle_file_upload()
    else:
        df = handle_sample_data()
    
    # Display data overview
    if df is not None:
        DataPreviewComponent.display_data_overview(df, show_quality=True)
        display_data_insights(df)
        
        # Analysis section
        st.header("🤖 AI Agent Analysis")
        
        # Analysis configuration summary
        with st.expander("⚙️ Analysis Configuration", expanded=False):
            st.write(f"**Selected Agents:** {', '.join(config['selected_agents'])}")
            st.write(f"**Sensitivity:** {config['sensitivity']}")
            st.write(f"**Max Anomalies:** {config['max_anomalies']}")
            st.write(f"**User Profile:** {config['profile']}")
            if config['custom_prompt']:
                st.write(f"**Custom Focus:** {config['custom_prompt']}")
        
        # Run analysis button
        if len(config['selected_agents']) > 0:
            if st.button("🚀 Start Agent Analysis", type="primary", use_container_width=True):
                with st.container():
                    results = run_analysis(df, config, orchestrator)
                    if results:
                        st.session_state.analysis_results = results
                        st.success("✅ Analysis completed successfully!")
        else:
            st.warning("⚠️ Please select at least one agent to run the analysis.")
    
    # Display results
    if st.session_state.analysis_results:
        st.header("📊 Analysis Results")
        
        # Display results using modular components
        AnomalyDisplayComponent.display_agent_results(
            st.session_state.analysis_results,
            df,
            st.session_state.user_profile
        )
        
        # Visualizations
        st.subheader("📈 Analysis Visualizations")
        VisualizationComponent.display_analysis_charts(st.session_state.analysis_results)
        
        # Export results
        with st.expander("📤 Export Results", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Analysis Report"):
                    report_data = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'configuration': {
                            'agents': config['selected_agents'],
                            'sensitivity': config['sensitivity'],
                            'custom_prompt': config.get('custom_prompt', '')
                        },
                        'results': st.session_state.analysis_results
                    }
                    
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=json.dumps(report_data, indent=2),
                        file_name=f"anomaly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("Reset Analysis"):
                    st.session_state.analysis_results = None
                    st.session_state.data_insights = None
                    st.rerun()
    
    # Feedback section
    if st.session_state.feedback_history:
        st.header("📊 Feedback Analytics")
        FeedbackComponent.display_feedback_summary()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🤖 Enhanced Anomaly Detection System • Built with Modular Architecture • 
        Powered by Multiple AI Agents
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
