# 🚀 Modular Anomaly Detection System

A modern, agent-driven AI solution for retail anomaly detection with business-friendly explanations and human-in-the-loop decision making.

## 🏗️ Clean Modular Architecture

### Core Application Files
- **`app_enhanced.py`** - Main Streamlit application (START HERE)
- **`agent_orchestrator.py`** - Coordinates multiple AI agents
- **`business_logic.py`** - User profiles, business rules, and feedback systems
- **`ui_components.py`** - Modular UI components (welcome, sidebar, anomaly display)
- **`data_processing.py`** - Data loading, validation, and sample generation
- **`enhanced_ai_agent.py`** - Enhanced AI agent with Azure OpenAI integration
- **`simple_agents.py`** - Statistical and specialized agent implementations

### Configuration & Documentation
- **`requirements.txt`** - Python dependencies
- **`.env.example`** - Environment configuration template
- **`BUSINESS_FOCUS_GUIDE.md`** - Comprehensive user guide with examples
- **`QUICK_REFERENCE.md`** - Quick reference for business focus areas
- **`ENHANCED_EXPLANATIONS_SUMMARY.md`** - Technical documentation of enhancements

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment (Optional):**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure OpenAI credentials if available
   ```

3. **Run the Application:**
   ```bash
   streamlit run app_enhanced.py
   ```

4. **Access the Interface:**
   - Open http://localhost:8501 in your browser
   - Choose your role (Business User recommended)
   - Select a business focus template or write custom prompts
   - Let the system suggest optimal AI agents
   - Analyze your data with business-friendly explanations

## 🎯 Key Features

### ✅ **Business-Friendly Interface**
- Role-based UI (Business User, Data Analyst, Technical Expert)
- Pre-built business focus templates
- Smart agent selection based on business needs
- Human-readable anomaly explanations

### ✅ **Advanced AI Analysis**
- Multiple specialized AI agents
- Azure OpenAI integration for contextual insights
- Statistical outlier detection with detailed row-level data
- Multivariate pattern analysis
- Historical trend recognition
- **NEW: Enhanced outliers table** showing specific data points for all agents

### ✅ **Human-in-the-Loop**
- Confirm/reject anomaly decisions
- Rich business context for each anomaly
- Actionable recommendations
- Confidence scoring and evidence
- **NEW: Detailed outlier data tables** with row indices and values

### ✅ **Comprehensive Data Support**
- Upload your own CSV data
- Built-in sample retail dataset
- Data quality assessment
- Missing value handling

## 🎛️ Business Focus Templates

The system includes pre-built templates for common business scenarios:

- **🛒 Retail Operations** - Fraud detection, sales anomalies, pricing errors
- **📦 Supply Chain & Inventory** - Stock levels, demand patterns, supplier issues  
- **⭐ Customer Experience** - Ratings, returns, satisfaction patterns
- **💰 Financial Performance** - Pricing optimization, profit margins
- **🔍 Data Quality Assurance** - Missing data, inconsistencies, entry errors
- **🚀 Growth Opportunities** - High performers, trends, market gaps
- **⚠️ Risk Management** - Supplier risks, volatility, inefficiencies

## 🤖 AI Agent Portfolio

- **Statistical Agent** - Basic outlier detection with detailed row-level data, fast results
- **Enhanced Statistical Agent** - Complex patterns, multivariate analysis with sample data
- **AI Agent** - Business context, natural language insights  
- **Embedding Agent** - Similarity analysis, product clustering
- **Memory Bank Agent** - Historical pattern recognition
- **Visual Agent** - Image and visual anomaly detection

*All agents now provide detailed outlier tables showing specific data points and row indices for transparent analysis.*

## 📊 Sample Business Use Cases

### **Fraud Detection:**
```
Find sales anomalies that might indicate fraud, unusual customer behavior, 
or pricing errors. Focus on products with unexpected sales spikes or drops.
```

### **Inventory Optimization:**
```
Identify inventory anomalies including overstocking, understocking, and 
unusual demand patterns that could affect operations.
```

### **Customer Experience:**
```
Detect anomalies in customer ratings and purchase patterns that might 
indicate quality issues or emerging market trends.
```

## 🔧 Configuration Options

- **Detection Sensitivity:** 0.1 (conservative) to 1.0 (sensitive)
- **Max Anomalies:** Limit results for focused analysis
- **Agent Selection:** Choose optimal combination for your use case
- **Custom Prompts:** Write specific business focus areas

## 📈 Business Value

- **Faster Decision Making** - Business-friendly explanations enable quick action
- **Reduced False Positives** - Smart agent selection focuses on relevant anomalies
- **Actionable Insights** - Each anomaly includes specific next steps
- **Risk Mitigation** - Early detection of business issues and opportunities
- **Process Improvement** - Data quality issues and operational inefficiencies

## 🛠️ Technical Architecture

The system follows a clean, modular architecture:

```
app_enhanced.py (Main UI)
├── ui_components.py (Modular UI)
├── business_logic.py (Business Rules)
├── agent_orchestrator.py (Agent Coordination)
│   ├── enhanced_ai_agent.py (AI Analysis)
│   └── simple_agents.py (Statistical Analysis)
└── data_processing.py (Data Handling)
```

## 🔄 Recent Enhancements

### ✅ **Outliers Table Feature (Latest)**
- **Enhanced Statistical Agent Display**: Fixed issue where Statistical Agent only showed aggregate counts
- **Row-Level Data Access**: All agents now provide specific outlier row indices and values
- **Improved Transparency**: Users can see exactly which data points were flagged as anomalies
- **Evidence Generation**: Better data evidence for Statistical Agent anomalies with meaningful context

### ✅ **Azure Container Apps Deployment**
- **Live Production Environment**: Deployed and running on Azure Container Apps
- **Scalable Infrastructure**: Auto-scaling based on demand
- **High Availability**: Production-ready deployment with monitoring
- **Environment Protection**: Sensitive configuration secured in Azure

### ✅ **Privacy & Security**
- **Protected Configuration**: Azure OpenAI keys and sensitive data secured
- **Git Security**: .gitignore configured to protect secrets
- **Environment Templates**: .env.example provided for easy setup

## 🌐 Live Demo

The enhanced anomaly detection system is deployed and accessible at:
- **Production URL**: Available via Azure Container Apps
- **Features**: Full functionality including all AI agents and outliers tables
- **Performance**: Optimized for production workloads

## 📊 Sample Business Use Cases

### **Fraud Detection:**
```
Find sales anomalies that might indicate fraud, unusual customer behavior, 
or pricing errors. Focus on products with unexpected sales spikes or drops.
```

### **Inventory Optimization:**
```
Identify inventory anomalies including overstocking, understocking, and 
unusual demand patterns that could affect operations.
```

### **Customer Experience:**
```
Detect anomalies in customer ratings and purchase patterns that might 
indicate quality issues or emerging market trends.
```

## 📞 Support

- Check `BUSINESS_FOCUS_GUIDE.md` for detailed examples
- Review `QUICK_REFERENCE.md` for quick tips
- Examine `ENHANCED_EXPLANATIONS_SUMMARY.md` for technical details
