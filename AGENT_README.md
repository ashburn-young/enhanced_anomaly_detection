# Agent Architecture Documentation

## Overview

This document provides comprehensive documentation of the **Modular Anomaly Detection Agent System** - a sophisticated, enterprise-grade solution that combines **Semantic Kernel-powered AI agents** with **statistical analysis agents** to provide business-focused anomaly detection and insights.

---

## 🏗️ System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT APPLICATION                        │
│                      (app_enhanced.py)                          │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                 AGENT ORCHESTRATOR                              │
│                (agent_orchestrator.py)                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            Agent Loading & Coordination                 │    │
│  │  • Semantic Kernel Agent Discovery                      │    │
│  │  • Fallback Agent Registration                          │    │
│  │  • Async/Sync Method Harmonization                      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────┬───────────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼─────────┐ ┌───────▼─────────┐
│ SEMANTIC KERNEL │ │ SIMPLE AGENTS   │
│    AGENTS       │ │   (Fallback)    │
│ (AI-Powered)    │ │ (Rule-Based)    │
└─────────────────┘ └─────────────────┘
        │                   │
┌───────▼─────────┐ ┌───────▼─────────┐
│ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │ContextAgent │ │ │ │StatisticalA │ │
│ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │EmbeddingAgt │ │ │ │EnhancedStat │ │
│ └─────────────┘ │ │ └─────────────┘ │
│                 │ │ ┌─────────────┐ │
│                 │ │ │MemoryBankA  │ │
│                 │ │ └─────────────┘ │
│                 │ │ ┌─────────────┐ │
│                 │ │ │ VisualAgent │ │
│                 │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘
        │                   │
┌───────▼─────────┐ ┌───────▼─────────┐
│   AZURE OPENAI  │ │  STATISTICAL    │
│   / OPENAI API  │ │   LIBRARIES     │
│                 │ │ • NumPy         │
│ • GPT-4/4o      │ │ • SciPy         │
│ • Embeddings    │ │ • Scikit-learn  │
│ • Functions     │ │ • Pandas        │
└─────────────────┘ └─────────────────┘
```

---

## 🧠 Agent Types & Capabilities

### **Tier 1: Semantic Kernel AI Agents** 🤖

These agents leverage **Microsoft Semantic Kernel** framework with **Azure OpenAI** or **OpenAI API** for advanced AI-powered analysis.

#### **1. ContextAgent** (`agents/analysis/context_agent.py`)

**Purpose**: Provides business-focused contextual analysis and human-readable explanations

**Analysis Method**:
- **Statistical Foundation**: Performs Z-score anomaly detection (threshold: 2.5σ)
- **AI Enhancement**: Uses Semantic Kernel `@kernel_function` decorators to generate business explanations
- **Contextual Analysis**: Analyzes each anomaly in business context using AI
- **Root Cause Inference**: AI-powered analysis of potential causes

**Key Features**:
- **Business Explanations**: Converts statistical anomalies into business language
- **Impact Assessment**: Evaluates business impact of each anomaly
- **Actionable Recommendations**: Provides specific next steps
- **Custom Prompting**: Adapts analysis based on user's business context

**Analysis Flow**:
1. Detects statistical anomalies using Z-score > 2.5
2. For each anomaly, creates business context prompt
3. Calls Azure OpenAI via Semantic Kernel
4. Parses AI response into structured insights
5. Returns business-friendly explanations with recommendations

**Implementation Details**:
```python
async def analyze_async(self, data: pd.DataFrame, options: Dict[str, Any] = None) -> AnalysisResult:
    # Statistical detection
    anomalies = self._get_statistical_anomalies(data, options)
    
    # AI-powered contextual analysis
    for anomaly in anomalies:
        context = await self._generate_context_explanation(data, anomaly, options)
        anomaly.update(context)
    
    return AnalysisResult(...)
```

#### **2. EmbeddingAgent** (`agents/analysis/embedding_agent.py`)

**Purpose**: Semantic similarity analysis and multi-dimensional pattern detection

**Analysis Method**:
- **Text Semantic Analysis**: Uses AI embeddings to find semantically unusual text
- **Multi-dimensional Outliers**: Detects patterns across multiple numeric features
- **Similarity Clustering**: Groups similar records and identifies outliers
- **Cross-feature Analysis**: Analyzes relationships between different data dimensions

**Key Features**:
- **Text Anomaly Detection**: Finds semantically different text entries
- **Pattern Recognition**: Identifies complex multi-dimensional outliers
- **Embedding Caching**: Optimizes performance for repeated analysis
- **Fallback Simulation**: Provides deterministic embedding simulation when AI unavailable

**Analysis Flow**:
1. **Text Analysis**:
   - Identifies text columns with meaningful content
   - Generates embeddings using Semantic Kernel embedding service
   - Finds outliers using cosine similarity (threshold: 0.3)
   
2. **Pattern Analysis**:
   - Creates normalized feature vectors for all numeric columns
   - Uses distance-based outlier detection in multi-dimensional space
   - Identifies records that are distant from all others

**Implementation Details**:
```python
async def _detect_text_anomalies(self, data: pd.DataFrame, text_columns: List[str], options: Dict[str, Any]):
    # Generate embeddings for text
    embeddings = await self._get_text_embeddings(text_data)
    
    # Find semantic outliers
    outlier_indices = self._find_embedding_outliers(embeddings, threshold=0.3)
    
    return text_anomalies
```

### **Tier 2: Enhanced Analysis Agents** 🔬

#### **3. EnhancedAIAgent** (`enhanced_ai_agent.py`)

**Purpose**: Comprehensive AI-powered analysis with business intelligence

**Analysis Method**:
- **Intelligent Sampling**: Analyzes representative data samples (up to 100 records)
- **Business Context Integration**: Incorporates custom business rules and focus areas
- **Multi-method Detection**: Combines statistical and AI approaches
- **Natural Language Processing**: Generates detailed business narratives

**Key Features**:
- **Custom Prompting**: Responds to user-defined business focus areas
- **Azure OpenAI Integration**: Uses GPT-4/4o for advanced analysis
- **Business Intelligence**: Provides strategic insights and recommendations
- **Adaptive Analysis**: Adjusts analysis based on data characteristics

### **Tier 3: Statistical Foundation Agents** 📊

#### **4. StatisticalAgent** (`simple_agents.py`)

**Purpose**: Fundamental statistical anomaly detection

**Analysis Method**:
- **Z-Score Analysis**: Detects outliers using standard deviation (threshold: 3σ)
- **IQR Method**: Interquartile Range outlier detection (1.5 × IQR)
- **Multi-column Analysis**: Analyzes all numeric columns independently
- **Confidence Scoring**: Assigns confidence based on deviation magnitude

**Implementation**:
```python
def analyze_data(self, df: pd.DataFrame) -> dict:
    for col in numeric_columns:
        # Z-score method
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers_mask = z_scores > 3
        
        # IQR method
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        iqr_outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR))
```

#### **5. EnhancedStatisticalAgent** (`simple_agents.py`)

**Purpose**: Advanced multi-method statistical analysis

**Analysis Method**:
- **Multiple Detection Methods**: Z-score, IQR, Modified Z-score, Isolation Forest
- **Ensemble Scoring**: Combines results from multiple methods
- **Distribution Analysis**: Assesses normality and chooses appropriate methods
- **Robust Statistics**: Uses median-based methods for non-normal distributions

#### **6. MemoryBankAgent** (`simple_agents.py`)

**Purpose**: Historical pattern-based anomaly detection

**Analysis Method**:
- **Historical Comparison**: Compares current data against stored patterns
- **Pattern Memory**: Maintains memory of previously seen data patterns
- **Trend Analysis**: Detects deviations from historical trends
- **Adaptive Learning**: Updates patterns based on new data

#### **7. VisualAgent** (`simple_agents.py`)

**Purpose**: Visual pattern analysis and chart-based detection

**Analysis Method**:
- **Distribution Analysis**: Analyzes data distributions for anomalies
- **Visual Pattern Recognition**: Identifies patterns visible in charts
- **Correlation Analysis**: Detects unusual correlations between variables
- **Trend Detection**: Identifies abnormal trends and seasonality

---

## 🔄 Agent Orchestration System

### **AgentOrchestrator** (`agent_orchestrator.py`)

The orchestrator is the central coordination system that manages all agents and provides a unified interface.

#### **Agent Loading Strategy**

1. **Semantic Kernel Priority**: Attempts to load SK agents first
2. **Fallback Registration**: Loads simple agents as backup
3. **Conflict Resolution**: SK agents take priority over simple agents
4. **Dynamic Discovery**: Automatically discovers available agents

```python
def _load_agents(self):
    # Try Semantic Kernel agents first
    kernel = get_kernel()
    if kernel:
        agent_manager = AgentManager()
        sk_agents = agent_manager.list_agents()
    
    # Load fallback agents
    fallback_agents = {
        'statistical_agent': StatisticalAgent(),
        'enhanced_statistical_agent': EnhancedStatisticalAgent(),
        'ai_agent': EnhancedAIAgent()
    }
    
    # SK agents override fallback agents
    self.agents.update(fallback_agents)
    self.agents.update(sk_agents)
```

#### **Universal Agent Runner**

The orchestrator provides a universal interface that handles different agent types:

```python
async def _run_agent_universal(self, agent_name: str, df: pd.DataFrame, options: dict):
    agent = self.agents[agent_name]
    
    # Semantic Kernel agents (async)
    if hasattr(agent, 'analyze_async'):
        result = await agent.analyze_async(df, options)
        return self._convert_sk_result(result)
    
    # Simple agents (sync)
    elif hasattr(agent, 'detect_anomalies'):
        result = agent.detect_anomalies(df)
        return self._limit_results(result, options)
    
    # AI agents (custom interface)
    elif hasattr(agent, 'analyze_data'):
        return agent.analyze_data(df, options.get('custom_prompt'))
```

---

## 📊 Analysis Methods Deep Dive

### **Statistical Methods**

#### **Z-Score Analysis**
- **Formula**: `z = (x - μ) / σ`
- **Threshold**: Typically 2.5 to 3.0 standard deviations
- **Use Case**: Detecting outliers in normally distributed data
- **Advantages**: Simple, fast, interpretable
- **Limitations**: Assumes normal distribution, sensitive to extreme outliers

#### **Interquartile Range (IQR)**
- **Formula**: `outlier if x < Q1 - 1.5×IQR or x > Q3 + 1.5×IQR`
- **Use Case**: Robust outlier detection for any distribution
- **Advantages**: Distribution-free, robust to outliers
- **Limitations**: May miss subtle anomalies

#### **Modified Z-Score**
- **Formula**: `M = 0.6745 × (x - median) / MAD`
- **Use Case**: Robust alternative for non-normal data
- **Advantages**: Uses median instead of mean (more robust)

### **AI-Powered Methods**

#### **Semantic Kernel Functions**
```python
@kernel_function(
    description="Generate business explanation for anomaly",
    name="explain_anomaly"
)
def explain_anomaly(prompt: str) -> str:
    # AI processes the prompt and returns business explanation
    return ai_response
```

#### **Embedding-Based Similarity**
- **Method**: Cosine similarity in embedding space
- **Threshold**: Typically 0.3 for outlier detection
- **Use Case**: Finding semantically different text or multi-dimensional patterns

#### **Business Context Integration**
- **Custom Prompts**: User-defined business focus areas
- **Domain Knowledge**: Retail-specific analysis rules
- **Contextual Explanations**: AI generates human-readable insights

---

## 🎯 Business Focus Templates & Agent Selection

### **Smart Template System**

The system includes 7 predefined business focus templates that automatically suggest appropriate agents:

1. **🛒 Retail Operations** → `ai_agent`, `statistical_agent`
2. **📦 Supply Chain & Inventory** → `enhanced_statistical_agent`, `memory_bank_agent`
3. **⭐ Customer Experience** → `ai_agent`, `embedding_agent`
4. **💰 Financial Performance** → `statistical_agent`, `ai_agent`
5. **🔍 Data Quality Assurance** → `enhanced_statistical_agent`, `visual_agent`
6. **🚀 Growth Opportunities** → `ai_agent`, `embedding_agent`
7. **⚠️ Risk Management** → `enhanced_statistical_agent`, `memory_bank_agent`

### **Agent Suggestion Algorithm**

```python
def _suggest_agents_for_focus(custom_prompt: str, selected_template: str) -> List[str]:
    focus_text = (custom_prompt + " " + selected_template).lower()
    suggested = ["statistical_agent"]  # Always include baseline
    
    # Business context → AI agent
    if any(word in focus_text for word in ['fraud', 'behavior', 'trend', 'insight']):
        suggested.append("ai_agent")
    
    # Similarity analysis → Embedding agent
    if any(word in focus_text for word in ['similar', 'cluster', 'group']):
        suggested.append("embedding_agent")
    
    return suggested
```

---

## 🔧 Technical Implementation Details

### **Asynchronous Processing**

The system handles both synchronous and asynchronous agents seamlessly:

```python
# For Semantic Kernel agents (async)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(agent.analyze_async(df, options))
loop.close()

# For simple agents (sync)
result = agent.detect_anomalies(df)
```

### **Error Handling & Fallbacks**

```python
try:
    # Try Semantic Kernel agent
    result = await sk_agent.analyze_async(data, options)
except Exception as e:
    # Fall back to simple agent
    result = simple_agent.detect_anomalies(data)
```

### **Result Standardization**

All agents return results in a standardized format:

```python
{
    'analysis': 'Human-readable summary',
    'anomalies': [
        {
            'index': 4,
            'column': 'price',
            'value': 25.99,
            'confidence': 0.85,
            'z_score': 3.2,
            'description': 'Price significantly higher than average',
            'business_impact': 'May indicate pricing error or premium product',
            'recommended_actions': ['Verify pricing strategy', 'Check competitor prices']
        }
    ],
    'metadata': {'total_anomalies': 5, 'method': 'z_score_with_ai'}
}
```

---

## 🚀 Performance & Scalability

### **Optimization Strategies**

1. **Data Sampling**: Large datasets are sampled (max 100 records for AI analysis)
2. **Embedding Caching**: Embeddings are cached to avoid recomputation
3. **Async Processing**: Non-blocking AI calls using asyncio
4. **Fallback Mechanisms**: Fast statistical methods when AI unavailable

### **Memory Management**

```python
# Sample large datasets
sample_size = min(100, len(df))
df_sample = df.sample(n=sample_size) if len(df) > sample_size else df

# Limit results
max_anomalies = options.get('max_anomalies', 10)
result['anomalies'] = result['anomalies'][:max_anomalies]
```

---

## 🔒 Security & Configuration

### **Environment Variables**

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# OpenAI Fallback
OPENAI_API_KEY=your-openai-key
```

### **Kernel Configuration**

```python
# Core configuration handles multiple AI services
class SemanticKernelConfig:
    def __init__(self):
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Prioritize Azure OpenAI
        self.use_azure = bool(self.azure_openai_endpoint and self.azure_openai_api_key)
        self.use_openai = bool(self.openai_api_key and not self.use_azure)
```

---

## 📈 Usage Examples

### **Basic Analysis**

```python
from agent_orchestrator import AgentOrchestrator
import pandas as pd

# Load data
df = pd.read_csv('retail_data.csv')

# Initialize orchestrator
orchestrator = AgentOrchestrator()

# Configure analysis
options = {
    'selected_agents': ['statistical_agent', 'ai_agent', 'embedding_agent'],
    'custom_prompt': 'Find pricing anomalies that might indicate errors',
    'sensitivity': 0.7,
    'max_anomalies': 15
}

# Run analysis
results = orchestrator.analyze_with_agents(df, options)

# Access results
for agent_name, result in results.items():
    if agent_name != 'orchestrator_summary':
        print(f"{agent_name}: Found {len(result['anomalies'])} anomalies")
```

### **Semantic Kernel Agent Direct Usage**

```python
from agents.analysis import ContextAgent
from core import get_kernel

# Initialize with Semantic Kernel
kernel = get_kernel()
context_agent = ContextAgent(kernel)

# Run async analysis
import asyncio
result = asyncio.run(context_agent.analyze_async(df, {
    'business_context': 'retail_sales',
    'user_role': 'business_analyst'
}))

print(result.summary)
for anomaly in result.anomalies:
    print(f"Anomaly: {anomaly['business_explanation']}")
```

---

## 🧪 Testing & Validation

### **Agent Testing**

Each agent can be tested independently:

```python
# Test statistical agent
agent = StatisticalAgent()
result = agent.analyze_data(test_df)
assert len(result['anomalies']) > 0

# Test Semantic Kernel agent
sk_agent = ContextAgent(kernel)
result = asyncio.run(sk_agent.analyze_async(test_df))
assert result.confidence_score > 0.5
```

### **Integration Testing**

```python
# Test orchestrator integration
orchestrator = AgentOrchestrator()
available_agents = orchestrator.list_agents()
assert 'statistical_agent' in available_agents
assert 'embedding_agent' in available_agents
```

---

## 🔮 Future Enhancements

### **Planned Features**

1. **Custom Agent Development**: Framework for users to create domain-specific agents
2. **Agent Pipelines**: Chaining agents for complex multi-stage analysis
3. **Model Fine-tuning**: Custom embeddings for specific business domains
4. **Real-time Analysis**: Streaming anomaly detection for live data
5. **Federated Learning**: Privacy-preserving multi-tenant analysis

### **Architecture Evolution**

1. **Microservices**: Each agent as independent service
2. **Event-driven**: Pub/sub architecture for agent communication
3. **MLOps Integration**: Model versioning and deployment automation
4. **Multi-modal Analysis**: Support for images, time series, and graph data

---

## 📚 References & Dependencies

### **Core Dependencies**

- **Semantic Kernel**: Microsoft's AI orchestration framework
- **Azure OpenAI**: Enterprise AI services
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web application framework

### **Architecture Patterns**

- **Strategy Pattern**: Interchangeable analysis algorithms
- **Factory Pattern**: Dynamic agent creation
- **Observer Pattern**: Result aggregation and notification
- **Adapter Pattern**: Unified interface for different agent types

---

## 🧪 Empirical Analysis Results

### **Agent Performance Analysis** (June 2025)

After comprehensive testing and fixing critical bugs, here are the key findings about agent combinations and effectiveness:

#### **Individual Agent Performance**

| Agent | Anomalies Found | Strengths | Best Use Cases |
|-------|----------------|-----------|----------------|
| **statistical_agent** | 8 | Fast, reliable, mathematically sound | General-purpose, baseline detection |
| **ai_agent** | 3-4 | Business context, detailed explanations | Complex patterns, business insights |
| **visual_agent** | 3 | Distribution analysis, skewness detection | Data quality, pattern visualization |
| **embedding_agent** | 1-3 | Text similarity, semantic analysis | Text data, categorical anomalies |
| **enhanced_statistical_agent** | 1 | Multi-method ensemble | Robust detection, non-normal data |
| **context_agent** | 0 | Business context (when SK available) | Semantic Kernel environments |
| **memory_bank_agent** | 0-1 | Historical pattern matching | Time series, trend analysis |

#### **Agent Combination Results**

| Combination | Total Anomalies | Unique Types | Recommendation |
|-------------|----------------|--------------|----------------|
| **All Agents** | 18 | 4 | ⚠️ More isn't always better |
| **Top 3 (Statistical + AI + Visual)** | 15 | 5 | ✅ **Best balance** |
| **Statistical + AI** | 11 | 4 | ✅ **Efficient combination** |
| **Statistical Only** | 9 | 1 | ✅ Fast baseline |
| **AI Only** | 7 | 4 | Good for business context |

#### **Key Insights: Why All Agents May Find Fewer *Effective* Anomalies**

1. **🎯 Deduplication Effects**: Multiple agents detect the same patterns
   - Statistical agents overlap significantly (correlation in detection)
   - Similar mathematical foundations lead to redundant findings

2. **🔄 Signal-to-Noise Ratio**: More agents ≠ better signal
   - Some agents are specialized for specific data types
   - Generic agents may add noise rather than value

3. **📊 Aggregation Dilution**: Results get averaged/combined conservatively
   - High-confidence findings from specialized agents get diluted
   - Conservative thresholds applied across all agents

4. **🎛️ Agent Specialization**: Different agents excel at different anomaly types
   - **Statistical agents**: Numerical outliers, distribution anomalies
   - **AI agents**: Business context, complex patterns, text analysis
   - **Visual agents**: Distribution shapes, correlation patterns

### **Best Practices for Agent Selection**

#### **For Different Dataset Types**

```python
# Numerical data with clear outliers
recommended_agents = ['statistical_agent', 'enhanced_statistical_agent']

# Business data requiring context
recommended_agents = ['statistical_agent', 'ai_agent']

# Mixed data with text and numbers
recommended_agents = ['statistical_agent', 'ai_agent', 'embedding_agent']

# Large datasets (performance focus)
recommended_agents = ['statistical_agent']  # Fast and reliable

# Complex business analysis
recommended_agents = ['statistical_agent', 'ai_agent', 'visual_agent']  # Best combination
```

#### **When to Use Specific Agents**

1. **Always Include**: `statistical_agent` (reliable baseline)
2. **For Business Insights**: `ai_agent` (when Azure OpenAI available)
3. **For Data Quality**: `visual_agent` + `enhanced_statistical_agent`
4. **For Text Data**: `embedding_agent` + `ai_agent`
5. **For Speed**: `statistical_agent` only

#### **Agent Combination Strategy**

```python
# Smart agent selection based on data characteristics
def select_optimal_agents(df, business_focus):
    agents = ['statistical_agent']  # Always include baseline
    
    # Add AI for business context
    if business_focus in ['retail', 'financial', 'customer']:
        agents.append('ai_agent')
    
    # Add visual for data quality focus
    if business_focus == 'data_quality':
        agents.extend(['visual_agent', 'enhanced_statistical_agent'])
    
    # Add embedding for text-heavy data
    text_cols = df.select_dtypes(include=['object']).shape[1]
    if text_cols > df.shape[1] * 0.3:  # >30% text columns
        agents.append('embedding_agent')
    
    return agents
```

### **Fixed Issues** ✅

1. **AI Agent**: Fixed null prompt bug - now working correctly
2. **Context Agent**: Added missing orchestrator handler - no longer "Unknown agent"
3. **Agent Selection**: Corrected parameter passing in orchestrator
4. **Error Handling**: Improved fallback mechanisms

### **Performance Recommendations**

1. **Use 2-3 agents maximum** for optimal balance of coverage and performance
2. **Start with statistical_agent + ai_agent** for most business use cases
3. **Add visual_agent** for comprehensive analysis of complex data
4. **Avoid all agents together** unless specifically needed for research/comparison
5. **Monitor agent overlap** and adjust combinations based on your specific data patterns

---
