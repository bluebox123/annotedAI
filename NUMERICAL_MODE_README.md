# 🧮 Numerical Mode - Advanced Probability & Statistics Solver

## Overview

The **Numerical Mode** is a sophisticated mathematical problem-solving system specifically designed for probability and statistics questions. It implements the hybrid architecture described in the main README.md, combining:

- 🧠 **PEFT-tuned reasoning**: Domain-specific mathematical problem understanding
- 📚 **Formula-based RAG**: Retrieval of relevant formulas and worked examples
- 🔢 **Step-by-step computation**: Detailed solution process with mathematical rigor
- 📊 **Comprehensive coverage**: Support for the full probability & statistics syllabus

## Key Features

### ✨ Problem Types Supported

1. **Descriptive Statistics**
   - Mean, median, mode calculations
   - Variance, standard deviation
   - Quartiles and percentiles
   - Skewness and kurtosis

2. **Probability Distributions**
   - Binomial, Poisson, Normal
   - Exponential, Gamma, Weibull
   - Hypergeometric, Negative Binomial
   - Beta, Log-Normal, Uniform

3. **Hypothesis Testing**
   - Z-tests and t-tests
   - Chi-square tests
   - F-tests and ANOVA
   - Power analysis

4. **Correlation & Regression**
   - Pearson correlation
   - Linear regression
   - Multiple regression
   - Coefficient of determination

5. **Reliability Theory**
   - System reliability (series/parallel)
   - MTBF calculations
   - Hazard rate analysis
   - Availability metrics

6. **Advanced Topics**
   - Confidence intervals
   - Sample size determination
   - Bayesian inference
   - Time series analysis
   - Survival analysis

### 🚀 Interface Features

#### 🔍 Problem Solver Tab
- **Text Input**: Natural language problem description
- **Guided Input**: Step-by-step parameter entry for specific problem types
- **Example Problems**: Pre-loaded problems from the dataset

#### 📖 Formula Library Tab
- Organized by mathematical topic
- LaTeX-formatted formulas
- Context and usage examples

#### 📈 Solution History Tab
- Track all solved problems
- Review previous solutions
- Export solution history

#### 🎯 Practice Problems Tab
- Problems from the comprehensive dataset
- Categorized by difficulty and topic
- Interactive learning experience

## Architecture

### Core Components

```
Numerical Mode
├── numerical_solver.py          # Main problem solver engine
├── numerical_rag_engine.py      # Formula and example retrieval
├── numerical_interface.py       # Streamlit user interface
└── comprehensive_probability_statistics_dataset.json  # Training data
```

### Problem-Solving Pipeline

1. **Problem Analysis**
   - Natural language processing to identify problem type
   - Parameter extraction using regex patterns
   - Context understanding

2. **Formula Retrieval**
   - Semantic search through formula database
   - Similar example identification
   - Context-aware formula selection

3. **Solution Generation**
   - Step-by-step mathematical computation
   - Error checking and validation
   - Confidence scoring

4. **Result Presentation**
   - Formatted mathematical output
   - Visualization when applicable
   - Similar examples for learning

## Dataset

The system uses a comprehensive dataset with **100+ problems** covering:

- **Basic Statistics**: 15 problems (mean, median, variance, etc.)
- **Probability Distributions**: 25 problems (binomial, normal, poisson, etc.)
- **Hypothesis Testing**: 20 problems (z-test, t-test, chi-square, etc.)
- **Correlation/Regression**: 10 problems (correlation, linear regression)
- **Reliability Theory**: 10 problems (MTBF, system reliability)
- **Advanced Topics**: 20+ problems (confidence intervals, ANOVA, etc.)

Each problem includes:
```json
{
  "id": 1,
  "topic": "Arithmetic Mean",
  "question": "Find the arithmetic mean of marks: 85, 92, 78, 96, 87, 91, 83",
  "data": [85, 92, 78, 96, 87, 91, 83],
  "solution": 87.43,
  "formula": "Mean = Σx/n"
}
```

## Usage Examples

### Example 1: Binomial Distribution
```
Input: "A coin is tossed 10 times. Find the probability of getting exactly 6 heads."

Output:
Problem Type: Binomial Distribution
Formula: P(X=x) = C(n,x) × p^x × (1-p)^(n-x)

Steps:
1. Identify parameters: n=10, p=0.5, x=6
2. Calculate combination: C(10,6) = 210
3. Apply formula: P(X=6) = 210 × 0.5^6 × 0.5^4
4. Final answer: 0.205078
```

### Example 2: Hypothesis Testing
```
Input: "Sample mean is 52, population std is 8, n=36. Test if μ=50 at α=0.05"

Output:
Problem Type: Z-test for Mean
Formula: z = (x̄ - μ)/(σ/√n)

Steps:
1. State hypotheses: H₀: μ=50, H₁: μ≠50
2. Calculate test statistic: z = (52-50)/(8/√36) = 1.5
3. Find critical value: ±1.96 for α=0.05
4. Decision: Fail to reject H₀ (|1.5| < 1.96)
```

## Installation & Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run main.py
   ```

3. **Select Numerical Mode**
   - Choose "Numerical Mode" from the sidebar
   - Start solving problems!

## Testing

Run the test suite to verify functionality:

```bash
python test_numerical_solver.py
```

This will test:
- Basic statistical calculations
- Distribution problems
- Hypothesis testing
- RAG engine functionality

## Technical Details

### Mathematical Libraries Used
- **NumPy**: Numerical computations
- **SciPy**: Statistical functions and distributions
- **SymPy**: Symbolic mathematics
- **Matplotlib/Plotly**: Visualizations

### AI/ML Components
- **Sentence Transformers**: Semantic similarity for problem matching
- **FAISS**: Vector similarity search (optional)
- **Pattern Recognition**: Problem type classification

### Confidence Scoring
The system provides confidence scores based on:
- Problem type recognition accuracy
- Parameter extraction completeness  
- Formula matching quality
- Similar example availability

## Limitations & Future Enhancements

### Current Limitations
- Limited to probability and statistics domain
- Requires numerical parameters in problems
- English language only
- No symbolic manipulation for complex derivations

### Planned Enhancements
- Support for calculus-based problems
- Multi-language support
- Interactive visualizations
- Step-by-step derivation explanations
- Integration with computer algebra systems

## Contributing

To add new problem types or enhance functionality:

1. **Add Problems**: Extend `comprehensive_probability_statistics_dataset.json`
2. **New Solvers**: Add methods to `numerical_solver.py`
3. **Formula Library**: Update formula definitions
4. **UI Enhancements**: Modify `numerical_interface.py`

## Support

For questions or issues with the Numerical Mode:
1. Check the test suite output
2. Review the formula library
3. Examine similar examples in the dataset
4. Verify input parameter format

---

**Note**: This mode operates independently of the existing text-based RAG system and PDF highlighting functionality. It's designed as a complementary tool for numerical problem-solving in probability and statistics. 