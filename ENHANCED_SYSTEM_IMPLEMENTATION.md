# Enhanced Numerical Problem Solver - Implementation Guide

## Overview

This document provides exact steps to implement an enhanced numerical problem solver that combines:
1. **Question Classification Model** - Identifies problem types and required formulas
2. **Formula Cross-Reference System** - Checks if concepts exist in extracted knowledge
3. **Parameter Extraction** - Uses LLM or trained model to extract and label numerical values
4. **Step-by-Step Solver** - Provides detailed solutions with explanations

## System Architecture

```
User Question â†’ Classification Model â†’ Formula Check â†’ Parameter Extraction â†’ Problem Solver â†’ Solution
```

## Implementation Steps

### Step 1: Create Question Classification Model

#### 1.1 Data Preparation
```python
# File: question_classifier_data_prep.py
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def prepare_classification_data():
    """Prepare training data from both datasets"""
    
    # Load datasets
    with open('numerical_questions_dataset.json', 'r') as f:
        numerical_data = json.load(f)
    
    with open('comprehensive_probability_statistics_dataset.json', 'r') as f:
        stats_data = json.load(f)
    
    # Extract questions and labels
    training_data = []
    
    # From numerical dataset
    for question in numerical_data['questions']:
        training_data.append({
            'question': question['question'],
            'type': question['type'],
            'subtype': question['subtype'],
            'formula': get_formula_for_type(question['type'], question['subtype'])
        })
    
    # From statistics dataset
    stats_questions = stats_data['probability_statistics_numerical_questions']
    for section, content in stats_questions.items():
        if isinstance(content, dict):
            for subsection, problems in content.items():
                if isinstance(problems, list):
                    for problem in problems:
                        training_data.append({
                            'question': problem['question'],
                            'type': map_stats_type(section, subsection),
                            'subtype': problem.get('topic', ''),
                            'formula': problem.get('formula', '')
                        })
    
    return training_data

def get_formula_for_type(q_type, subtype):
    """Map question types to formulas"""
    formula_map = {
        'basic_arithmetic': {
            'addition': 'a + b = c',
            'subtraction': 'a - b = c',
            'multiplication': 'a Ã— b = c',
            'division': 'a Ã· b = c'
        },
        'percentage': {
            'percentage_of_number': 'percentage = (part/whole) Ã— 100',
            'percentage_increase': 'new_value = original Ã— (1 + percentage/100)',
            'percentage_decrease': 'new_value = original Ã— (1 - percentage/100)'
        },
        'geometry': {
            'area_rectangle': 'Area = length Ã— width',
            'perimeter_rectangle': 'Perimeter = 2 Ã— (length + width)',
            'area_triangle': 'Area = (1/2) Ã— base Ã— height'
        }
        # Add more mappings...
    }
    return formula_map.get(q_type, {}).get(subtype, '')

def map_stats_type(section, subsection):
    """Map statistics sections to question types"""
    mapping = {
        'descriptive_statistics': 'descriptive_stats',
        'probability_distributions': 'probability_dist',
        'hypothesis_testing': 'hypothesis_test',
        'correlation_and_regression': 'correlation_regression'
    }
    return mapping.get(section, section)
```

#### 1.2 Train Classification Model
```python
# File: question_classifier.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

class QuestionClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.type_encoder = LabelEncoder()
        self.subtype_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, training_data):
        """Train the classification model"""
        # Prepare features and labels
        questions = [item['question'] for item in training_data]
        types = [item['type'] for item in training_data]
        subtypes = [item['subtype'] for item in training_data]
        
        # Vectorize questions
        X = self.vectorizer.fit_transform(questions)
        
        # Encode labels
        y_type = self.type_encoder.fit_transform(types)
        y_subtype = self.subtype_encoder.fit_transform(subtypes)
        
        # Train model for type classification
        self.model.fit(X, y_type)
        
        # Train subtype classifier (separate model for each type)
        self.subtype_models = {}
        for type_label in np.unique(y_type):
            type_mask = y_type == type_label
            if np.sum(type_mask) > 1:  # Need at least 2 samples
                subtype_model = RandomForestClassifier(n_estimators=50)
                subtype_model.fit(X[type_mask], y_subtype[type_mask])
                self.subtype_models[type_label] = subtype_model
        
        return self
    
    def predict(self, question):
        """Predict question type and subtype"""
        X = self.vectorizer.transform([question])
        
        # Predict type
        type_pred = self.model.predict(X)[0]
        type_name = self.type_encoder.inverse_transform([type_pred])[0]
        
        # Predict subtype
        subtype_name = 'general'
        if type_pred in self.subtype_models:
            subtype_pred = self.subtype_models[type_pred].predict(X)[0]
            subtype_name = self.subtype_encoder.inverse_transform([subtype_pred])[0]
        
        return {
            'type': type_name,
            'subtype': subtype_name,
            'confidence': self.model.predict_proba(X).max()
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'type_encoder': self.type_encoder,
            'subtype_encoder': self.subtype_encoder,
            'model': self.model,
            'subtype_models': self.subtype_models
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.type_encoder = model_data['type_encoder']
        self.subtype_encoder = model_data['subtype_encoder']
        self.model = model_data['model']
        self.subtype_models = model_data['subtype_models']
        return self
```

### Step 2: Create Formula Cross-Reference System

#### 2.1 Formula Knowledge Base
```python
# File: formula_knowledge_base.py
import json
import re
from typing import Dict, List, Set

class FormulaKnowledgeBase:
    def __init__(self):
        self.formulas = {}
        self.concepts = set()
        self.load_formulas()
    
    def load_formulas(self):
        """Load all formulas from datasets"""
        # Load from comprehensive dataset
        with open('comprehensive_probability_statistics_dataset.json', 'r') as f:
            stats_data = json.load(f)
        
        # Extract formulas
        self._extract_formulas_from_stats(stats_data)
        
        # Load from numerical dataset
        with open('numerical_questions_dataset.json', 'r') as f:
            numerical_data = json.load(f)
        
        self._extract_formulas_from_numerical(numerical_data)
    
    def _extract_formulas_from_stats(self, data):
        """Extract formulas from statistics dataset"""
        sections = data['probability_statistics_numerical_questions']
        
        for section_name, section_data in sections.items():
            if isinstance(section_data, dict):
                for subsection_name, problems in section_data.items():
                    if isinstance(problems, list):
                        for problem in problems:
                            if 'formula' in problem:
                                formula = problem['formula']
                                topic = problem.get('topic', '')
                                
                                self.formulas[topic] = {
                                    'formula': formula,
                                    'section': section_name,
                                    'subsection': subsection_name,
                                    'concepts': self._extract_concepts(formula)
                                }
                                
                                # Add concepts to global set
                                self.concepts.update(self._extract_concepts(formula))
    
    def _extract_concepts(self, formula: str) -> Set[str]:
        """Extract mathematical concepts from formula"""
        concepts = set()
        
        # Common mathematical concepts
        concept_patterns = {
            'mean': r'[Î¼Î¼]|mean|average',
            'variance': r'ÏƒÂ²|variance|var',
            'standard_deviation': r'Ïƒ|std|standard deviation',
            'probability': r'P\(|probability',
            'binomial': r'C\(|binomial|nCr',
            'poisson': r'Î»|lambda|poisson',
            'normal': r'Z|normal|gaussian',
            'correlation': r'r|correlation',
            'regression': r'slope|intercept|regression'
        }
        
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, formula, re.IGNORECASE):
                concepts.add(concept)
        
        return concepts
    
    def check_formula_availability(self, question_type: str, subtype: str) -> Dict:
        """Check if formula exists for given question type"""
        # Search for matching formulas
        matching_formulas = []
        
        for topic, formula_info in self.formulas.items():
            if (question_type.lower() in topic.lower() or 
                subtype.lower() in topic.lower() or
                question_type.lower() in formula_info['section'].lower()):
                matching_formulas.append(formula_info)
        
        if matching_formulas:
            return {
                'available': True,
                'formulas': matching_formulas,
                'best_match': matching_formulas[0]  # Could implement better matching
            }
        else:
            return {
                'available': False,
                'reason': 'No matching formulas found in knowledge base',
                'suggested_concepts': list(self.concepts)
            }
    
    def get_formula_by_concept(self, concept: str) -> List[Dict]:
        """Get formulas containing specific concept"""
        matching_formulas = []
        
        for topic, formula_info in self.formulas.items():
            if concept.lower() in [c.lower() for c in formula_info['concepts']]:
                matching_formulas.append(formula_info)
        
        return matching_formulas
```

### Step 3: Parameter Extraction System

#### 3.1 LLM-based Parameter Extractor
```python
# File: parameter_extractor.py
import openai
import re
import json
from typing import Dict, List, Any, Optional

class ParameterExtractor:
    def __init__(self, api_key: str = None):
        if api_key:
            openai.api_key = api_key
        self.extraction_prompt = self._create_extraction_prompt()
    
    def _create_extraction_prompt(self) -> str:
        """Create prompt for parameter extraction"""
        return """
        Extract numerical parameters from the following mathematical question.
        Return a JSON object with the following structure:
        {
            "numbers": [
                {"value": number, "label": "parameter_name", "unit": "unit_if_any"}
            ],
            "question_type": "detected_type",
            "operation": "required_operation"
        }
        
        Common parameter labels:
        - n: number of trials, sample size
        - p: probability of success
        - lambda: rate parameter
        - mean: average value
        - std_dev: standard deviation
        - alpha: significance level
        - x: specific value
        - data: list of numbers
        
        Question: {question}
        
        JSON Response:
        """
    
    def extract_parameters(self, question: str) -> Dict[str, Any]:
        """Extract parameters using LLM"""
        try:
            prompt = self.extraction_prompt.format(question=question)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a mathematical parameter extraction expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            # Fallback to rule-based extraction
            return self._rule_based_extraction(question)
    
    def _rule_based_extraction(self, question: str) -> Dict[str, Any]:
        """Fallback rule-based parameter extraction"""
        numbers = re.findall(r'-?\d+\.?\d*', question)
        parameters = []
        
        # Simple heuristics for parameter labeling
        for num in numbers:
            value = float(num)
            label = self._infer_parameter_label(question, num)
            parameters.append({
                "value": value,
                "label": label,
                "unit": ""
            })
        
        return {
            "numbers": parameters,
            "question_type": "unknown",
            "operation": "unknown"
        }
    
    def _infer_parameter_label(self, question: str, number: str) -> str:
        """Infer parameter label based on context"""
        question_lower = question.lower()
        
        # Context-based labeling
        if 'tossed' in question_lower or 'trials' in question_lower:
            return 'n'
        elif 'probability' in question_lower and float(number) <= 1:
            return 'p'
        elif 'mean' in question_lower or 'average' in question_lower:
            return 'mean'
        elif 'standard deviation' in question_lower or 'std' in question_lower:
            return 'std_dev'
        elif 'lambda' in question_lower or 'rate' in question_lower:
            return 'lambda'
        else:
            return 'x'
```

### Step 4: Enhanced Problem Solver

#### 4.1 Main Enhanced Solver
```python
# File: enhanced_numerical_solver.py
from question_classifier import QuestionClassifier
from formula_knowledge_base import FormulaKnowledgeBase
from parameter_extractor import ParameterExtractor
from numerical_solver import NumericalSolver
import json

class EnhancedNumericalSolver:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.formula_kb = FormulaKnowledgeBase()
        self.param_extractor = ParameterExtractor()
        self.solver = NumericalSolver()
        
        # Load trained models
        try:
            self.classifier.load_model('models/question_classifier.joblib')
        except FileNotFoundError:
            print("Warning: Question classifier not found. Please train the model first.")
    
    def solve_problem(self, question: str) -> Dict[str, Any]:
        """Main problem solving pipeline"""
        
        # Step 1: Classify question
        classification = self.classifier.predict(question)
        
        # Step 2: Check formula availability
        formula_check = self.formula_kb.check_formula_availability(
            classification['type'], 
            classification['subtype']
        )
        
        if not formula_check['available']:
            return {
                'status': 'unsolvable',
                'reason': 'Question cannot be solved with current knowledge',
                'classification': classification,
                'formula_check': formula_check,
                'suggested_concepts': formula_check.get('suggested_concepts', [])
            }
        
        # Step 3: Extract parameters
        parameters = self.param_extractor.extract_parameters(question)
        
        # Step 4: Solve problem
        try:
            solution = self.solver.solve_problem(question)
            
            return {
                'status': 'solved',
                'classification': classification,
                'formula_used': formula_check['best_match']['formula'],
                'parameters': parameters,
                'solution': solution,
                'confidence': classification['confidence']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'classification': classification,
                'formula_check': formula_check,
                'parameters': parameters
            }
    
    def get_solution_explanation(self, result: Dict[str, Any]) -> str:
        """Generate detailed explanation of the solution process"""
        if result['status'] == 'unsolvable':
            return f"""
            âŒ **Cannot Solve Problem**
            
            **Reason:** {result['reason']}
            
            **Question Type:** {result['classification']['type']}
            **Subtype:** {result['classification']['subtype']}
            
            **Available Concepts:** {', '.join(result['suggested_concepts'][:5])}
            """
        
        elif result['status'] == 'solved':
            return f"""
            âœ… **Problem Solved Successfully**
            
            **Question Type:** {result['classification']['type']} ({result['classification']['subtype']})
            **Confidence:** {result['classification']['confidence']:.1%}
            
            **Formula Used:** {result['formula_used']}
            
            **Extracted Parameters:**
            {self._format_parameters(result['parameters'])}
            
            **Solution:** {result['solution'].final_answer}
            """
        
        else:
            return f"""
            âš ï¸ **Error in Solving**
            
            **Error:** {result['error']}
            **Question Type:** {result['classification']['type']}
            """
    
    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for display"""
        if 'numbers' in parameters:
            param_list = []
            for param in parameters['numbers']:
                param_list.append(f"- {param['label']}: {param['value']} {param['unit']}")
            return '\n'.join(param_list)
        return "No parameters extracted"
```

### Step 5: Training Scripts

#### 5.1 Train Classification Model
```python
# File: train_classifier.py
from question_classifier_data_prep import prepare_classification_data
from question_classifier import QuestionClassifier
import joblib
import os

def train_question_classifier():
    """Train the question classification model"""
    print("Preparing training data...")
    training_data = prepare_classification_data()
    
    print(f"Training with {len(training_data)} examples...")
    classifier = QuestionClassifier()
    classifier.train(training_data)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    classifier.save_model('models/question_classifier.joblib')
    print("Model saved to models/question_classifier.joblib")
    
    # Test the model
    test_questions = [
        "What is 25 + 37?",
        "Find the mean of 1, 2, 3, 4, 5",
        "A coin is tossed 10 times. Find P(exactly 6 heads)",
        "Calculate 20% of 150"
    ]
    
    print("\nTesting model:")
    for question in test_questions:
        prediction = classifier.predict(question)
        print(f"Q: {question}")
        print(f"A: {prediction}\n")

if __name__ == "__main__":
    train_question_classifier()
```

### Step 6: Integration with Existing System

#### 6.1 Enhanced Interface
```python
# File: enhanced_numerical_interface.py
import streamlit as st
from enhanced_numerical_solver import EnhancedNumericalSolver
import json

class EnhancedNumericalInterface:
    def __init__(self):
        self.solver = EnhancedNumericalSolver()
    
    def render_enhanced_interface(self):
        """Render the enhanced interface"""
        st.header("íº€ Enhanced Numerical Problem Solver")
        
        st.markdown("""
        **AI-Powered Problem Classification & Solving**
        
        This enhanced system provides:
        - í·  **Smart Classification**: Automatically identifies problem types
        - í´ **Formula Matching**: Cross-references with knowledge base
        - í³Š **Parameter Extraction**: Uses AI to extract and label numbers
        - âœ… **Solvability Check**: Determines if problem can be solved
        - í³ **Step-by-Step Solutions**: Detailed explanations
        """)
        
        # Problem input
        problem_text = st.text_area(
            "Enter your mathematical problem:",
            height=120,
            placeholder="Example: A fair coin is tossed 8 times. What is the probability of getting exactly 5 heads?"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            solve_button = st.button("í´ Analyze & Solve", type="primary")
        
        with col2:
            classify_button = st.button("í¿·ï¸ Classify Only")
        
        with col3:
            st.markdown("*The system will automatically determine if your problem can be solved*")
        
        if solve_button and problem_text:
            with st.spinner("Analyzing problem..."):
                result = self.solver.solve_problem(problem_text)
                self.display_enhanced_solution(result, problem_text)
        
        if classify_button and problem_text:
            with st.spinner("Classifying problem..."):
                classification = self.solver.classifier.predict(problem_text)
                self.display_classification(classification)
    
    def display_enhanced_solution(self, result: Dict, problem: str):
        """Display enhanced solution with all steps"""
        
        # Status indicator
        if result['status'] == 'solved':
            st.success("âœ… Problem Successfully Solved!")
        elif result['status'] == 'unsolvable':
            st.error("âŒ Problem Cannot Be Solved")
        else:
            st.warning("âš ï¸ Error in Solving")
        
        # Create tabs for different aspects
        tab1, tab2, tab3, tab4 = st.tabs([
            "í³Š Analysis", 
            "í´ Classification", 
            "í³ Parameters", 
            "í²¡ Solution"
        ])
        
        with tab1:
            st.subheader("Problem Analysis")
            st.write(f"**Original Problem:** {problem}")
            st.write(f"**Status:** {result['status'].title()}")
            
            if result['status'] == 'solved':
                st.write(f"**Confidence:** {result['confidence']:.1%}")
            elif result['status'] == 'unsolvable':
                st.write(f"**Reason:** {result['reason']}")
        
        with tab2:
            st.subheader("Question Classification")
            if 'classification' in result:
                classification = result['classification']
                st.write(f"**Type:** {classification['type']}")
                st.write(f"**Subtype:** {classification['subtype']}")
                st.write(f"**Confidence:** {classification['confidence']:.1%}")
        
        with tab3:
            st.subheader("Extracted Parameters")
            if 'parameters' in result:
                parameters = result['parameters']
                if 'numbers' in parameters:
                    for param in parameters['numbers']:
                        st.write(f"**{param['label']}:** {param['value']} {param['unit']}")
                else:
                    st.write("No parameters extracted")
        
        with tab4:
            st.subheader("Solution Details")
            if result['status'] == 'solved':
                if 'formula_used' in result:
                    st.write(f"**Formula Used:** {result['formula_used']}")
                
                if 'solution' in result:
                    solution = result['solution']
                    st.write(f"**Final Answer:** {solution.final_answer}")
                    
                    if hasattr(solution, 'step_by_step_solution'):
                        st.write("**Steps:**")
                        for i, step in enumerate(solution.step_by_step_solution, 1):
                            st.write(f"{i}. {step}")
            else:
                st.write(self.solver.get_solution_explanation(result))
    
    def display_classification(self, classification: Dict):
        """Display only the classification results"""
        st.subheader("Question Classification Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Question Type", classification['type'])
            st.metric("Subtype", classification['subtype'])
        
        with col2:
            confidence_color = "green" if classification['confidence'] > 0.8 else "orange"
            st.metric(
                "Confidence", 
                f"{classification['confidence']:.1%}",
                delta=None
            )
        
        # Show formula availability
        formula_check = self.solver.formula_kb.check_formula_availability(
            classification['type'], 
            classification['subtype']
        )
        
        if formula_check['available']:
            st.success("âœ… Formula available in knowledge base")
            st.write(f"**Best Match:** {formula_check['best_match']['formula']}")
        else:
            st.error("âŒ No matching formula found")
            st.write(f"**Reason:** {formula_check['reason']}")

# Integration function
def render_enhanced_numerical_mode():
    """Main function to render enhanced numerical mode"""
    interface = EnhancedNumericalInterface()
    interface.render_enhanced_interface()
```

### Step 7: Update Main Application

#### 7.1 Update main.py
```python
# Add to main.py
from enhanced_numerical_interface import render_enhanced_numerical_mode

# In the mode selection section, add:
elif mode == "Enhanced Numerical":
    render_enhanced_numerical_mode()
```

### Step 8: Installation and Setup

#### 8.1 Requirements
```txt
# Add to requirements.txt
scikit-learn>=1.0.0
joblib>=1.0.0
openai>=0.27.0
torch>=1.9.0
transformers>=4.20.0
numpy>=1.21.0
pandas>=1.3.0
```

#### 8.2 Setup Script
```bash
# File: setup_enhanced_system.sh
#!/bin/bash

echo "Setting up Enhanced Numerical Solver..."

# Create directories
mkdir -p models
mkdir -p data

# Install requirements
pip install -r requirements.txt

# Train classification model
echo "Training question classifier..."
python train_classifier.py

# Optional: Train parameter extraction model
# echo "Training parameter extractor..."
# python train_parameter_extractor.py

echo "Setup complete!"
echo "Run the application with: streamlit run main.py"
```

## Usage Instructions

### 1. Initial Setup
```bash
# Clone and setup
git clone <repository>
cd annotedAI
chmod +x setup_enhanced_system.sh
./setup_enhanced_system.sh
```

### 2. Training Models
```bash
# Train question classifier
python train_classifier.py

# Optional: Train parameter extractor
python train_parameter_extractor.py
```

### 3. Running the System
```bash
# Start the application
streamlit run main.py

# Select "Enhanced Numerical" mode
```

### 4. Example Usage
1. Enter a mathematical problem
2. Click "Analyze & Solve"
3. View the classification, parameter extraction, and solution
4. If unsolvable, see suggested concepts and alternatives

## Testing the System

### Test Cases
```python
# File: test_enhanced_system.py
def test_enhanced_solver():
    solver = EnhancedNumericalSolver()
    
    test_cases = [
        "What is 25 + 37?",
        "Find the mean of 1, 2, 3, 4, 5",
        "A coin is tossed 10 times. Find P(exactly 6 heads)",
        "Calculate 20% of 150",
        "Find the area of a rectangle with length 5 and width 8",
        "Solve for x: 2x + 5 = 15"
    ]
    
    for question in test_cases:
        print(f"\nQuestion: {question}")
        result = solver.solve_problem(question)
        print(f"Status: {result['status']}")
        if result['status'] == 'solved':
            print(f"Answer: {result['solution'].final_answer}")
        print("-" * 50)

if __name__ == "__main__":
    test_enhanced_solver()
```

## Troubleshooting

### Common Issues
1. **Model not found**: Run training scripts first
2. **Low classification confidence**: Add more training data
3. **Parameter extraction errors**: Check LLM API key or use rule-based fallback
4. **Formula not found**: Add more formulas to knowledge base

### Performance Optimization
1. Cache classification results
2. Use smaller models for faster inference
3. Implement batch processing for multiple questions
4. Add model compression techniques

This implementation provides a comprehensive, AI-powered numerical problem solver that can classify questions, check formula availability, extract parameters, and provide detailed solutions with explanations.
