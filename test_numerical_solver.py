#!/usr/bin/env python3
"""
Test script for the numerical solver system.
Tests various probability and statistics problems to ensure the solver works correctly.
"""

from numerical_solver import NumericalSolver, ProblemSolution
from numerical_rag_engine import NumericalRAGEngine

def test_binomial_problem():
    """Test binomial distribution problem"""
    solver = NumericalSolver()
    problem = "A coin is tossed 10 times. Find the probability of getting exactly 6 heads."
    
    solution = solver.solve_problem(problem)
    print("=== Binomial Problem Test ===")
    print(f"Problem: {problem}")
    print(f"Problem Type: {solution.problem_type}")
    print(f"Formula Used: {solution.formula_used}")
    print(f"Final Answer: {solution.final_answer:.6f}")
    print(f"Confidence: {solution.confidence_score:.1%}")
    print("Steps:")
    for i, step in enumerate(solution.step_by_step_solution, 1):
        print(f"  {i}. {step}")
    print()

def test_normal_problem():
    """Test normal distribution problem"""
    solver = NumericalSolver()
    problem = "For normal distribution with mean=50 and standard deviation=10, find P(X < 60)."
    
    solution = solver.solve_problem(problem)
    print("=== Normal Distribution Test ===")
    print(f"Problem: {problem}")
    print(f"Problem Type: {solution.problem_type}")
    print(f"Formula Used: {solution.formula_used}")
    print(f"Final Answer: {solution.final_answer:.6f}")
    print(f"Confidence: {solution.confidence_score:.1%}")
    print("Steps:")
    for i, step in enumerate(solution.step_by_step_solution, 1):
        print(f"  {i}. {step}")
    print()

def test_mean_calculation():
    """Test mean calculation problem"""
    solver = NumericalSolver()
    problem = "Find the mean of the following data: [85, 92, 78, 96, 87, 91, 83]"
    
    solution = solver.solve_problem(problem)
    print("=== Mean Calculation Test ===")
    print(f"Problem: {problem}")
    print(f"Problem Type: {solution.problem_type}")
    print(f"Formula Used: {solution.formula_used}")
    print(f"Final Answer: {solution.final_answer:.6f}")
    print(f"Confidence: {solution.confidence_score:.1%}")
    print("Steps:")
    for i, step in enumerate(solution.step_by_step_solution, 1):
        print(f"  {i}. {step}")
    print()

def test_rag_engine():
    """Test the RAG engine for similar problems"""
    rag_engine = NumericalRAGEngine()
    query = "binomial distribution probability"
    
    similar_problems = rag_engine.search_similar_problems(query, top_k=3)
    print("=== RAG Engine Test ===")
    print(f"Query: {query}")
    print(f"Found {len(similar_problems)} similar problems:")
    for i, prob in enumerate(similar_problems, 1):
        print(f"  {i}. {prob.get('topic', 'Unknown')}: {prob.get('question', 'No question')[:60]}...")
        print(f"     Formula: {prob.get('formula', 'No formula')}")
        print(f"     Similarity: {prob.get('similarity_score', 0):.3f}")
    print()

def test_hypothesis_testing():
    """Test hypothesis testing problem"""
    solver = NumericalSolver()
    problem = "Sample mean is 52, population mean is 50, population standard deviation is 8, sample size is 36. Perform a z-test at α=0.05."
    
    solution = solver.solve_problem(problem)
    print("=== Hypothesis Testing Test ===")
    print(f"Problem: {problem}")
    print(f"Problem Type: {solution.problem_type}")
    print(f"Formula Used: {solution.formula_used}")
    print(f"Final Answer: {solution.final_answer:.6f}")
    print(f"Confidence: {solution.confidence_score:.1%}")
    print("Steps:")
    for i, step in enumerate(solution.step_by_step_solution, 1):
        print(f"  {i}. {step}")
    print()

def test_reliability_problem():
    """Test reliability problem"""
    solver = NumericalSolver()
    problem = "Two components in series with reliabilities R1=0.9 and R2=0.8. Find system reliability."
    
    solution = solver.solve_problem(problem)
    print("=== Reliability Test ===")
    print(f"Problem: {problem}")
    print(f"Problem Type: {solution.problem_type}")
    print(f"Formula Used: {solution.formula_used}")
    print(f"Final Answer: {solution.final_answer:.6f}")
    print(f"Confidence: {solution.confidence_score:.1%}")
    print("Steps:")
    for i, step in enumerate(solution.step_by_step_solution, 1):
        print(f"  {i}. {step}")
    print()

if __name__ == "__main__":
    print("Testing Numerical Solver System")
    print("=" * 50)
    
    try:
        # Test individual problem types
        test_mean_calculation()
        test_binomial_problem()
        test_normal_problem()
        test_hypothesis_testing()
        test_reliability_problem()
        
        # Test RAG engine
        test_rag_engine()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc() 