import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
from numerical_solver import NumericalSolver, ProblemSolution
from numerical_rag_engine import NumericalRAGEngine

class NumericalInterface:
    """
    Streamlit interface for the numerical probability and statistics solver.
    Provides interactive problem solving with step-by-step solutions.
    """
    
    def __init__(self):
        self.solver = NumericalSolver()
        self.rag_engine = NumericalRAGEngine()
        
        # Initialize session state
        if 'numerical_history' not in st.session_state:
            st.session_state.numerical_history = []
        if 'current_solution' not in st.session_state:
            st.session_state.current_solution = None
    
    def render_interface(self):
        """Render the main numerical solver interface"""
        st.header("🧮 Numerical Problem Solver")
        st.markdown("""
        **Advanced Probability & Statistics Calculator**
        
        This mode uses a hybrid approach combining:
        - 📚 **Formula-based RAG**: Retrieves relevant formulas and examples
        - 🧠 **Step-by-step reasoning**: Shows detailed solution process
        - 🔢 **Computational tools**: Accurate numerical calculations
        - 📊 **Similar examples**: Learning from the comprehensive dataset
        """)
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Problem Solver", 
            "📖 Formula Library", 
            "📈 Solution History", 
            "🎯 Practice Problems"
        ])
        
        with tab1:
            self.render_problem_solver()
        
        with tab2:
            self.render_formula_library()
        
        with tab3:
            self.render_solution_history()
        
        with tab4:
            self.render_practice_problems()
    
    def render_problem_solver(self):
        """Render the main problem solving interface"""
        st.subheader("Enter Your Problem")
        
        # Problem input methods
        input_method = st.radio(
            "Input method:",
            ["Text Input", "Guided Input", "Example Problems"],
            horizontal=True
        )
        
        if input_method == "Text Input":
            self.render_text_input()
        elif input_method == "Guided Input":
            self.render_guided_input()
        else:
            self.render_example_problems()
    
    def render_text_input(self):
        """Render free-form text input for problems"""
        # Text area for problem input
        problem_text = st.text_area(
            "Describe your probability/statistics problem:",
            height=100,
            placeholder="Example: A coin is tossed 10 times. Find the probability of getting exactly 6 heads."
        )
        
        # Solve button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            solve_button = st.button("🔍 Solve Problem", type="primary")
        
        with col2:
            if st.button("💡 Get Hints"):
                if problem_text:
                    self.show_hints(problem_text)
        
        with col3:
            st.markdown("*Tip: Be specific with numerical values and parameters*")
        
        if solve_button and problem_text:
            with st.spinner("Solving problem..."):
                solution = self.solver.solve_problem(problem_text)
                self.display_solution(solution, problem_text)
                
                # Add to history
                st.session_state.numerical_history.append({
                    'problem': problem_text,
                    'solution': solution,
                    'timestamp': st.session_state.get('timestamp', 'now')
                })
    
    def render_guided_input(self):
        """Render guided input for specific problem types"""
        st.subheader("Guided Problem Setup")
        
        # Problem type selection
        problem_types = [
            "Descriptive Statistics",
            "Binomial Distribution", 
            "Poisson Distribution",
            "Normal Distribution",
            "Hypothesis Testing",
            "Correlation & Regression",
            "Reliability Theory"
        ]
        
        selected_type = st.selectbox("Select problem type:", problem_types)
        
        if selected_type == "Descriptive Statistics":
            self.render_descriptive_stats_input()
        elif selected_type == "Binomial Distribution":
            self.render_binomial_input()
        elif selected_type == "Poisson Distribution":
            self.render_poisson_input()
        elif selected_type == "Normal Distribution":
            self.render_normal_input()
        elif selected_type == "Hypothesis Testing":
            self.render_hypothesis_input()
        elif selected_type == "Correlation & Regression":
            self.render_correlation_input()
        elif selected_type == "Reliability Theory":
            self.render_reliability_input()
    
    def render_descriptive_stats_input(self):
        """Guided input for descriptive statistics"""
        st.write("**Descriptive Statistics Calculator**")
        
        # Data input
        data_input_method = st.radio("Data input:", ["Manual Entry", "Upload CSV"], horizontal=True)
        
        if data_input_method == "Manual Entry":
            data_text = st.text_input("Enter data (comma-separated):", placeholder="85, 92, 78, 96, 87, 91, 83")
            
            if data_text:
                try:
                    data = [float(x.strip()) for x in data_text.split(',')]
                    st.write(f"Data: {data}")
                    
                    if st.button("Calculate Statistics"):
                        problem = f"Calculate descriptive statistics for the data: {data}"
                        solution = self.solver.solve_problem(problem)
                        self.display_solution(solution, problem)
                        
                except ValueError:
                    st.error("Please enter valid numbers separated by commas")
    
    def render_binomial_input(self):
        """Guided input for binomial distribution"""
        st.write("**Binomial Distribution Calculator**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n = st.number_input("Number of trials (n):", min_value=1, value=10, step=1)
            p = st.number_input("Probability of success (p):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        with col2:
            calculation_type = st.selectbox("Calculate:", [
                "P(X = k) - Exact probability",
                "P(X ≤ k) - Cumulative probability", 
                "Mean and Variance",
                "Most likely value"
            ])
            
            if "P(X" in calculation_type:
                k = st.number_input("Value of k:", min_value=0, max_value=n, value=5, step=1)
        
        if st.button("Calculate"):
            if "Exact" in calculation_type:
                problem = f"For binomial distribution with n={n}, p={p}, find P(X = {k})"
            elif "Cumulative" in calculation_type:
                problem = f"For binomial distribution with n={n}, p={p}, find P(X ≤ {k})"
            else:
                problem = f"For binomial distribution with n={n}, p={p}, find mean and variance"
            
            solution = self.solver.solve_problem(problem)
            self.display_solution(solution, problem)
    
    def render_poisson_input(self):
        """Guided input for Poisson distribution"""
        st.write("**Poisson Distribution Calculator**")
        
        lambda_val = st.number_input("Rate parameter (λ):", min_value=0.1, value=3.0, step=0.1)
        
        calculation_type = st.selectbox("Calculate:", [
            "P(X = k) - Exact probability",
            "P(X ≤ k) - Cumulative probability",
            "Mean, Variance, and Standard Deviation"
        ])
        
        if "P(X" in calculation_type:
            k = st.number_input("Value of k:", min_value=0, value=5, step=1)
        
        if st.button("Calculate"):
            if "Exact" in calculation_type:
                problem = f"For Poisson distribution with λ={lambda_val}, find P(X = {k})"
            elif "Cumulative" in calculation_type:
                problem = f"For Poisson distribution with λ={lambda_val}, find P(X ≤ {k})"
            else:
                problem = f"For Poisson distribution with λ={lambda_val}, find mean, variance, and standard deviation"
            
            solution = self.solver.solve_problem(problem)
            self.display_solution(solution, problem)
    
    def render_normal_input(self):
        """Guided input for normal distribution"""
        st.write("**Normal Distribution Calculator**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mu = st.number_input("Mean (μ):", value=50.0, step=0.1)
            sigma = st.number_input("Standard deviation (σ):", min_value=0.1, value=10.0, step=0.1)
        
        with col2:
            calculation_type = st.selectbox("Calculate:", [
                "P(X < x) - Left tail probability",
                "P(X > x) - Right tail probability",
                "P(a < X < b) - Between two values",
                "Percentile (x for given probability)"
            ])
        
        if "P(X <" in calculation_type or "P(X >" in calculation_type:
            x = st.number_input("Value of x:", value=60.0, step=0.1)
        elif "P(a <" in calculation_type:
            col_a, col_b = st.columns(2)
            with col_a:
                a = st.number_input("Lower bound (a):", value=40.0, step=0.1)
            with col_b:
                b = st.number_input("Upper bound (b):", value=60.0, step=0.1)
        elif "Percentile" in calculation_type:
            percentile = st.number_input("Percentile (0-100):", min_value=0.1, max_value=99.9, value=90.0, step=0.1)
        
        if st.button("Calculate"):
            if "P(X <" in calculation_type:
                problem = f"For normal distribution N({mu}, {sigma}²), find P(X < {x})"
            elif "P(X >" in calculation_type:
                problem = f"For normal distribution N({mu}, {sigma}²), find P(X > {x})"
            elif "P(a <" in calculation_type:
                problem = f"For normal distribution N({mu}, {sigma}²), find P({a} < X < {b})"
            else:
                problem = f"For normal distribution N({mu}, {sigma}²), find the {percentile}th percentile"
            
            solution = self.solver.solve_problem(problem)
            self.display_solution(solution, problem)
    
    def render_hypothesis_input(self):
        """Guided input for hypothesis testing"""
        st.write("**Hypothesis Testing Calculator**")
        
        test_type = st.selectbox("Test type:", [
            "One-sample z-test (known σ)",
            "One-sample t-test (unknown σ)",
            "Two-sample t-test",
            "Proportion test"
        ])
        
        alpha = st.number_input("Significance level (α):", min_value=0.001, max_value=0.5, value=0.05, step=0.001)
        
        if "One-sample z-test" in test_type:
            col1, col2 = st.columns(2)
            with col1:
                sample_mean = st.number_input("Sample mean:", value=52.0)
                pop_mean = st.number_input("Population mean (H₀):", value=50.0)
            with col2:
                pop_std = st.number_input("Population std (σ):", min_value=0.1, value=8.0)
                n = st.number_input("Sample size:", min_value=1, value=36, step=1)
            
            if st.button("Perform Test"):
                problem = f"One-sample z-test: sample mean={sample_mean}, population mean={pop_mean}, σ={pop_std}, n={n}, α={alpha}"
                solution = self.solver.solve_problem(problem)
                self.display_solution(solution, problem)
    
    def render_correlation_input(self):
        """Guided input for correlation and regression"""
        st.write("**Correlation & Regression Calculator**")
        
        analysis_type = st.selectbox("Analysis type:", [
            "Correlation coefficient",
            "Linear regression",
            "Coefficient of determination"
        ])
        
        # Data input
        col1, col2 = st.columns(2)
        with col1:
            x_data = st.text_input("X values (comma-separated):", placeholder="1, 2, 3, 4, 5")
        with col2:
            y_data = st.text_input("Y values (comma-separated):", placeholder="2, 4, 6, 8, 10")
        
        if st.button("Calculate") and x_data and y_data:
            try:
                x_vals = [float(x.strip()) for x in x_data.split(',')]
                y_vals = [float(y.strip()) for y in y_data.split(',')]
                
                if len(x_vals) != len(y_vals):
                    st.error("X and Y must have the same number of values")
                    return
                
                if analysis_type == "Correlation coefficient":
                    problem = f"Calculate correlation coefficient for X={x_vals} and Y={y_vals}"
                elif analysis_type == "Linear regression":
                    problem = f"Find linear regression equation for X={x_vals} and Y={y_vals}"
                else:
                    problem = f"Calculate coefficient of determination for X={x_vals} and Y={y_vals}"
                
                solution = self.solver.solve_problem(problem)
                self.display_solution(solution, problem)
                
            except ValueError:
                st.error("Please enter valid numbers")
    
    def render_reliability_input(self):
        """Guided input for reliability problems"""
        st.write("**Reliability Theory Calculator**")
        
        system_type = st.selectbox("System type:", [
            "Single component (exponential)",
            "Series system",
            "Parallel system",
            "MTBF calculation"
        ])
        
        if system_type == "Single component (exponential)":
            col1, col2 = st.columns(2)
            with col1:
                lambda_val = st.number_input("Failure rate (λ):", min_value=0.001, value=0.01, step=0.001)
            with col2:
                time = st.number_input("Time (t):", min_value=0.0, value=100.0, step=1.0)
            
            if st.button("Calculate Reliability"):
                problem = f"Calculate reliability for exponential distribution with λ={lambda_val} at time t={time}"
                solution = self.solver.solve_problem(problem)
                self.display_solution(solution, problem)
        
        elif system_type in ["Series system", "Parallel system"]:
            num_components = st.number_input("Number of components:", min_value=2, max_value=10, value=2, step=1)
            
            reliabilities = []
            cols = st.columns(min(num_components, 4))
            for i in range(num_components):
                with cols[i % 4]:
                    r = st.number_input(f"R{i+1}:", min_value=0.0, max_value=1.0, value=0.9, step=0.01, key=f"r_{i}")
                    reliabilities.append(r)
            
            if st.button("Calculate System Reliability"):
                problem = f"Calculate {system_type.lower()} reliability with components R={reliabilities}"
                solution = self.solver.solve_problem(problem)
                self.display_solution(solution, problem)
    
    def display_solution(self, solution: ProblemSolution, original_problem: str):
        """Display the complete solution with formatting"""
        st.session_state.current_solution = solution
        
        # Solution header
        st.success("✅ Problem Solved!")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Problem type and confidence
            st.subheader("📊 Solution Details")
            
            confidence_color = "green" if solution.confidence_score > 0.8 else "orange" if solution.confidence_score > 0.6 else "red"
            st.markdown(f"""
            **Problem Type:** {solution.problem_type.replace('_', ' ').title()}  
            **Confidence:** <span style="color: {confidence_color}">●</span> {solution.confidence_score:.1%}
            """, unsafe_allow_html=True)
            
            # Given values
            if solution.given_values:
                st.subheader("📝 Given Values")
                for key, value in solution.given_values.items():
                    if isinstance(value, list):
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            # Formula used
            st.subheader("📐 Formula Used")
            st.code(solution.formula_used, language="text")
            
            # Step-by-step solution
            st.subheader("🔢 Step-by-Step Solution")
            for i, step in enumerate(solution.step_by_step_solution, 1):
                st.write(f"**Step {i}:** {step}")
            
            # Final answer
            st.subheader("🎯 Final Answer")
            st.success(f"**Result: {solution.final_answer:.6f}**")
        
        with col2:
            # Similar examples
            if solution.similar_examples:
                st.subheader("💡 Similar Examples")
                for i, example in enumerate(solution.similar_examples[:2], 1):
                    with st.expander(f"Example {i}: {example.get('topic', 'Related Problem')}"):
                        st.write(f"**Question:** {example.get('question', 'N/A')}")
                        if example.get('formula'):
                            st.write(f"**Formula:** {example['formula']}")
                        if example.get('solution'):
                            st.write(f"**Answer:** {example['solution']}")
            
            # Action buttons
            st.subheader("🔧 Actions")
            if st.button("📊 Visualize"):
                self.create_visualization(solution, original_problem)
            
            if st.button("📋 Export Solution"):
                self.export_solution(solution, original_problem)
    
    def create_visualization(self, solution: ProblemSolution, problem: str):
        """Create visualizations for the solution"""
        try:
            if solution.problem_type == "binomial":
                self.plot_binomial_distribution(solution.given_values)
            elif solution.problem_type == "poisson":
                self.plot_poisson_distribution(solution.given_values)
            elif solution.problem_type == "normal":
                self.plot_normal_distribution(solution.given_values)
            elif solution.problem_type == "correlation":
                self.plot_correlation(solution.given_values)
            else:
                st.info("Visualization not available for this problem type")
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
    
    def plot_binomial_distribution(self, values: Dict):
        """Plot binomial distribution"""
        if 'n' in values and 'p' in values:
            import numpy as np
            from scipy.stats import binom
            
            n, p = int(values['n']), values['p']
            x = np.arange(0, n+1)
            pmf = binom.pmf(x, n, p)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=x, y=pmf, name='Probability'))
            fig.update_layout(
                title=f'Binomial Distribution (n={n}, p={p})',
                xaxis_title='Number of Successes',
                yaxis_title='Probability'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_poisson_distribution(self, values: Dict):
        """Plot Poisson distribution"""
        if 'lambda' in values:
            import numpy as np
            from scipy.stats import poisson
            
            lam = values['lambda']
            x = np.arange(0, int(lam * 3) + 1)
            pmf = poisson.pmf(x, lam)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=x, y=pmf, name='Probability'))
            fig.update_layout(
                title=f'Poisson Distribution (λ={lam})',
                xaxis_title='Number of Events',
                yaxis_title='Probability'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_normal_distribution(self, values: Dict):
        """Plot normal distribution"""
        if 'mean' in values and 'std' in values:
            import numpy as np
            from scipy.stats import norm
            
            mu, sigma = values['mean'], values['std']
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
            pdf = norm.pdf(x, mu, sigma)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF'))
            fig.update_layout(
                title=f'Normal Distribution (μ={mu}, σ={sigma})',
                xaxis_title='Value',
                yaxis_title='Probability Density'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_correlation(self, values: Dict):
        """Plot correlation scatter plot"""
        if 'x_values' in values and 'y_values' in values:
            x, y = values['x_values'], values['y_values']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers+lines', name='Data'))
            fig.update_layout(
                title='Correlation Plot',
                xaxis_title='X Values',
                yaxis_title='Y Values'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def export_solution(self, solution: ProblemSolution, problem: str):
        """Export solution to various formats"""
        export_format = st.selectbox("Export format:", ["JSON", "Text", "LaTeX"])
        
        if export_format == "JSON":
            export_data = {
                "problem": problem,
                "solution": {
                    "type": solution.problem_type,
                    "given_values": solution.given_values,
                    "formula": solution.formula_used,
                    "steps": solution.step_by_step_solution,
                    "answer": solution.final_answer,
                    "confidence": solution.confidence_score
                }
            }
            st.download_button(
                "Download JSON",
                json.dumps(export_data, indent=2),
                file_name="solution.json",
                mime="application/json"
            )
    
    def show_hints(self, problem_text: str):
        """Show hints for solving the problem"""
        similar_problems = self.rag_engine.search_similar_problems(problem_text, top_k=3)
        
        if similar_problems:
            st.info("💡 **Hints based on similar problems:**")
            for i, prob in enumerate(similar_problems, 1):
                st.write(f"**{i}.** {prob.get('topic', 'Related')}: Look for {prob.get('formula', 'relevant formula')}")
        else:
            st.info("💡 **General hints:** Make sure to identify the distribution type and extract all numerical parameters.")
    
    def render_formula_library(self):
        """Render the formula library interface"""
        st.subheader("📖 Formula Library")
        
        formula_lib = self.rag_engine.get_formula_library()
        
        if formula_lib:
            for category, formulas in formula_lib.items():
                with st.expander(f"📂 {category.replace('_', ' ').title()}"):
                    for formula in formulas:
                        st.code(formula, language="text")
        else:
            st.info("Formula library is being built from the dataset...")
    
    def render_solution_history(self):
        """Render solution history"""
        st.subheader("📈 Solution History")
        
        if st.session_state.numerical_history:
            for i, entry in enumerate(reversed(st.session_state.numerical_history), 1):
                with st.expander(f"Problem {len(st.session_state.numerical_history) - i + 1}: {entry['solution'].problem_type}"):
                    st.write(f"**Problem:** {entry['problem']}")
                    st.write(f"**Answer:** {entry['solution'].final_answer:.6f}")
                    st.write(f"**Formula:** {entry['solution'].formula_used}")
        else:
            st.info("No solutions in history yet. Solve some problems to see them here!")
        
        if st.session_state.numerical_history:
            if st.button("🗑️ Clear History"):
                st.session_state.numerical_history = []
                st.rerun()
    
    def render_practice_problems(self):
        """Render practice problems from the dataset"""
        st.subheader("🎯 Practice Problems")
        
        # Get problem types from dataset
        problem_types = self.rag_engine.get_problem_types()
        
        if problem_types:
            selected_category = st.selectbox("Choose category:", problem_types)
            
            # Get problems for the selected category
            category_problems = self.rag_engine.get_formula_by_topic(selected_category)
            
            if category_problems:
                selected_problem = st.selectbox(
                    "Choose a practice problem:",
                    options=range(len(category_problems)),
                    format_func=lambda x: f"{category_problems[x].get('topic', 'Problem')} - {category_problems[x].get('question', '')[:50]}..."
                )
                
                if selected_problem is not None:
                    prob = category_problems[selected_problem]
                    
                    st.write("**Problem:**")
                    st.write(prob.get('question', ''))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💡 Show Hint"):
                            if prob.get('formula'):
                                st.info(f"**Hint:** Use the formula: {prob['formula']}")
                    
                    with col2:
                        if st.button("✅ Show Solution"):
                            st.success(f"**Answer:** {prob.get('solution', 'Solution not available')}")
                            
                            if st.button("🔍 Solve with our system"):
                                solution = self.solver.solve_problem(prob.get('question', ''))
                                self.display_solution(solution, prob.get('question', ''))
        else:
            st.info("Loading practice problems from dataset...")

# Initialize and render the interface
def render_numerical_mode():
    """Main function to render the numerical mode interface"""
    interface = NumericalInterface()
    interface.render_interface() 