import streamlit as st
from typing import Dict
from enhanced_numerical_solver import EnhancedNumericalSolver


def render_enhanced_numerical_mode():
    st.header("🤖 Enhanced Numerical Problem Solver")
    st.markdown(
        "AI classification → Formula check → Parameter extraction → Step-by-step solution"
    )

    solver = EnhancedNumericalSolver()

    problem_text = st.text_area(
        "Enter your problem:",
        height=120,
        placeholder="Example: A fair coin is tossed 8 times. What is the probability of exactly 5 heads?",
    )

    col1, col2 = st.columns(2)
    with col1:
        btn_solve = st.button("🔎 Analyze & Solve", type="primary")
    with col2:
        btn_classify = st.button("🧭 Classify Only")

    if btn_classify and problem_text:
        with st.spinner("Classifying..."):
            cls = solver.classifier.predict(problem_text)
        _render_classification(cls)

    if btn_solve and problem_text:
        with st.spinner("Analyzing & solving..."):
            result = solver.solve_problem(problem_text)
        _render_result(result)


def _render_classification(cls: Dict):
    st.subheader("Classification")
    st.write(f"Type: {cls['type']}")
    st.write(f"Subtype: {cls['subtype']}")
    st.write(f"Confidence: {cls['confidence']:.1%}")


def _render_result(result: Dict):
    status = result.get('status')
    if status == 'unsolvable':
        st.error("❌ Cannot solve with current knowledge")
        _render_classification(result.get('classification', {}))
        fc = result.get('formula_check', {})
        if fc.get('suggested_concepts'):
            st.write("Suggested concepts to add:")
            st.write(", ".join(fc['suggested_concepts'][:10]))
        return

    st.success(f"Status: {status}")
    _render_classification(result.get('classification', {}))

    if 'parameters' in result and result['parameters'].get('numbers'):
        st.subheader("Parameters")
        for p in result['parameters']['numbers']:
            st.write(f"- {p['label']}: {p['value']} {p.get('unit','')}")

    if 'formula_used' in result:
        st.subheader("Formula")
        st.write(result['formula_used'])

    if status == 'solved' and 'solution' in result:
        st.subheader("Solution")
        sol = result['solution']
        st.write(f"Final Answer: {sol.get('final_answer')}")
        steps = sol.get('steps') or []
        if steps:
            st.write("Steps:")
            for i, step in enumerate(steps, 1):
                st.write(f"{i}. {step}") 