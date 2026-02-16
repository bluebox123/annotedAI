import json
from typing import List, Dict
import re


def _safe_load_json(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _map_stats_section_to_type(section: str) -> str:
    mapping = {
        'descriptive_statistics': 'statistics',
        'probability_distributions': 'probability',
        'hypothesis_testing': 'hypothesis_testing',
        'correlation_and_regression': 'correlation_regression',
        'analysis_of_variance': 'anova',
        'reliability_theory': 'reliability',
        'advanced_topics': 'advanced_statistics',
        'module_1_descriptive_statistics': 'statistics',
        'module_2_random_variables': 'random_variables',
        'module_3_correlation_regression': 'correlation_regression',
        'module_4_probability_distributions': 'probability',
        'module_5_hypothesis_testing_large': 'hypothesis_testing',
        'module_6_hypothesis_testing_small': 'hypothesis_testing',
        'module_7_reliability': 'reliability',
        'comprehensive_problems': 'comprehensive',
        'additional_advanced_problems': 'advanced_statistics',
    }
    return mapping.get(section, 'statistics')


def _infer_subtype_from_stats(section: str, subsection: str, problem: Dict) -> str:
    # Derive a granular subtype keyword to help downstream routing
    topic = (problem.get('topic') or '').lower()
    q = (problem.get('question') or '').lower()
    if section == 'probability_distributions':
        return subsection
    if 'binomial' in topic or 'binomial' in q:
        return 'binomial_distribution'
    if 'poisson' in topic or 'poisson' in q:
        return 'poisson_distribution'
    if 'normal' in topic or 'normal' in q:
        return 'normal_distribution'
    if 'exponential' in topic or 'exponential' in q:
        return 'exponential_distribution'
    if 'gamma' in topic or 'gamma' in q:
        return 'gamma_distribution'
    if 'anova' in topic or 'anova' in q:
        return 'anova'
    if 'regression' in topic or 'correlation' in topic:
        return 'correlation_regression'
    if 'hypothesis' in topic or 'z-test' in q or 't-test' in q:
        return 'hypothesis_testing'
    if 'reliability' in topic:
        return 'reliability'
    if 'skewness' in topic or 'kurtosis' in topic or 'quartile' in q:
        return 'descriptive_statistics'
    return subsection or 'general'


def _rule_infer_basic_subtype(question: str) -> Dict[str, str]:
    q = question.lower()
    if any(op in question for op in ['+', '-', '×', '÷', '*', '/']):
        if '+' in question:
            return {'type': 'basic_arithmetic', 'subtype': 'addition'}
        if '-' in question:
            return {'type': 'basic_arithmetic', 'subtype': 'subtraction'}
        if '×' in question or '*' in question:
            return {'type': 'basic_arithmetic', 'subtype': 'multiplication'}
        if '÷' in question or '/' in question:
            return {'type': 'basic_arithmetic', 'subtype': 'division'}
    if 'percent' in q or '%' in q:
        return {'type': 'percentage', 'subtype': 'percentage_of_number'}
    if 'area' in q and 'rectangle' in q:
        return {'type': 'geometry', 'subtype': 'area_rectangle'}
    if 'area' in q and 'triangle' in q:
        return {'type': 'geometry', 'subtype': 'area_triangle'}
    if re.search(r'\bsolve\s+for\s+x\b', q) and ('+' in question or '-' in question):
        return {'type': 'algebra', 'subtype': 'linear_equation'}
    return {}


def prepare_classification_data(
    numerical_path: str = 'numerical_questions_dataset.json',
    stats_path: str = 'comprehensive_probability_statistics_dataset.json'
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    num = _safe_load_json(numerical_path)
    for q in num.get('questions', []):
        question_text = q.get('question')
        q_type = q.get('type') or 'unknown'
        q_subtype = q.get('subtype') or 'general'
        if question_text:
            rows.append({'question': question_text, 'type': q_type, 'subtype': q_subtype})

    comp = _safe_load_json(stats_path)
    root = comp.get('probability_statistics_numerical_questions', {})
    for section_name, section in root.items():
        if isinstance(section, dict):
            for subsection_name, problems in section.items():
                if isinstance(problems, list):
                    for problem in problems:
                        if not isinstance(problem, dict):
                            continue
                        question_text = problem.get('question')
                        if not question_text:
                            continue
                        type_main = _map_stats_section_to_type(section_name)
                        subtype = _infer_subtype_from_stats(section_name, subsection_name, problem)
                        rows.append({'question': question_text, 'type': type_main, 'subtype': subtype})
        elif isinstance(section, list):
            for problem in section:
                if not isinstance(problem, dict):
                    continue
                question_text = problem.get('question')
                if not question_text:
                    continue
                type_main = _map_stats_section_to_type(section_name)
                subtype = _infer_subtype_from_stats(section_name, 'general', problem)
                rows.append({'question': question_text, 'type': type_main, 'subtype': subtype})

    # Add synthetic basic examples to anchor classifier
    synthetic = [
        'What is 2 + 3?',
        'Compute 20% of 150',
        'Find the area of a rectangle with length 5 and width 8',
        'Solve for x: x + 5 = 12',
        'What is 10 × 15?',
    ]
    for s in synthetic:
        lab = _rule_infer_basic_subtype(s)
        if lab:
            rows.append({'question': s, 'type': lab['type'], 'subtype': lab['subtype']})

    return rows 