import json
from typing import Dict, Any, List


class FormulaKnowledgeBase:
    def __init__(self, dataset_path: str = 'comprehensive_probability_statistics_dataset.json'):
        self.dataset_path = dataset_path
        self.formulas: List[Dict[str, Any]] = []
        self._load()

    def _builtin_formulas(self) -> List[Dict[str, Any]]:
        return [
            {'section': 'basic_arithmetic', 'subsection': 'addition', 'topic': 'basic_arithmetic', 'question': '', 'formula': 'c = a + b'},
            {'section': 'basic_arithmetic', 'subsection': 'subtraction', 'topic': 'basic_arithmetic', 'question': '', 'formula': 'c = a - b'},
            {'section': 'basic_arithmetic', 'subsection': 'multiplication', 'topic': 'basic_arithmetic', 'question': '', 'formula': 'c = a × b'},
            {'section': 'basic_arithmetic', 'subsection': 'division', 'topic': 'basic_arithmetic', 'question': '', 'formula': 'c = a ÷ b'},
            {'section': 'percentage', 'subsection': 'percentage_of_number', 'topic': 'percentage', 'question': '', 'formula': 'part = (percent/100) × whole'},
            {'section': 'geometry', 'subsection': 'area_rectangle', 'topic': 'geometry', 'question': '', 'formula': 'Area = length × width'},
            {'section': 'geometry', 'subsection': 'perimeter_rectangle', 'topic': 'geometry', 'question': '', 'formula': 'Perimeter = 2 × (length + width)'},
            {'section': 'geometry', 'subsection': 'area_triangle', 'topic': 'geometry', 'question': '', 'formula': 'Area = 1/2 × base × height'},
            {'section': 'fractions', 'subsection': 'addition_same_denominator', 'topic': 'fractions', 'question': '', 'formula': 'a/n + b/n = (a+b)/n'},
            {'section': 'time_calculation', 'subsection': 'duration', 'topic': 'time', 'question': '', 'formula': 'end_time = start_time + duration'},
            {'section': 'time_calculation', 'subsection': 'age', 'topic': 'time', 'question': '', 'formula': 'age = current_year - birth_year'},
            {'section': 'algebra', 'subsection': 'linear_equation', 'topic': 'algebra', 'question': '', 'formula': 'x + a = b → x = b - a'},
            {'section': 'algebra', 'subsection': 'linear_equation_multiplication', 'topic': 'algebra', 'question': '', 'formula': 'k x = b → x = b/k'},
        ]

    def _load(self):
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        root = data.get('probability_statistics_numerical_questions', {})
        formulas: List[Dict[str, Any]] = []
        for section_name, section in root.items():
            if isinstance(section, dict):
                for subsection_name, problems in section.items():
                    if isinstance(problems, list):
                        for problem in problems:
                            if isinstance(problem, dict) and problem.get('formula'):
                                formulas.append({
                                    'section': section_name,
                                    'subsection': subsection_name,
                                    'topic': problem.get('topic', ''),
                                    'question': problem.get('question', ''),
                                    'formula': problem.get('formula', ''),
                                })
            elif isinstance(section, list):
                for problem in section:
                    if isinstance(problem, dict) and problem.get('formula'):
                        formulas.append({
                            'section': section_name,
                            'subsection': 'general',
                            'topic': problem.get('topic', ''),
                            'question': problem.get('question', ''),
                            'formula': problem.get('formula', ''),
                        })
        # Prepend built-in formulas to ensure coverage of basic types
        self.formulas = self._builtin_formulas() + formulas

    def _match_score(self, type_name: str, subtype: str, entry: Dict[str, Any]) -> int:
        score = 0
        t = (type_name or '').lower()
        st = (subtype or '').lower()
        section = (entry.get('section') or '').lower()
        subsection = (entry.get('subsection') or '').lower()
        topic = (entry.get('topic') or '').lower()
        if t and t in section:
            score += 2
        if st and st in subsection:
            score += 3
        if st and st in topic:
            score += 1
        return score

    def check_formula_availability(self, type_name: str, subtype: str) -> Dict[str, Any]:
        if not self.formulas:
            return {"available": False, "reason": "No formulas loaded", "suggested_concepts": []}
        ranked = sorted(self.formulas, key=lambda e: self._match_score(type_name, subtype, e), reverse=True)
        best = ranked[0]
        if self._match_score(type_name, subtype, best) == 0:
            suggestions = sorted({e.get('topic') or e.get('subsection') for e in self.formulas if (e.get('topic') or e.get('subsection'))})
            return {
                'available': False,
                'reason': 'No matching formula for this type/subtype',
                'suggested_concepts': suggestions[:10]
            }
        return {
            'available': True,
            'best_match': best,
            'reason': 'Formula found'
        } 