from typing import Dict, Any
from question_classifier import QuestionClassifier
from question_classifier_data_prep import prepare_classification_data
from formula_knowledge_base import FormulaKnowledgeBase
from parameter_extractor import ParameterExtractor
import re
import math


class EnhancedNumericalSolver:
    def __init__(self):
        self.classifier = QuestionClassifier()
        try:
            self.classifier.load_model('models/question_classifier.joblib')
        except Exception:
            rows = prepare_classification_data()
            if rows:
                self.classifier.train(rows)
                import os
                os.makedirs('models', exist_ok=True)
                self.classifier.save_model('models/question_classifier.joblib')
        self.kb = FormulaKnowledgeBase()
        self.params = ParameterExtractor()

    def _heuristic_override(self, question: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        q = question.lower()
        if '+' in question:
            return {'type': 'basic_arithmetic', 'subtype': 'addition', 'confidence': max(0.9, classification.get('confidence', 0.0))}
        if '-' in question and 'solve for x' not in q:
            return {'type': 'basic_arithmetic', 'subtype': 'subtraction', 'confidence': max(0.9, classification.get('confidence', 0.0))}
        if '×' in question or '*' in question:
            return {'type': 'basic_arithmetic', 'subtype': 'multiplication', 'confidence': max(0.9, classification.get('confidence', 0.0))}
        if '÷' in question or '/' in question:
            return {'type': 'basic_arithmetic', 'subtype': 'division', 'confidence': max(0.9, classification.get('confidence', 0.0))}
        if 'percent' in q or '%' in q:
            return {'type': 'percentage', 'subtype': 'percentage_of_number', 'confidence': max(0.9, classification.get('confidence', 0.0))}
        if 'area' in q and 'rectangle' in q:
            return {'type': 'geometry', 'subtype': 'area_rectangle', 'confidence': max(0.95, classification.get('confidence', 0.0))}
        if 'area' in q and 'triangle' in q:
            return {'type': 'geometry', 'subtype': 'area_triangle', 'confidence': max(0.95, classification.get('confidence', 0.0))}
        return classification

    def solve_problem(self, question: str) -> Dict[str, Any]:
        classification = self.classifier.predict(question)
        classification = self._heuristic_override(question, classification)
        kb_check = self.kb.check_formula_availability(classification['type'], classification['subtype'])
        if not kb_check['available']:
            return {
                'status': 'unsolvable',
                'reason': 'Question cannot be solved with current knowledge',
                'classification': classification,
                'formula_check': kb_check
            }
        extracted = self.params.extract_parameters(question)
        answer, steps = self._try_solve_inline(question, classification, extracted)
        if answer is None:
            return {
                'status': 'formula_only',
                'classification': classification,
                'formula_used': kb_check['best_match']['formula'],
                'parameters': extracted,
                'confidence': classification['confidence']
            }
        return {
            'status': 'solved',
            'classification': classification,
            'formula_used': kb_check['best_match']['formula'],
            'parameters': extracted,
            'solution': {
                'final_answer': answer,
                'steps': steps
            },
            'confidence': classification['confidence']
        }

    def _nCr(self, n: int, r: int) -> int:
        if r < 0 or r > n:
            return 0
        r = min(r, n - r)
        numer = 1
        denom = 1
        for k in range(1, r + 1):
            numer *= (n - (r - k))
            denom *= k
        return numer // denom

    def _try_solve_inline(self, question: str, cls: Dict[str, Any], params: Dict[str, Any]):
        ql = question.lower()
        nums = [p['value'] for p in params.get('numbers', [])]
        labels = {p['label']: p['value'] for p in params.get('numbers', [])}
        steps = []
        # Basic arithmetic
        if cls['type'] == 'basic_arithmetic':
            if '+' in question:
                s = sum(nums)
                steps.append(f"Sum numbers: {nums} -> {s}")
                return s, steps
            if '-' in question and len(nums) >= 2:
                res = nums[0] - nums[1]
                steps.append(f"Subtract: {nums[0]} - {nums[1]} = {res}")
                return res, steps
            if any(x in question for x in ['×', '*']) and nums:
                prod = 1
                for v in nums:
                    prod *= v
                steps.append(f"Multiply: product({nums}) = {prod}")
                return prod, steps
            if any(x in question for x in ['÷', '/']) and len(nums) >= 2 and nums[1] != 0:
                res = nums[0] / nums[1]
                steps.append(f"Divide: {nums[0]} / {nums[1]} = {res}")
                return res, steps
        # Percentage
        if cls['type'] == 'percentage' or 'percent' in ql:
            pct = labels.get('percent')
            other = None
            if pct is None:
                m = re.search(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', ql)
                if m:
                    pct = float(m.group(1))
                    other = float(m.group(2))
            if other is None and nums:
                other = max(nums)
            if pct is not None and other is not None:
                res = (pct / 100.0) * other
                steps.append(f"Compute {pct}% of {other} = {res}")
                return res, steps
        # Geometry rectangle area
        if 'area' in ql and 'rectangle' in ql:
            length = labels.get('length')
            width = labels.get('width')
            if length is None or width is None:
                if len(nums) >= 2:
                    length, width = nums[0], nums[1]
            if length is not None and width is not None:
                area = length * width
                steps.append(f"Area = length × width = {length} × {width} = {area}")
                return area, steps
        # Triangle area
        if 'area' in ql and 'triangle' in ql:
            base = labels.get('base')
            height = labels.get('height')
            if base is None or height is None:
                if len(nums) >= 2:
                    base, height = nums[0], nums[1]
            if base is not None and height is not None:
                area = 0.5 * base * height
                steps.append(f"Area = 1/2 × base × height = 0.5 × {base} × {height} = {area}")
                return area, steps
        # Binomial probability: detect 'toss' or 'trials' and 'exactly k'
        if 'toss' in ql or 'trials' in ql or cls.get('subtype','').startswith('binomial'):
            # infer n and k, assume p=0.5 if not provided
            n = None
            k = None
            p = None
            for pitem in params.get('numbers', []):
                if pitem['label'] == 'n' and n is None:
                    n = int(pitem['value'])
                if pitem['label'] == 'p' and p is None:
                    p = float(pitem['value'])
            # Try extract from text: "tossed N times" and "exactly K"
            m1 = re.search(r'(?:tossed|trials?)\s+(\d+)', ql)
            if m1 and n is None:
                n = int(m1.group(1))
            m2 = re.search(r'exactly\s+(\d+)', ql)
            if m2 and k is None:
                k = int(m2.group(1))
            if p is None:
                p = 0.5
            if n is not None and k is not None:
                comb = self._nCr(n, k)
                prob = comb * (p ** k) * ((1 - p) ** (n - k))
                steps.append(f"C(n,k) = C({n},{k}) = {comb}")
                steps.append(f"P = C(n,k) p^k (1-p)^(n-k) = {comb} × {p}^{k} × {1-p}^{n-k} = {prob}")
                return prob, steps
        return None, [] 