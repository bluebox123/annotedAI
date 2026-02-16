import os
import re
import json
from typing import Dict, Any, List
import requests


class ParameterExtractor:
    def __init__(self) -> None:
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        self.perplexity_url = 'https://api.perplexity.ai/chat/completions'
        self.model = 'sonar-pro'

    def extract_parameters(self, question: str) -> Dict[str, Any]:
        if self.perplexity_api_key:
            try:
                return self._extract_with_llm(question)
            except Exception:
                pass
        return self._extract_rule_based(question)

    def _extract_rule_based(self, question: str) -> Dict[str, Any]:
        numbers = re.findall(r'-?\d+\.?\d*', question)
        params: List[Dict[str, Any]] = []
        ql = question.lower()
        for i, s in enumerate(numbers):
            val = float(s)
            label = self._infer_label(ql, s, i)
            params.append({'label': label, 'value': val, 'unit': ''})
        return {'numbers': params, 'source': 'rule'}

    def _infer_label(self, ql: str, num_str: str, idx: int) -> str:
        v = float(num_str)
        if 'probability' in ql and v <= 1:
            return 'p'
        if any(k in ql for k in ['toss', 'trials', 'sample size', 'n=']):
            return 'n'
        if any(k in ql for k in ['mean', 'x̄', 'average']):
            return 'mean'
        if any(k in ql for k in ['standard deviation', 'std', 'σ']):
            return 'std_dev'
        if any(k in ql for k in ['variance', 'σ^2', 's^2']):
            return 'variance'
        if any(k in ql for k in ['alpha', 'α=']):
            return 'alpha'
        if any(k in ql for k in ['beta', 'β=']):
            return 'beta'
        if 'length' in ql and idx == 0:
            return 'length'
        if 'width' in ql and idx == 1:
            return 'width'
        if 'base' in ql:
            return 'base'
        if 'height' in ql:
            return 'height'
        if '%' in ql or 'percent' in ql:
            return 'percent'
        return f'x{idx + 1}'

    def _extract_with_llm(self, question: str) -> Dict[str, Any]:
        system_prompt = (
            "You extract numeric parameters from a math question and assign semantic labels. "
            "Return strictly JSON with a 'numbers' array of objects like {label, value, unit}. "
            "Prefer labels among: n, p, mean, std_dev, variance, alpha, beta, length, width, base, height, percent, x1..xk."
        )
        user_payload = {
            'instruction': 'Extract and label the numeric values from this question.',
            'question': question,
            'labels_hint': ['n','p','mean','std_dev','variance','alpha','beta','length','width','base','height','percent']
        }
        headers = {"Authorization": f"Bearer {self.perplexity_api_key}", "Content-Type": "application/json"}
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': json.dumps(user_payload, ensure_ascii=False)}
            ],
            'max_tokens': 400,
            'temperature': 0.0,
            'top_p': 1.0,
            'stream': False,
        }
        resp = requests.post(self.perplexity_url, json=payload, headers=headers, timeout=45)
        if resp.status_code != 200:
            return self._extract_rule_based(question)
        content = resp.json().get('choices', [{}])[0].get('message', {}).get('content', '{}')
        try:
            if not content.strip().startswith('{'):
                import re as _re
                m = _re.search(r"\{[\s\S]*\}", content)
                if m:
                    content = m.group(0)
            parsed = json.loads(content)
            nums = parsed.get('numbers', []) if isinstance(parsed, dict) else []
            normalized: List[Dict[str, Any]] = []
            for item in nums:
                try:
                    label = str(item.get('label'))
                    value = float(item.get('value'))
                    unit = str(item.get('unit') or '')
                    normalized.append({'label': label, 'value': value, 'unit': unit})
                except Exception:
                    continue
            if not normalized:
                return self._extract_rule_based(question)
            return {'numbers': normalized, 'source': 'llm'}
        except Exception:
            return self._extract_rule_based(question) 