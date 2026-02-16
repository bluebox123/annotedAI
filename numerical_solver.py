from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import math


@dataclass
class ProblemSolution:
    problem_type: str
    formula_used: str
    final_answer: float
    step_by_step_solution: List[str]
    confidence_score: float = 1.0
    given_values: Optional[Dict[str, Any]] = None
    similar_examples: Optional[List[Dict[str, Any]]] = None


class NumericalSolver:
    """Minimal numerical solver scaffolding for integration."""

    def solve_problem(self, problem: str) -> ProblemSolution:
        pl = problem.lower()
        steps: List[str] = []

        # Mean of a list
        if 'mean' in pl and any(ch in pl for ch in ['[', ']', ',']):
            import re
            nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', problem)]
            n = len(nums) if nums else 1
            s = sum(nums)
            mean = s / n
            steps.append(f"Sum values: {nums} -> {s}")
            steps.append(f"Divide by count {n}: {s}/{n} = {mean}")
            
            # Extract given values for the mean calculation
            given_values = {
                'data_values': nums,
                'count': n,
                'sum': s
            }
            
            return ProblemSolution(
                problem_type='descriptive_statistics',
                formula_used='Mean = Sum(x)/n',
                final_answer=mean,
                step_by_step_solution=steps,
                confidence_score=0.9,
                given_values=given_values,
                similar_examples=[]
            )

        # Simple normal probability P(X < a) when mean and std given plainly
        if 'normal' in pl and ('p(x <' in pl or 'p(x<' in pl or 'p(x < ' in pl or 'find p(x <' in pl or 'find p(x<' in pl):
            import re
            nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', problem)]
            # naive parse: expect mean, std, x
            if len(nums) >= 3:
                mean, std, x = nums[0], nums[1], nums[2]
                z = (x - mean) / std if std else 0
                steps.append(f"z = (x - mean)/std = ({x} - {mean})/{std} = {z}")
                # standard normal CDF approximation via error function
                cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
                steps.append(f"Normal CDF({z}) = {cdf}")
                
                given_values = {
                    'mean': mean,
                    'standard_deviation': std,
                    'x_value': x,
                    'z_score': z
                }
                
                return ProblemSolution(
                    problem_type='normal_distribution',
                    formula_used='Z = (X - mean)/std',
                    final_answer=cdf,
                    step_by_step_solution=steps,
                    confidence_score=0.6,
                    given_values=given_values,
                    similar_examples=[]
                )

        # Fallback
        return ProblemSolution(
            problem_type='general',
            formula_used='N/A',
            final_answer=float('nan'),
            step_by_step_solution=['No specialized solver available.'],
            confidence_score=0.2,
            given_values=None,
            similar_examples=None
        )
