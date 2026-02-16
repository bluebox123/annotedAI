import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from text_encoder import get_text_encoder
import re

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

class NumericalRAGEngine:
    """
    Specialized RAG engine for numerical probability and statistics problems.
    Indexes formulas, worked examples, and mathematical concepts.
    """
    
    def __init__(self, dataset_path: str = "comprehensive_probability_statistics_dataset.json"):
        self.encoder = get_text_encoder()
        self.index = None
        self.formula_chunks: List[Dict] = []
        self.embeddings: np.ndarray = None
        self.use_faiss = _HAS_FAISS
        
        # Load and process the dataset
        self.dataset = self._load_dataset(dataset_path)
        self._build_formula_index()
    
    def _load_dataset(self, dataset_path: str) -> Dict:
        """Load the comprehensive dataset"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"probability_statistics_numerical_questions": {}}
    
    def _build_formula_index(self):
        """Build searchable index from formulas and examples in the dataset"""
        self.formula_chunks = []
        
        if "probability_statistics_numerical_questions" in self.dataset:
            dataset_sections = self.dataset["probability_statistics_numerical_questions"]
            
            for section_name, section_data in dataset_sections.items():
                if isinstance(section_data, dict):
                    for subsection_name, problems in section_data.items():
                        if isinstance(problems, list):
                            for problem in problems:
                                if isinstance(problem, dict):
                                    self._process_problem_entry(problem, section_name, subsection_name)
                elif isinstance(section_data, list):
                    for problem in section_data:
                        if isinstance(problem, dict):
                            self._process_problem_entry(problem, section_name, "general")
        
        # Create embeddings and index
        if self.formula_chunks:
            texts = [chunk["searchable_text"] for chunk in self.formula_chunks]
            self.embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            
            if self.use_faiss:
                dim = int(self.embeddings.shape[1])
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(self.embeddings)
    
    def _process_problem_entry(self, problem: Dict, section: str, subsection: str):
        """Process a single problem entry and create searchable chunks"""
        if not isinstance(problem, dict):
            return
        
        # Create comprehensive searchable text
        searchable_parts = []
        
        # Add topic and question
        if "topic" in problem:
            searchable_parts.append(f"Topic: {problem['topic']}")
        if "question" in problem:
            searchable_parts.append(f"Question: {problem['question']}")
        
        # Add formula information
        if "formula" in problem:
            searchable_parts.append(f"Formula: {problem['formula']}")
        
        # Add parameter information
        for key in ["n", "p", "lambda", "mean", "std_dev", "alpha", "beta"]:
            if key in problem:
                searchable_parts.append(f"{key}: {problem[key]}")
        
        # Add solution context
        if "solution" in problem:
            searchable_parts.append(f"Solution: {problem['solution']}")
        
        # Create the chunk
        chunk = {
            "id": problem.get("id", ""),
            "section": section,
            "subsection": subsection,
            "topic": problem.get("topic", ""),
            "question": problem.get("question", ""),
            "formula": problem.get("formula", ""),
            "solution": problem.get("solution", ""),
            "searchable_text": " | ".join(searchable_parts),
            "full_problem": problem
        }
        
        self.formula_chunks.append(chunk)
    
    def search_similar_problems(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar problems and formulas"""
        if not self.formula_chunks or self.embeddings is None:
            return []
        
        # Encode the query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        if self.use_faiss and self.index is not None:
            # Use FAISS for similarity search
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.formula_chunks)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.formula_chunks):
                    chunk = self.formula_chunks[idx].copy()
                    chunk["similarity_score"] = float(score)
                    results.append(chunk)
            
            return results
        else:
            # Fallback to numpy similarity search
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                chunk = self.formula_chunks[idx].copy()
                chunk["similarity_score"] = float(similarities[idx])
                results.append(chunk)
            
            return results
    
    def get_formula_by_topic(self, topic: str) -> List[Dict]:
        """Get formulas and examples for a specific topic"""
        topic_lower = topic.lower()
        matching_chunks = []
        
        for chunk in self.formula_chunks:
            if (topic_lower in chunk["topic"].lower() or 
                topic_lower in chunk["section"].lower() or 
                topic_lower in chunk["subsection"].lower()):
                matching_chunks.append(chunk)
        
        return matching_chunks
    
    def get_problem_types(self) -> List[str]:
        """Get all available problem types from the dataset"""
        types = set()
        for chunk in self.formula_chunks:
            types.add(chunk["topic"])
            types.add(chunk["section"])
        
        return sorted(list(types))
    
    def search_by_parameters(self, parameters: Dict[str, float]) -> List[Dict]:
        """Search for problems with similar parameters"""
        matching_chunks = []
        
        for chunk in self.formula_chunks:
            problem = chunk["full_problem"]
            match_score = 0
            total_params = len(parameters)
            
            if total_params == 0:
                continue
            
            for param, value in parameters.items():
                if param in problem:
                    try:
                        problem_value = float(problem[param])
                        # Check if values are close (within 20% tolerance)
                        if abs(problem_value - value) / max(abs(value), 1) < 0.2:
                            match_score += 1
                    except (ValueError, TypeError):
                        continue
            
            # If at least 50% of parameters match, include it
            if match_score / total_params >= 0.5:
                chunk_copy = chunk.copy()
                chunk_copy["parameter_match_score"] = match_score / total_params
                matching_chunks.append(chunk_copy)
        
        # Sort by match score
        matching_chunks.sort(key=lambda x: x["parameter_match_score"], reverse=True)
        return matching_chunks[:5]
    
    def get_formula_library(self) -> Dict[str, List[str]]:
        """Get organized formula library by category"""
        formula_lib = {}
        
        for chunk in self.formula_chunks:
            section = chunk["section"]
            if section not in formula_lib:
                formula_lib[section] = []
            
            if chunk["formula"] and chunk["formula"] not in formula_lib[section]:
                formula_lib[section].append(chunk["formula"])
        
        return formula_lib
    
    def enhance_query_with_context(self, query: str) -> str:
        """Enhance user query with mathematical context"""
        enhanced_query = query
        
        # Add mathematical keywords based on query content
        math_keywords = {
            "mean": ["average", "expected value", "central tendency"],
            "probability": ["chance", "likelihood", "odds"],
            "distribution": ["random variable", "pdf", "pmf"],
            "test": ["hypothesis", "significance", "p-value"],
            "correlation": ["relationship", "association", "regression"]
        }
        
        query_lower = query.lower()
        for main_term, related_terms in math_keywords.items():
            if main_term in query_lower:
                enhanced_query += " " + " ".join(related_terms)
        
        return enhanced_query 