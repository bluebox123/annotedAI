from typing import List, Dict, Any
from dataclasses import dataclass
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import ClassifierChain
import numpy as np


@dataclass
class ClassificationResult:
    type: str
    subtype: str
    confidence: float


class QuestionClassifier:
    """Classify question into type and subtype using TF-IDF + RandomForest.

    - type: coarse class (e.g., basic_arithmetic, statistics, probability)
    - subtype: finer-grained tag (e.g., binomial_distribution, area_rectangle)
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words='english')
        self.type_encoder = LabelEncoder()
        self.subtype_encoder = LabelEncoder()
        self.type_model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.subtype_model = RandomForestClassifier(n_estimators=200, random_state=42)
        self._is_trained = False

    def train(self, training_rows: List[Dict[str, str]]):
        questions = [r['question'] for r in training_rows]
        types = [r['type'] for r in training_rows]
        subtypes = [r.get('subtype', 'general') for r in training_rows]

        X = self.vectorizer.fit_transform(questions)
        y_type = self.type_encoder.fit_transform(types)
        y_subtype = self.subtype_encoder.fit_transform(subtypes)

        self.type_model.fit(X, y_type)
        # Train subtype model possibly conditioned on type using ClassifierChain idea
        # For simplicity, just fit directly on X here
        self.subtype_model.fit(X, y_subtype)

        self._is_trained = True

    def predict(self, question: str) -> Dict[str, Any]:
        if not self._is_trained:
            return {"type": "unknown", "subtype": "general", "confidence": 0.0}

        X = self.vectorizer.transform([question])
        type_idx = self.type_model.predict(X)[0]
        subtype_idx = self.subtype_model.predict(X)[0]

        type_proba = self.type_model.predict_proba(X)
        subtype_proba = self.subtype_model.predict_proba(X)
        conf = float(max(type_proba.max(), subtype_proba.max()))

        return {
            "type": self.type_encoder.inverse_transform([type_idx])[0],
            "subtype": self.subtype_encoder.inverse_transform([subtype_idx])[0],
            "confidence": conf,
        }

    def save_model(self, path: str):
        if not self._is_trained:
            return
        joblib.dump({
            'vectorizer': self.vectorizer,
            'type_encoder': self.type_encoder,
            'subtype_encoder': self.subtype_encoder,
            'type_model': self.type_model,
            'subtype_model': self.subtype_model,
            'is_trained': self._is_trained
        }, path)

    def load_model(self, path: str):
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.type_encoder = data['type_encoder']
        self.subtype_encoder = data['subtype_encoder']
        self.type_model = data['type_model']
        self.subtype_model = data['subtype_model']
        self._is_trained = data.get('is_trained', True) 