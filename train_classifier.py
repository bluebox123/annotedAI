from question_classifier_data_prep import prepare_classification_data
from question_classifier import QuestionClassifier
import os


def train_question_classifier():
    print("Preparing training data...")
    rows = prepare_classification_data()
    print(f"Training with {len(rows)} examples...")
    clf = QuestionClassifier()
    clf.train(rows)
    os.makedirs('models', exist_ok=True)
    clf.save_model('models/question_classifier.joblib')
    print("Saved to models/question_classifier.joblib")


if __name__ == '__main__':
    train_question_classifier() 