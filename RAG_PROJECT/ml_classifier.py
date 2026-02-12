"""
ML Classification Service for ArXiv Paper Category Prediction
Uses the trained and tuned Logistic Regression model to classify abstracts
"""

import joblib
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


class PaperClassifier:
    """
    Service for classifying academic paper abstracts into categories.
    Uses the pre-trained ML model with TF-IDF vectorization and text preprocessing.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier by loading the trained model pipeline.
        
        Args:
            model_path: Path to the saved model pickle file.
                       Defaults to the tuned logistic regression model.
        """
        if model_path is None:
            # Default to the tuned best model
            base_dir = Path(__file__).parent.parent / "arxiv_8class_60k.parquet" / "models"
            model_path = base_dir / "tuned_best_model_logistic_regression.pkl"
        
        self.model_path = Path(model_path)
        self.model = None
        self.classes = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model pipeline from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}\n"
                f"Please ensure the model has been trained and saved."
            )
        
        try:
            self.model = joblib.load(self.model_path)
            self.classes = self.model.classes_
            print(f"✓ Loaded ML model from: {self.model_path}")
            print(f"  Model type: {type(self.model.named_steps['clf']).__name__}")
            print(f"  Categories: {', '.join(self.classes)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, text: str) -> str:
        """
        Predict the category for a single abstract.
        
        Args:
            text: The abstract text (can include title)
        
        Returns:
            Predicted category as a string
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        prediction = self.model.predict([text])[0]
        return prediction
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get probability scores for all categories.
        
        Args:
            text: The abstract text (can include title)
        
        Returns:
            Dictionary mapping category names to probability scores
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        probabilities = self.model.predict_proba([text])[0]
        return {cls: float(prob) for cls, prob in zip(self.classes, probabilities)}
    
    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k predictions with confidence scores.
        
        Args:
            text: The abstract text (can include title)
            k: Number of top predictions to return
        
        Returns:
            List of (category, probability) tuples, sorted by probability (descending)
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        probabilities = self.model.predict_proba([text])[0]
        
        # Get top k indices
        top_k_idx = probabilities.argsort()[-k:][::-1]
        top_k_classes = self.classes[top_k_idx]
        top_k_probs = probabilities[top_k_idx]
        
        return [(cls, float(prob)) for cls, prob in zip(top_k_classes, top_k_probs)]
    
    def classify_with_details(self, text: str, top_k: int = 3) -> Dict:
        """
        Comprehensive classification with all details.
        
        Args:
            text: The abstract text (can include title)
            top_k: Number of top predictions to include
        
        Returns:
            Dictionary containing:
                - predicted_category: The top prediction
                - confidence: Confidence score for top prediction
                - top_predictions: List of (category, score) tuples
                - all_probabilities: Full probability distribution
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Get prediction and probabilities
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        
        # Get top k
        top_k_idx = probabilities.argsort()[-top_k:][::-1]
        top_k_classes = self.classes[top_k_idx]
        top_k_probs = probabilities[top_k_idx]
        
        return {
            'predicted_category': prediction,
            'confidence': float(probabilities[self.classes == prediction][0]),
            'top_predictions': [
                {'category': cls, 'probability': float(prob)} 
                for cls, prob in zip(top_k_classes, top_k_probs)
            ],
            'all_probabilities': {
                cls: float(prob) for cls, prob in zip(self.classes, probabilities)
            }
        }


# Convenience function for quick predictions
def classify_abstract(text: str, model_path: str = None) -> str:
    """
    Quick classification function.
    
    Args:
        text: Abstract text to classify
        model_path: Optional path to model file
    
    Returns:
        Predicted category
    """
    classifier = PaperClassifier(model_path)
    return classifier.predict(text)
