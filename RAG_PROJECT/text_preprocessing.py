import re
import html

def clean_text(text):
    """
    Advanced text preprocessing for ArXiv abstracts.
    1. Normalizes whitespace (removes hard line breaks).
    2. Removes LaTeX math/commands.
    3. Removes citations and brackets.
    4. Normalizes hyphens and special chars.
    """
    if not isinstance(text, str):
        return ""

    # 1. Normalize line breaks and whitespace
    text = text.replace('\n', ' ')
    
    # 2. Remove LaTeX commands and math
    # Remove inline math $...$
    text = re.sub(r'\$.*?\$', ' ', text)
    # Remove specific LaTeX commands like \partial, \frac, \textit, etc.
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    # Remove LaTeX braces {} often left after command removal
    text = re.sub(r'[\{\}]', ' ', text)
    
    # 3. Remove Citations
    # Remove [1], [12], [Author et al.]
    text = re.sub(r'\[.*?\]', ' ', text)
    # Remove \cite{...} (if any parts survived previous steps)
    text = re.sub(r'\\cite\{.*?\}', ' ', text)

    # 4. Remove HTML entities if any
    text = html.unescape(text)
    
    # 5. Remove special characters but keep hyphens and alphanumeric
    # This keeps "non-linear", "co-engagement"
    text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
    
    # 6. Lowercase
    text = text.lower()
    
    # 7. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn Transformer wrapper for clean_text.
    Useful for including preprocessing in a Pipeline.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [clean_text(text) for text in X]

