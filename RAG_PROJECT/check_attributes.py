import joblib
from pathlib import Path
import sklearn
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings to focus on output
warnings.filterwarnings("ignore")

model_path = "/home/sunbeam/STUDY_NEW/PROJECT/arxiv_8class_60k.parquet/models/tuned_best_model_logistic_regression.pkl"

try:
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    
    if isinstance(model, Pipeline):
        print("Pipeline contains steps:")
        for name, step in model.named_steps.items():
            print(f"  - {name}: {type(step)}")
            if hasattr(step, 'idf_'):
                print(f"    - idf_ attribute exists: True")
                print(f"    - idf_ length: {len(step.idf_)}")
            elif hasattr(step, 'vocabulary_'):
                print(f"    - vocabulary_ attribute exists: True")
                print(f"    - vocabulary_ size: {len(step.vocabulary_)}")
            
            # Check for TfidfTransformer within TfidfVectorizer if relevant
            if hasattr(step, 'transformer_'):
                print(f"    - transformer_ attribute exists: True")
                if hasattr(step.transformer_, 'idf_'):
                    print(f"      - transformer_.idf_ exists: True")

    # Manually check if it can transform
    if 'tfidf' in model.named_steps:
        tfidf = model.named_steps['tfidf']
        try:
            sample_transform = tfidf.transform(["test"])
            print("Successfully transformed sample text with tfidf step")
        except Exception as e:
            print(f"Failed to transform sample text with tfidf step: {e}")

except Exception as e:
    print(f"Error: {e}")
