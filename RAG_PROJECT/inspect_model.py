import joblib
from pathlib import Path
import sklearn
from sklearn.pipeline import Pipeline

model_path = "/home/sunbeam/STUDY_NEW/PROJECT/arxiv_8class_60k.parquet/models/tuned_best_model_logistic_regression.pkl"

try:
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    if isinstance(model, Pipeline):
        print("Pipeline steps:")
        for name, step in model.named_steps.items():
            print(f"  - {name}: {type(step)}")
            if hasattr(step, 'idf_'):
                print(f"    - idf_ fitted: {True}")
            elif hasattr(step, 'vocabulary_'):
                 print(f"    - vocabulary_ fitted: {True}")
            
    if hasattr(model, 'classes_'):
        print(f"Classes: {model.classes_}")
    
    # Try a dummy prediction
    test_text = "This is a test abstract about computer science and machine learning."
    pred = model.predict([test_text])
    print(f"Prediction: {pred}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
