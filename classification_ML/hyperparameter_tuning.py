"""
Hyperparameter Tuning for ArXiv Classification Models
Optimizes parameters for the best performing models using GridSearchCV
"""

import pandas as pd
import numpy as np
import glob
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

print("="*80)
print("HYPERPARAMETER TUNING")
print("="*80)

# Create output directory
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

# 1. Load Data
print("\nLoading data...")
parquet_files = glob.glob('*.parquet')
df = pd.read_parquet(parquet_files)

# Combine text
df['title'] = df['title'].fillna('')
df['abstract'] = df['abstract'].fillna('')
df['combined_text'] = df['title'] + ' ' + df['abstract']

X = df['combined_text']
y = df['final_category']

# Use a smaller subset for tuning to speed up execution
# maintain class balance with stratify
# ============================================================================
# 1. SPLIT DATA STRATEGY
# ============================================================================
# Strategy:
# 1. Split total data into DEV (80%) and TEST (20%).
#    - TEST is locked away until the very end.
#    - DEV is used for Training and Cross-Validation.
# 2. Perform Grid Search with 3-Fold CV on the DEV set.
# 3. Select best model, Retrain on full DEV set.
# 4. Final Evaluation on TEST set.

print("\n" + "="*80)
print("1. DATA SPLITTING STRATEGY")
print("="*80)

# Initial Stratified Split: 80% Dev, 20% Test
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Total samples: {len(df)}")
print(f"Development Set (for Tuning & Training): {len(X_dev)} samples")
print(f"Held-out Test Set (for Final Evaluation): {len(X_test)} samples")

# Optional: Downsample DEV set for faster Grid Search execution if needed
# For production/final run, use the full X_dev.
# For rapid prototyping, we can take a subset. 
# Uncomment the line below to speed up tuning by using only 30% of Dev set
# X_dev_tune, _, y_dev_tune, _ = train_test_split(X_dev, y_dev, train_size=0.3, stratify=y_dev, random_state=42)
X_dev_tune, y_dev_tune = X_dev, y_dev # Default: Use full dev set

# ============================================================================
# 2. SETUP PIPELINE & GRID
# ============================================================================
print("\n" + "="*80)
print("2. HYPERPARAMETER TUNING (Cross-Validation)")
print("="*80)

# Import the custom cleaning function and class
try:
    from text_preprocessing import TextCleaner
except ImportError:
    import sys
    sys.path.append('.')
    from text_preprocessing import TextCleaner

pipeline = Pipeline([
    ('cleaner', TextCleaner()),
    ('tfidf', TfidfVectorizer(min_df=3, max_df=0.8, strip_accents='unicode', stop_words='english')),
    ('clf', LogisticRegression()) # Placeholder
])

# Define Parameter Grids
# We focus on Logistic Regression as it was the best baseline
param_grid_lr = {
    'clf': [LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)],
    'clf__C': [1.0, 10.0],
    'clf__solver': ['liblinear', 'saga'], 
    'tfidf__ngram_range': [(1, 2)],
    'tfidf__max_features': [10000, 20000]
}

grids = [('Logistic Regression', param_grid_lr)]

best_overall_score = 0
best_overall_model = None
best_overall_params = None
best_overall_name = ""

for name, param_grid in grids:
    print(f"\nScanning parameters for {name}...")
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3,                 # 3-Fold Cross Validation
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # Run Grid Search on DEV set (or subset)
    grid_search.fit(X_dev_tune, y_dev_tune)
    
    print(f"  Best CV Score: {grid_search.best_score_:.4f}")
    print(f"  Best Params: {grid_search.best_params_}")
    
    if grid_search.best_score_ > best_overall_score:
        best_overall_score = grid_search.best_score_
        best_overall_model = grid_search.best_estimator_
        best_overall_params = grid_search.best_params_
        best_overall_name = name

# ============================================================================
# 3. FINAL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("3. FINAL EVALUATION ON HELD-OUT TEST SET")
print("="*80)

print(f"Best Model Selected: {best_overall_name}")
print(f"Retraining best model on FULL Development Set ({len(X_dev)} samples)...")

# Retrain on full DEV set (in case we tuned on a subset)
best_overall_model.fit(X_dev, y_dev)

print("Predicting on Held-out Test Set...")
y_pred = best_overall_model.predict(X_test)

# Report
print("\nFinal Test Results:")
print(classification_report(y_test, y_pred))

# Save
tuned_model_path = models_dir / f'tuned_best_model_{best_overall_name.lower().replace(" ", "_")}.pkl'
joblib.dump(best_overall_model, tuned_model_path)
print(f"\n✓ Saved final tuned model to: {tuned_model_path}")

