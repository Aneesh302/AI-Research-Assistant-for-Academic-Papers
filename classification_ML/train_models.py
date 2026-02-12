"""
Machine Learning Classification Model Training for ArXiv Dataset
Trains multiple models and evaluates their performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import glob
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠ XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

print("="*80)
print("ARXIV MULTI-CLASS CLASSIFICATION PROJECT")
print("="*80)

# Create output directories
models_dir = Path('models')
results_dir = Path('results')
models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "="*80)
print("1. LOADING AND PREPARING DATA")
print("="*80)

# Load dataset
parquet_files = glob.glob('*.parquet')
df = pd.read_parquet(parquet_files)
print(f"✓ Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")

# Handle missing values in text columns
df['title'] = df['title'].fillna('')
df['abstract'] = df['abstract'].fillna('')

# Combine title and abstract for richer features
df['combined_text'] = df['title'] + ' ' + df['abstract']
print(f"✓ Combined title and abstract into single feature")

# Prepare features and target
X = df['combined_text']
y = df['final_category']

print(f"\nFeature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Number of classes: {y.nunique()}")
print(f"Classes: {sorted(y.unique())}")

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("2. SPLITTING DATA")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# Verify class balance
print("\nClass distribution in training set:")
print(y_train.value_counts().sort_index())

# ============================================================================
# 3. FEATURE EXTRACTION & PIPELINE SETUP
# ============================================================================
print("\n" + "="*80)
print("3. SETTING UP PREPROCESSING PIPELINE")
print("="*80)

from sklearn.pipeline import Pipeline
try:
    from text_preprocessing import TextCleaner
except ImportError:
    import sys
    sys.path.append('.')
    from text_preprocessing import TextCleaner

# Define base pipeline components (without classifier)
# We will append classifier in the loop
def create_pipeline(classifier):
    return Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.8,
            strip_accents='unicode',
            stop_words='english'
        )),
        ('clf', classifier)
    ])

print("Pipeline created with:")
print("  - TextCleaning (LaTeX removal, normalization)")
print("  - TfidfVectorizer (10k features, bigrams)")
print("  - Classifier")

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================
print("\n" + "="*80)
print("4. TRAINING CLASSIFICATION MODELS")
print("="*80)

# Define models (base estimators)
base_models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=0
    ),
    'Linear SVM': LinearSVC(
        max_iter=1000,
        C=1.0,
        random_state=42,
        verbose=0
    ),
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    base_models['XGBoost'] = XGBClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

results = []
models = {} # Store trained pipelines

for name, clf in base_models.items():
    print(f"\n{'─'*80}")
    print(f"Training: {name}")
    print(f"{'─'*80}")
    
    # Create valid pipeline
    model_pipeline = create_pipeline(clf)
    
    try:
        # Train (Pipeline handles transformation)
        print("  Fitting model pipeline...")
        model_pipeline.fit(X_train, y_train)
        
        # Predict
        print("  Making predictions...")
        # Note: X_test is raw text, pipeline handles it
        y_pred = model_pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n  ✓ Results:")
        print(f"     Accuracy:  {accuracy:.4f}")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall:    {recall:.4f}")
        print(f"     F1-Score:  {f1:.4f}")
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Save model pipeline
        model_path = models_dir / f'{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model_pipeline, model_path)
        print(f"  ✓ Saved model pipeline to: {model_path}")
        
        models[name] = model_pipeline # Keep for detailed eval
        
    except Exception as e:
        print(f"  ✗ Error training {name}: {str(e)}")
        continue


# ============================================================================
# 5. COMPARE MODELS
# ============================================================================
print("\n" + "="*80)
print("5. MODEL COMPARISON")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False)

print("\nModel Performance Summary (sorted by F1-Score):")
print(results_df.to_string(index=False))

# Save results
results_path = results_dir / 'model_comparison.csv'
results_df.to_csv(results_path, index=False)
print(f"\n✓ Saved results to: {results_path}")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: All metrics comparison
x = np.arange(len(results_df))
width = 0.2

axes[0].bar(x - 1.5*width, results_df['Accuracy'], width, label='Accuracy', alpha=0.8)
axes[0].bar(x - 0.5*width, results_df['Precision'], width, label='Precision', alpha=0.8)
axes[0].bar(x + 0.5*width, results_df['Recall'], width, label='Recall', alpha=0.8)
axes[0].bar(x + 1.5*width, results_df['F1-Score'], width, label='F1-Score', alpha=0.8)

axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1.1])

# Plot 2: F1-Score ranking
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_df)))
axes[1].barh(results_df['Model'], results_df['F1-Score'], color=colors, edgecolor='black')
axes[1].set_xlabel('F1-Score', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Model', fontsize=12, fontweight='bold')
axes[1].set_title('F1-Score Ranking', fontsize=14, fontweight='bold')
axes[1].set_xlim([0, 1])
axes[1].grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(results_df['F1-Score']):
    axes[1].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {results_dir / 'model_comparison.png'}")
plt.close()

# ============================================================================
# 6. DETAILED EVALUATION OF BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("6. DETAILED EVALUATION OF BEST MODEL")
print("="*80)

best_model_name = results_df.iloc[0]['Model']
best_f1 = results_df.iloc[0]['F1-Score']

print(f"\nBest Model: {best_model_name}")
print(f"F1-Score: {best_f1:.4f}")

# Load best model
best_model_path = models_dir / f'{best_model_name.lower().replace(" ", "_")}.pkl'
best_model = joblib.load(best_model_path)

# Get predictions
# Note: X_test is raw text, pipeline handles it
y_pred_best = best_model.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, zero_division=0))

# Save classification report
report = classification_report(y_test, y_pred_best, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_path = results_dir / 'classification_report_best_model.csv'
report_df.to_csv(report_path)
print(f"✓ Saved classification report to: {report_path}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
classes = sorted(y.unique())

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
plt.ylabel('True Category', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix - {best_model_name}\n(F1-Score: {best_f1:.4f})', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(results_dir / 'confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved confusion matrix to: {results_dir / 'confusion_matrix_best_model.png'}")
plt.close()

# Per-class performance
class_metrics = []
for cls in classes:
    cls_mask = y_test == cls
    cls_pred_mask = y_pred_best == cls
    
    tp = np.sum(cls_mask & cls_pred_mask)
    fp = np.sum(~cls_mask & cls_pred_mask)
    fn = np.sum(cls_mask & ~cls_pred_mask)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    class_metrics.append({
        'Class': cls,
        'Support': np.sum(cls_mask),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

class_metrics_df = pd.DataFrame(class_metrics)
print("\nPer-Class Performance:")
print(class_metrics_df.to_string(index=False))

# Visualize per-class performance
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(class_metrics_df))
width = 0.25

ax.bar(x - width, class_metrics_df['Precision'], width, label='Precision', alpha=0.8)
ax.bar(x, class_metrics_df['Recall'], width, label='Recall', alpha=0.8)
ax.bar(x + width, class_metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title(f'Per-Class Performance - {best_model_name}', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_metrics_df['Class'], rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig(results_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved per-class visualization to: {results_dir / 'per_class_performance.png'}")
plt.close()

# ============================================================================
# 7. GENERATE FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("7. GENERATING FINAL SUMMARY")
print("="*80)

# Get vocab size from best model pipeline
vocab_size = len(best_model.named_steps['tfidf'].vocabulary_)

summary = f"""
MACHINE LEARNING CLASSIFICATION - FINAL SUMMARY
{'='*80}

DATASET INFORMATION
-------------------
Total Samples: {len(df):,}
Training Samples: {len(X_train):,} (80%)
Test Samples: {len(X_test):,} (20%)
Number of Classes: {y.nunique()}
Classes: {', '.join(sorted(y.unique()))}

FEATURE EXTRACTION
------------------
Method: TF-IDF (Term Frequency-Inverse Document Frequency)
Preprocessing: Advanced Text Cleaning (LaTeX removal, normalization)
Features: Combined title + abstract
Vocabulary Size: {vocab_size:,}

MODELS TRAINED
--------------
{len(models)} models trained and evaluated:
{chr(10).join([f"  • {name}" for name in models.keys()])}

MODEL PERFORMANCE
-----------------
{results_df.to_string(index=False)}

BEST PERFORMING MODEL
---------------------
Model: {best_model_name}
Accuracy: {results_df.iloc[0]['Accuracy']:.4f}
Precision: {results_df.iloc[0]['Precision']:.4f}
Recall: {results_df.iloc[0]['Recall']:.4f}
F1-Score: {results_df.iloc[0]['F1-Score']:.4f}

FILES GENERATED
---------------
Models:
  ✓ {len(list(models_dir.glob('*.pkl')))} model files saved in '{models_dir}/'
  ✓ (Models are saved as complete Pipelines including preprocessing)

Results:
  ✓ model_comparison.csv - Performance metrics for all models
  ✓ model_comparison.png - Visual comparison of models
  ✓ classification_report_best_model.csv - Detailed per-class metrics
  ✓ confusion_matrix_best_model.png - Confusion matrix visualization
  ✓ per_class_performance.png - Per-class performance chart

DEPLOYMENT
----------
To use the trained model for predictions:
  1. Load pipeline: model = joblib.load('models/{best_model_name.lower().replace(" ", "_")}.pkl')
  2. Prepare text: text = "your title + your abstract"
  3. Predict: prediction = model.predict([text])
  (Note: The pipeline handles all preprocessing and vectorization automatically)

{'='*80}
Summary generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save summary
summary_path = results_dir / 'training_summary.txt'
with open(summary_path, 'w') as f:
    f.write(summary)

print(summary)
print(f"\n✓ Saved summary to: {summary_path}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Models saved in: {models_dir.absolute()}")
print(f"Results saved in: {results_dir.absolute()}")
