"""
Test Script - Verify Model Predictions
Demonstrates how to use the trained model for predictions
"""

import joblib
import pandas as pd
import glob

print("="*80)
print("TESTING TRAINED MODEL FOR PREDICTIONS")
print("="*80)

# Load the best model and vectorizer
print("\nLoading model and vectorizer...")
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/logistic_regression.pkl')
print("✓ Loaded Logistic Regression model and TF-IDF vectorizer")

# Load test data to get some examples
parquet_files = glob.glob('*.parquet')
df = pd.read_parquet(parquet_files)

print(f"\n{'='*80}")
print("TESTING WITH SAMPLE PAPERS FROM DATASET")
print(f"{'='*80}")

# Test with random samples from each category
categories = sorted(df['final_category'].unique())
test_samples = []

for category in categories:
    sample = df[df['final_category'] == category].sample(n=1, random_state=42).iloc[0]
    test_samples.append({
        'true_category': category,
        'title': sample['title'],
        'abstract': sample['abstract'][:200] + '...',  # Truncate for display
        'full_text': sample['title'] + ' ' + sample['abstract']
    })

# Make predictions
for i, sample in enumerate(test_samples, 1):
    print(f"\n{'-'*80}")
    print(f"Test {i}: {sample['true_category'].upper()}")
    print(f"{'-'*80}")
    print(f"Title: {sample['title']}")
    print(f"Abstract: {sample['abstract']}")
    
    # Transform and predict
    features = vectorizer.transform([sample['full_text']])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get top 3 predictions
    top_3_idx = probabilities.argsort()[-3:][::-1]
    top_3_classes = model.classes_[top_3_idx]
    top_3_probs = probabilities[top_3_idx]
    
    print(f"\n  Predicted: {prediction.upper()}")
    print(f"  Actual: {sample['true_category'].upper()}")
    print(f"  Match: {'✓ CORRECT' if prediction == sample['true_category'] else '✗ INCORRECT'}")
    
    print(f"\n  Top 3 Predictions:")
    for j, (cls, prob) in enumerate(zip(top_3_classes, top_3_probs), 1):
        marker = "←" if cls == prediction else " "
        print(f"    {j}. {cls:12} {prob*100:5.1f}% {marker}")

print(f"\n{'='*80}")
print("TESTING WITH CUSTOM EXAMPLES")
print(f"{'='*80}")

# Custom test examples
custom_examples = [
    {
        'title': 'Deep Learning for Image Recognition',
        'abstract': 'We present a novel deep neural network architecture for image classification tasks. Our approach uses convolutional layers and achieves state-of-the-art performance on ImageNet.'
    },
    {
        'title': 'Quantum Entanglement in Particle Physics',
        'abstract': 'This paper explores quantum entanglement phenomena in high-energy particle collisions. We derive theoretical predictions and compare with experimental data from the Large Hadron Collider.'
    },
    {
        'title': 'Statistical Analysis of Economic Growth',
        'abstract': 'We perform a comprehensive statistical study of GDP growth patterns across 50 countries over 20 years. Our regression analysis identifies key economic indicators.'
    }
]

for i, example in enumerate(custom_examples, 1):
    print(f"\n{'-'*80}")
    print(f"Custom Example {i}")
    print(f"{'-'*80}")
    print(f"Title: {example['title']}")
    print(f"Abstract: {example['abstract']}")
    
    # Combine and predict
    full_text = example['title'] + ' ' + example['abstract']
    features = vectorizer.transform([full_text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get top 3 predictions
    top_3_idx = probabilities.argsort()[-3:][::-1]
    top_3_classes = model.classes_[top_3_idx]
    top_3_probs = probabilities[top_3_idx]
    
    print(f"\n  Predicted Category: {prediction.upper()}")
    print(f"\n  Confidence Scores:")
    for j, (cls, prob) in enumerate(zip(top_3_classes, top_3_probs), 1):
        marker = "←" if cls == prediction else " "
        print(f"    {j}. {cls:12} {prob*100:5.1f}% {marker}")

print(f"\n{'='*80}")
print("PREDICTION TESTING COMPLETE!")
print("="*80)
print("\nThe model successfully loads and makes predictions.")
print("You can use this script as a template for making predictions on new papers.")
