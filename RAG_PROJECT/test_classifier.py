"""
Test script to verify ML classifier integration
"""

import sys
sys.path.append('/home/sunbeam/STUDY_NEW/PROJECT/RAG_PROJECT')

from ml_classifier import PaperClassifier

print("=" * 80)
print("TESTING ML CLASSIFIER")
print("=" * 80)

# Initialize classifier
print("\n1. Initializing classifier...")
try:
    classifier = PaperClassifier()
    print("✓ Classifier initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize classifier: {e}")
    sys.exit(1)

# Test examples
test_examples = [
    {
        'text': 'Deep Learning for Computer Vision. We present a novel convolutional neural network architecture for image classification tasks.',
        'expected': 'cs'
    },
    {
        'text': 'Quantum Entanglement in Particle Physics. This paper explores quantum entanglement phenomena in high-energy particle collisions.',
        'expected': 'physics'
    },
    {
        'text': 'Statistical Analysis of Economic Growth. We perform a comprehensive statistical study of GDP growth patterns across countries.',
        'expected': 'stat'
    }
]

print("\n2. Testing predictions...")
for i, example in enumerate(test_examples, 1):
    print(f"\n{'-' * 80}")
    print(f"Test {i}")
    print(f"{'-' * 80}")
    print(f"Text: {example['text'][:100]}...")
    
    try:
        # Get detailed results
        results = classifier.classify_with_details(example['text'], top_k=3)
        
        print(f"\nPredicted: {results['predicted_category'].upper()}")
        print(f"Confidence: {results['confidence']*100:.1f}%")
        print(f"\nTop 3 predictions:")
        for j, pred in enumerate(results['top_predictions'], 1):
            print(f"  {j}. {pred['category']:12} {pred['probability']*100:5.1f}%")
        
        # Check if prediction matches expected
        if results['predicted_category'] == example['expected']:
            print(f"\n✓ Correct prediction!")
        else:
            print(f"\n⚠ Expected {example['expected']}, got {results['predicted_category']}")
            
    except Exception as e:
        print(f"✗ Prediction failed: {e}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)
