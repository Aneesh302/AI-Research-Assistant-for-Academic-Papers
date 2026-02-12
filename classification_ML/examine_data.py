import pandas as pd
import numpy as np

# Read the parquet dataset from current directory (partitioned dataset)
import glob
parquet_files = glob.glob('*.parquet')
if parquet_files:
    df = pd.read_parquet(parquet_files)
else:
    # If we're in the directory, read from parent
    df = pd.read_parquet('../arxiv_8class_60k.parquet')

print('=' * 80)
print('DATASET OVERVIEW')
print('=' * 80)
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')

print('\n' + '=' * 80)
print('DATA TYPES')
print('=' * 80)
print(df.dtypes)

print('\n' + '=' * 80)
print('MISSING VALUES')
print('=' * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

print('\n' + '=' * 80)
print('TARGET VARIABLE: final_category')
print('=' * 80)
print(f'Unique values: {df["final_category"].nunique()}')
print('\nDistribution:')
print(df['final_category'].value_counts().sort_index())
print('\nPercentage distribution:')
print((df['final_category'].value_counts(normalize=True) * 100).sort_index())

print('\n' + '=' * 80)
print('SAMPLE DATA')
print('=' * 80)
# Show a few complete samples
for idx in range(3):
    print(f'\n--- Sample {idx + 1} ---')
    print(f"Title: {df.iloc[idx]['title']}")
    print(f"Abstract: {df.iloc[idx]['abstract'][:200]}...")
    print(f"Primary Category: {df.iloc[idx]['primary_category']}")
    print(f"Final Category: {df.iloc[idx]['final_category']}")

print('\n' + '=' * 80)
print('TEXT LENGTH STATISTICS')
print('=' * 80)
df['abstract_length'] = df['abstract'].str.len()
df['title_length'] = df['title'].str.len()
print('\nAbstract length:')
print(df['abstract_length'].describe())
print('\nTitle length:')
print(df['title_length'].describe())

print('\n' + '=' * 80)
print('SUMMARY')
print('=' * 80)
print(f"Total samples: {len(df)}")
print(f"Features: {len(df.columns)}")
print(f"Target: 'final_category' with {df['final_category'].nunique()} classes")
print(f"Main text features: 'abstract', 'title'")
print(f"Class balance: {'Balanced' if df['final_category'].value_counts().std() < df['final_category'].value_counts().mean() * 0.5 else 'Imbalanced'}")
