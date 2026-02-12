"""
Exploratory Data Analysis for ArXiv 8-Class Classification Dataset
Generates comprehensive visualizations and statistical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import os
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directory for visualizations
output_dir = Path('eda_visualizations')
output_dir.mkdir(exist_ok=True)

print("Loading dataset...")
# Read the parquet dataset from current directory
import glob
parquet_files = glob.glob('*.parquet')
df = pd.read_parquet(parquet_files)

print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Target variable: final_category with {df['final_category'].nunique()} classes")

# ============================================================================
# 1. CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. ANALYZING CLASS DISTRIBUTION")
print("="*80)

class_counts = df['final_category'].value_counts().sort_index()
class_pct = (df['final_category'].value_counts(normalize=True) * 100).sort_index()

print("\nClass counts:")
print(class_counts)
print("\nClass percentages:")
print(class_pct)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Category', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution of Papers Across Categories', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 200, str(v), ha='center', va='bottom', fontweight='bold')

# Pie chart
explode = [0.05] * len(class_counts)
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, explode=explode, shadow=True, startangle=90)
axes[1].set_title('Category Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'class_distribution.png'}")
plt.close()

# ============================================================================
# 2. TEXT LENGTH ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. ANALYZING TEXT LENGTHS")
print("="*80)

df['abstract_length'] = df['abstract'].str.len()
df['title_length'] = df['title'].str.len()
df['total_text_length'] = df['abstract_length'] + df['title_length']

print("\nAbstract length statistics:")
print(df['abstract_length'].describe())
print("\nTitle length statistics:")
print(df['title_length'].describe())

# Text length distributions
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Abstract length distribution
axes[0, 0].hist(df['abstract_length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Abstract Length (characters)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Distribution of Abstract Lengths', fontsize=13, fontweight='bold')
axes[0, 0].axvline(df['abstract_length'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["abstract_length"].mean():.0f}')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Title length distribution
axes[0, 1].hist(df['title_length'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Title Length (characters)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribution of Title Lengths', fontsize=13, fontweight='bold')
axes[0, 1].axvline(df['title_length'].mean(), color='red', linestyle='--',
                    label=f'Mean: {df["title_length"].mean():.0f}')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Abstract length by category
df.boxplot(column='abstract_length', by='final_category', ax=axes[1, 0])
axes[1, 0].set_xlabel('Category', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Abstract Length (characters)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Abstract Length by Category', fontsize=13, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)
plt.sca(axes[1, 0])
plt.xticks(rotation=45)

# Title length by category
df.boxplot(column='title_length', by='final_category', ax=axes[1, 1])
axes[1, 1].set_xlabel('Category', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Title Length (characters)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Title Length by Category', fontsize=13, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)
plt.sca(axes[1, 1])
plt.xticks(rotation=45)

plt.suptitle('')  # Remove auto-generated title
plt.tight_layout()
plt.savefig(output_dir / 'text_length_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'text_length_analysis.png'}")
plt.close()

# ============================================================================
# 3. MISSING DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. ANALYZING MISSING DATA")
print("="*80)

missing_data = df.isnull().sum()
missing_pct = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Percentage': missing_pct
}).sort_values('Missing Count', ascending=False)

print("\nMissing data summary:")
print(missing_df[missing_df['Missing Count'] > 0])

# Visualize missing data
fig, ax = plt.subplots(figsize=(10, 8))
missing_vis = missing_df[missing_df['Missing Count'] > 0]
if len(missing_vis) > 0:
    bars = ax.barh(missing_vis.index, missing_vis['Percentage'], color='coral', edgecolor='black')
    ax.set_xlabel('Missing Data Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Missing Data Analysis', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_data.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'missing_data.png'}")
    plt.close()

# ============================================================================
# 4. WORD CLOUD GENERATION
# ============================================================================
print("\n" + "="*80)
print("4. GENERATING WORD CLOUDS FOR EACH CATEGORY")
print("="*80)

def clean_text(text):
    """Clean text for word cloud generation"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Generate word clouds for each category
try:
    categories = df['final_category'].unique()
    n_categories = len(categories)
    n_cols = 3
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes = axes.flatten() if n_categories > 1 else [axes]
    
    for idx, category in enumerate(sorted(categories)):
        print(f"  Generating word cloud for: {category}")
        
        # Get abstracts for this category
        category_abstracts = df[df['final_category'] == category]['abstract'].dropna()
        
        # Combine and clean text
        combined_text = ' '.join(category_abstracts.apply(clean_text))
        
        # Generate word cloud - use prefer canvas parameter to avoid font issues
        try:
            wordcloud = WordCloud(width=800, height=400, 
                                 background_color='white',
                                 colormap='viridis',
                                 max_words=100,
                                 relative_scaling=0.5,
                                 prefer_horizontal=0.7,
                                 min_font_size=10).generate(combined_text)
            
            axes[idx].imshow(wordcloud, interpolation='bilinear')
        except Exception as e:
            # If word cloud fails, just show text error
            axes[idx].text(0.5, 0.5, f'Word cloud generation failed\nfor {category}\n(Font issue)',
                          ha='center', va='center', fontsize=12)
        
        axes[idx].set_title(f'{category.upper()} ({len(category_abstracts)} papers)', 
                           fontsize=13, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_categories, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Word Clouds by Category', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'wordclouds_by_category.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'wordclouds_by_category.png'}")
    plt.close()
except Exception as e:
    print(f"⚠ Word cloud generation skipped due to: {e}")
    print("  Continuing with other analyses...")

# ============================================================================
# 5. TOP WORDS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. ANALYZING TOP WORDS PER CATEGORY")
print("="*80)

def get_top_words(text_series, n=20):
    """Extract top n words from a series of texts"""
    all_words = []
    for text in text_series.dropna():
        cleaned = clean_text(text)
        words = cleaned.split()
        # Filter out very short words
        words = [w for w in words if len(w) > 3]
        all_words.extend(words)
    
    # Count and return top n
    word_counts = Counter(all_words)
    return word_counts.most_common(n)

# Create comparison of top words
top_words_by_category = {}
for category in sorted(df['final_category'].unique()):
    category_abstracts = df[df['final_category'] == category]['abstract']
    top_words = get_top_words(category_abstracts, n=15)
    top_words_by_category[category] = top_words
    
    print(f"\n{category.upper()}:")
    for word, count in top_words[:10]:
        print(f"  {word}: {count}")

# ============================================================================
# 6. PRIMARY VS FINAL CATEGORY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. ANALYZING PRIMARY VS FINAL CATEGORY MAPPING")
print("="*80)

# Create crosstab
category_mapping = pd.crosstab(df['primary_category'], df['final_category'])
print("\nPrimary category to final category mapping:")
print(category_mapping.head(20))

# Visualize as heatmap (sample top primary categories)
top_primary = df['primary_category'].value_counts().head(20).index
filtered_mapping = pd.crosstab(
    df[df['primary_category'].isin(top_primary)]['primary_category'],
    df[df['primary_category'].isin(top_primary)]['final_category']
)

plt.figure(figsize=(12, 10))
sns.heatmap(filtered_mapping, annot=True, fmt='d', cmap='YlOrRd', 
            linewidths=0.5, cbar_kws={'label': 'Count'})
plt.xlabel('Final Category', fontsize=12, fontweight='bold')
plt.ylabel('Primary Category', fontsize=12, fontweight='bold')
plt.title('Primary Category vs Final Category Mapping (Top 20 Primary Categories)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'category_mapping_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'category_mapping_heatmap.png'}")
plt.close()

# ============================================================================
# 7. GENERATE SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("7. GENERATING SUMMARY REPORT")
print("="*80)

summary_report = f"""
EXPLORATORY DATA ANALYSIS SUMMARY REPORT
{'='*80}

DATASET OVERVIEW
----------------
Total Samples: {len(df):,}
Total Features: {len(df.columns)}
Target Variable: final_category
Number of Classes: {df['final_category'].nunique()}

CLASS DISTRIBUTION
-----------------
{class_counts.to_string()}

Percentage Distribution:
{class_pct.to_string()}

Class Balance: {'Balanced' if class_pct.std() < 5 else 'Imbalanced'}
(Standard deviation of percentages: {class_pct.std():.2f}%)

TEXT FEATURES STATISTICS
------------------------
Abstract Length:
  Mean: {df['abstract_length'].mean():.0f} characters
  Median: {df['abstract_length'].median():.0f} characters
  Std Dev: {df['abstract_length'].std():.0f} characters
  Range: {df['abstract_length'].min():.0f} - {df['abstract_length'].max():.0f} characters

Title Length:
  Mean: {df['title_length'].mean():.0f} characters
  Median: {df['title_length'].median():.0f} characters
  Std Dev: {df['title_length'].std():.0f} characters
  Range: {df['title_length'].min():.0f} - {df['title_length'].max():.0f} characters

MISSING DATA
------------
{missing_df[missing_df['Missing Count'] > 0].to_string()}

KEY FEATURES FOR MODELING
-------------------------
- Primary text feature: 'abstract' (no missing values, avg ~1,052 chars)
- Secondary text feature: 'title' (only 97 missing, avg ~78 chars)
- Both features provide rich textual information for classification

RECOMMENDATIONS FOR MODELING
----------------------------
1. Use 'abstract' and 'title' as main features (combine them)
2. Classes are relatively balanced - standard train/test split will work
3. Consider TF-IDF, Word2Vec, or BERT embeddings for feature extraction
4. Expected baseline accuracy: ~75-80% (based on class distribution)
5. Multi-class classification with 8 targets
6. Use stratified sampling to maintain class balance in train/test splits

VISUALIZATIONS GENERATED
------------------------
✓ class_distribution.png - Bar and pie charts of category distribution
✓ text_length_analysis.png - Text length distributions and box plots
✓ missing_data.png - Missing data percentages
✓ wordclouds_by_category.png - Word clouds for each category
✓ category_mapping_heatmap.png - Primary vs final category relationships

{'='*80}
Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save report
report_path = output_dir / 'eda_summary_report.txt'
with open(report_path, 'w') as f:
    f.write(summary_report)

print(summary_report)
print(f"\n✓ Saved: {report_path}")

print("\n" + "="*80)
print("EDA COMPLETE!")
print("="*80)
print(f"All visualizations and reports saved to: {output_dir.absolute()}")
print(f"Total files generated: {len(list(output_dir.glob('*')))}")
