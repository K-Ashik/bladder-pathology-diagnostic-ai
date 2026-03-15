"""
Assign Diagnostic Categories
Label articles based on their source query and content
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("🏷️  Assigning Diagnostic Categories")
print("=" * 80)
print()

# Load dataset with features
df = pd.read_csv("dataset_with_features.csv")

print(f"Loading {len(df)} articles with features...")
print()

def assign_category(row):
    """
    Assign diagnostic category based on:
    1. Source query
    2. Morphological features
    3. Keywords in title/abstract
    """
    
    source_query = str(row['source_query']).lower()
    abstract = str(row['abstract']).lower() if pd.notna(row['abstract']) else ""
    title = str(row['title']).lower() if pd.notna(row['title']) else ""
    
    # Get feature values
    has_michaelis = row['michaelis_gutmann']
    has_von_hansemann = row['von_hansemann']
    has_terminal_spine = row['terminal_spine']
    has_egg = row['egg']
    has_foam_cells = row['foam_cells']
    has_granuloma = row['granuloma']
    
    # Category 1: MALAKOPLAKIA
    if 'malakoplakia' in source_query:
        if has_michaelis or has_von_hansemann or has_foam_cells:
            return 'malakoplakia'
        elif has_granuloma and not has_egg:
            return 'malakoplakia'
        elif 'malakoplakia' in abstract or 'malakoplakia' in title:
            return 'malakoplakia'
    
    # Category 2: SCHISTOSOMIASIS
    if 'schistosomiasis' in source_query:
        if has_terminal_spine or has_egg:
            return 'schistosomiasis'
        elif 'schistosomiasis' in abstract or 'schistosomiasis' in title:
            return 'schistosomiasis'
    
    # Category 3: PARASITIC BLADDER (non-schistosomiasis parasites)
    if 'parasitic' in source_query:
        if not has_terminal_spine and not has_michaelis:
            return 'parasitic_bladder'
    
    # Category 4: ATYPICAL MALAKOPLAKIA (granuloma without classic features)
    if 'malakoplakia' in source_query and has_granuloma and not has_michaelis:
        return 'atypical_malakoplakia'
    
    # Category 5: DIFFERENTIAL DIAGNOSIS (general bladder pathology)
    if 'differential' in source_query or 'bladder_pathology' in source_query:
        return 'differential_diagnosis'
    
    # Secondary classification if source_query doesn't help
    if has_michaelis or has_von_hansemann:
        return 'malakoplakia'
    elif has_terminal_spine or has_egg:
        return 'schistosomiasis'
    elif has_granuloma and has_foam_cells:
        return 'atypical_malakoplakia'
    elif has_granuloma:
        return 'differential_diagnosis'
    else:
        return 'differential_diagnosis'

# Assign categories
print("Assigning diagnostic categories...")
df['category'] = df.apply(assign_category, axis=1)

# Save
df.to_csv("dataset_with_features.csv", index=False)
print("✅ Saved to dataset_with_features.csv")
print()

# Show category distribution
print("=" * 80)
print("📊 DIAGNOSTIC CATEGORY DISTRIBUTION")
print("=" * 80)
print()

category_counts = df['category'].value_counts()
print(f"{'Category':<30} {'Count':<10} {'Percentage':<15}")
print("-" * 80)

for category, count in category_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{category:<30} {count:<10} {percentage:>6.1f}%")

print()
print(f"Total articles categorized: {len(df)}")
print()

# Check for any uncategorized
if df['category'].isna().any():
    print(f"⚠️  Warning: {df['category'].isna().sum()} articles without category")
else:
    print("✅ All articles categorized!")

print()

# Show sample articles per category
print("=" * 80)
print("📋 SAMPLE ARTICLES PER CATEGORY")
print("=" * 80)
print()

for category in sorted(df['category'].unique()):
    sample = df[df['category'] == category].iloc[0]
    print(f"Category: {category.upper()}")
    print(f"  PMID: {sample['pmid']}")
    print(f"  Title: {sample['title'][:70]}...")
    print()

print("=" * 80)
print("✅ CATEGORIZATION COMPLETE")
print("=" * 80)
print()