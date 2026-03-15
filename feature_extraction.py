"""
Feature Extraction
Extract morphological features from article abstracts
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("🔬 Morphological Feature Extraction")
print("=" * 80)
print()

# Load dataset
df = pd.read_csv("dataset.csv")

print(f"Loading {len(df)} articles...")
print()

# Define comprehensive morphological features
morphology_keywords = {
    'michaelis_gutmann': ['michaelis-gutmann', 'michaelis gutmann', 'mg bodies', 'mg body'],
    'granuloma': ['granuloma', 'granulomatous', 'granulomatosis'],
    'von_hansemann': ['von hansemann', 'von hanseman', 'histiocytes'],
    'terminal_spine': ['terminal spine', 'terminal spined egg', 'terminal spined'],
    'egg': ['egg', 'eggs', 'ova', 'ovum', 'schistosome egg'],
    'inflammation': ['inflammation', 'inflammatory', 'inflamed', 'inflammatory response'],
    'ulceration': ['ulceration', 'ulcerated', 'ulcerative', 'ulcer'],
    'fibrosis': ['fibrosis', 'fibrotic', 'scarring', 'fibrous'],
    'calcification': ['calcification', 'calcified', 'calcify', 'calcified egg'],
    'cystitis': ['cystitis', 'cystitic'],
    'necrosis': ['necrosis', 'necrotic', 'necrotic debris', 'necrotic material'],
    'foam_cells': ['foam cell', 'foamy', 'lipid-laden', 'foamy macrophage'],
    'eosinophil': ['eosinophil', 'eosinophilic', 'eosinophilia'],
    'lamination': ['laminated', 'lamination', 'concentric', 'laminated material'],
    'central_material': ['central material', 'central core', 'granular center', 'central debris'],
}

def extract_features(text):
    """Extract morphological features from text"""
    if pd.isna(text):
        text = ""
    
    text_lower = text.lower()
    features = {}
    
    for feature, keywords in morphology_keywords.items():
        found = 0
        for keyword in keywords:
            if keyword in text_lower:
                found = 1
                break
        features[feature] = found
    
    return features

# Extract features for each article
print("Extracting morphological features from abstracts...")
print("(This may take a minute...)")
print()

feature_list = []

for idx, row in df.iterrows():
    features = extract_features(row['abstract'])
    features['pmid'] = row['pmid']
    feature_list.append(features)
    
    if (idx + 1) % 100 == 0:
        print(f"  ✅ Processed {idx + 1}/{len(df)} articles")

if len(df) % 100 != 0:
    print(f"  ✅ Processed {len(df)}/{len(df)} articles")

print()

# Create features dataframe
features_df = pd.DataFrame(feature_list)

# Merge with original data
df_with_features = df.merge(features_df, on='pmid')

# Save
df_with_features.to_csv("dataset_with_features.csv", index=False)
print(f"✅ Saved to dataset_with_features.csv")
print()

# Show summary statistics
print("=" * 80)
print("📊 MORPHOLOGICAL FEATURES SUMMARY")
print("=" * 80)
print()

feature_cols = [col for col in features_df.columns if col not in ['pmid']]

print("Feature Presence Across Dataset:")
print("-" * 80)
print(f"{'Feature':<25} {'Count':<10} {'Percentage':<15}")
print("-" * 80)

for feature in sorted(feature_cols):
    count = features_df[feature].sum()
    percentage = (count / len(features_df)) * 100
    print(f"{feature:<25} {count:<10} {percentage:>6.1f}%")

print()
print("=" * 80)
print("✅ Feature extraction complete!")
print("=" * 80)
print()
print(f"Total articles processed: {len(df_with_features)}")
print(f"Total features extracted: {len(feature_cols)}")
print()