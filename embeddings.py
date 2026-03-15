"""
Create Semantic Embeddings
Generate embeddings for RAG (Retrieval Augmented Generation) system
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

print("=" * 80)
print("🧠 Creating Semantic Embeddings for RAG System")
print("=" * 80)
print()

# Load dataset
df = pd.read_csv("dataset_with_features.csv")

print(f"Loading {len(df)} articles...")
print()

# Load embedding model
print("Loading sentence-transformer model...")
print("(This downloads ~400MB on first run)")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model loaded")
print()

# Combine title and abstract for embedding
print("Creating text combinations...")
df['combined_text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
print("✅ Text prepared")
print()

# Create embeddings
print("Generating embeddings...")
print("(This may take 2-3 minutes...)")
print()

embeddings_list = []
batch_size = 50

for i in range(0, len(df), batch_size):
    batch = df['combined_text'].iloc[i:i+batch_size].tolist()
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    embeddings_list.extend(batch_embeddings)
    
    if (i + batch_size) % 100 == 0 or (i + len(batch)) == len(df):
        print(f"  ✅ Processed {min(i + batch_size, len(df))}/{len(df)} articles")

# Convert to numpy array
embeddings_array = np.array(embeddings_list)

print()
print(f"✅ Embeddings created: {embeddings_array.shape}")
print()

# Save embeddings
print("Saving embeddings...")
with open("embeddings.pkl", 'wb') as f:
    pickle.dump(embeddings_array, f)
print("✅ Saved embeddings.pkl")
print()

# Save metadata
metadata = df[['pmid', 'title', 'category', 'combined_text']].copy()
metadata.to_csv("embeddings_metadata.csv", index=False)
print("✅ Saved embeddings_metadata.csv")
print()

print("=" * 80)
print("✅ EMBEDDING GENERATION COMPLETE")
print("=" * 80)
print()

print(f"📊 STATISTICS:")
print(f"   Total articles: {len(df)}")
print(f"   Embedding dimension: {embeddings_array.shape[1]}")
print(f"   Memory used: {embeddings_array.nbytes / 1024 / 1024:.1f} MB")
print()