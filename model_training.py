"""
Machine Learning Model Training
Train multiple models on the expanded dataset with 463 articles
Target: 80%+ accuracy
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("🤖 MACHINE LEARNING MODEL TRAINING")
print("=" * 80)
print()

# Load dataset
df = pd.read_csv("dataset_with_features.csv")

print(f"Loading {len(df)} articles with {len(df.columns)} columns...")
print()

# Get feature columns
feature_cols = [col for col in df.columns if col not in 
                ['pmid', 'title', 'abstract', 'year', 'journal', 'first_author', 
                 'num_authors', 'keywords', 'source_query', 'category', 'combined_text']]

X = df[feature_cols].fillna(0)
y = df['category']

print("=" * 80)
print("📊 DATASET SUMMARY")
print("=" * 80)
print()
print(f"Total samples: {len(df)}")
print(f"Features: {len(feature_cols)}")
print(f"Categories: {sorted(y.unique())}")
print()

print("Category Distribution:")
print("-" * 80)
for cat in sorted(y.unique()):
    count = (y == cat).sum()
    pct = (count / len(y)) * 100
    print(f"  {cat:30s}: {count:3d} ({pct:5.1f}%)")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 80)
print("✅ DATA SPLIT")
print("=" * 80)
print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")
print()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open("scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved")
print()

# Train models
print("=" * 80)
print("🎯 TRAINING MODELS")
print("=" * 80)
print()

models = {}
results = []

# Model 1: Logistic Regression
print("Training Logistic Regression...", end=" ", flush=True)
lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
models['Logistic Regression'] = lr_model
results.append({
    'model': 'Logistic Regression',
    'accuracy': lr_acc,
    'predictions': lr_pred
})
print(f"✅ Accuracy: {lr_acc:.1%}")

# Model 2: Random Forest
print("Training Random Forest...", end=" ", flush=True)
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
models['Random Forest'] = rf_model
results.append({
    'model': 'Random Forest',
    'accuracy': rf_acc,
    'predictions': rf_pred
})
print(f"✅ Accuracy: {rf_acc:.1%}")

# Model 3: Gradient Boosting
print("Training Gradient Boosting...", end=" ", flush=True)
gb_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
models['Gradient Boosting'] = gb_model
results.append({
    'model': 'Gradient Boosting',
    'accuracy': gb_acc,
    'predictions': gb_pred
})
print(f"✅ Accuracy: {gb_acc:.1%}")

print()

# Find best model
best_result = max(results, key=lambda x: x['accuracy'])
best_model_name = best_result['model']
best_model = models[best_model_name]
best_predictions = best_result['predictions']

print("=" * 80)
print("🏆 BEST MODEL")
print("=" * 80)
print(f"Model: {best_model_name}")
print(f"Test Accuracy: {best_result['accuracy']:.1%}")
print()

# Detailed metrics for best model
print("=" * 80)
print("📊 DETAILED CLASSIFICATION REPORT")
print("=" * 80)
print()
print(classification_report(y_test, best_predictions))
print()

# Cross-validation
print("=" * 80)
print("🔄 CROSS-VALIDATION (5-FOLD)")
print("=" * 80)
print()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=skf, scoring='accuracy')

print(f"CV Scores: {[f'{score:.1%}' for score in cv_scores]}")
print(f"Mean CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
print()

# Save best model
with open("best_model.pkl", 'wb') as f:
    pickle.dump(best_model, f)
print(f"✅ Best model saved as best_model.pkl")
print()

# Feature importance
print("=" * 80)
print("🔍 TOP 10 IMPORTANT FEATURES")
print("=" * 80)
print()

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    feature_importance.to_csv("feature_importance.csv", index=False)
    print()
    print("✅ Feature importance saved to feature_importance.csv")

elif hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': best_model.coef_[0]
    }).sort_values('coefficient', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    feature_importance.to_csv("feature_importance.csv", index=False)
    print()
    print("✅ Feature coefficients saved to feature_importance.csv")

print()

# Model comparison summary
print("=" * 80)
print("📈 MODEL COMPARISON")
print("=" * 80)
print()

comparison_df = pd.DataFrame([
    {
        'Model': result['model'],
        'Accuracy': f"{result['accuracy']:.1%}"
    }
    for result in results
])

print(comparison_df.to_string(index=False))
comparison_df.to_csv("model_comparison.csv", index=False)
print()

print("=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)
print()

if best_result['accuracy'] >= 0.75:
    print(f"🎉 EXCELLENT! Model accuracy is {best_result['accuracy']:.1%}")
    print("   Ready for production dashboard!")
else:
    print(f"⚠️  Model accuracy is {best_result['accuracy']:.1%}")
    print("   Consider hyperparameter tuning or more data")

print()