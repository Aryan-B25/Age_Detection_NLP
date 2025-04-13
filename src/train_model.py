import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

# Load preprocessed data
df = pd.read_csv('/home/ubuntu/nlp_age_detection/data/processed/preprocessed_data.csv')
print(f"Loaded preprocessed data with {df.shape[0]} samples")

# Use the processed text for model training
X = df['processed_text']
y = df['age_group']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Create a pipeline with TF-IDF vectorizer and classifier
# We'll try multiple classifiers to see which performs best

# 1. Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 2. Random Forest Pipeline
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 3. SVM Pipeline
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', SVC(probability=True, random_state=42))
])

# Train and evaluate each model
models = {
    'Logistic Regression': lr_pipeline,
    'Random Forest': rf_pipeline,
    'SVM': svm_pipeline
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'report': report
    }

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Plot confusion matrix for the best model
y_pred = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.savefig('/home/ubuntu/nlp_age_detection/models/confusion_matrix.png')
plt.close()

# Hyperparameter tuning for the best model
print("\nPerforming hyperparameter tuning for the best model...")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'tfidf__max_features': [3000, 5000, 7000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 1.0, 10.0]
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'tfidf__max_features': [3000, 5000, 7000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    }
else:  # SVM
    param_grid = {
        'tfidf__max_features': [3000, 5000, 7000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf']
    }

grid_search = GridSearchCV(best_model, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_report = classification_report(y_test, y_pred_tuned)

print(f"Tuned model accuracy: {tuned_accuracy:.4f}")
print(f"Classification Report:\n{tuned_report}")

# Plot confusion matrix for the tuned model
cm_tuned = confusion_matrix(y_test, y_pred_tuned)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues', xticklabels=tuned_model.classes_, yticklabels=tuned_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Tuned Model')
plt.tight_layout()
plt.savefig('/home/ubuntu/nlp_age_detection/models/tuned_confusion_matrix.png')
plt.close()

# Save the best model
print("\nSaving the best model...")
joblib.dump(tuned_model, '/home/ubuntu/nlp_age_detection/models/age_classifier_model.joblib')

# Feature importance analysis (if applicable)
if best_model_name == 'Random Forest':
    # Get feature names and importance scores
    feature_names = tuned_model.named_steps['tfidf'].get_feature_names_out()
    importances = tuned_model.named_steps['classifier'].feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(12, 8))
    plt.title('Top 20 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/nlp_age_detection/models/feature_importance.png')
    plt.close()

print("\nModel development and evaluation complete!")
