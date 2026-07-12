import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)

def train_model(X_train, y_train, model_params=None):
    """
    Trains an XGBoost classifier with given parameters.
    """
    if model_params is None:
        model_params = {'n_estimators': 100, 'max_depth': 6, 'random_state': 42, 'eval_metric': 'logloss'}
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, title="Model Performance"):
    """
    Evaluates the model and prints performance metrics, including a confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- {title} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['On-Time (0)', 'Late Risk (1)'],
                yticklabels=['On-Time (0)', 'Late Risk (1)'])
    plt.title(f'{title}\nConfusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def get_shap_values(model, X_data_sample):
    """
    Generates SHAP values for a given model and data sample.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_data_sample)

    if len(shap_values.values.shape) == 3:
        shap_values.values = shap_values.values[:, :, 1] # For binary classification (positive class)
    return shap_values

def plot_shap_summary(shap_values, X_data_sample, title="SHAP Feature Importance", max_display=20):
    """
    Plots a SHAP summary plot.
    """
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_data_sample, plot_type="bar", max_display=max_display, show=False)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def save_model_artifacts(model, preprocessor, extended_cat_preprocessor, lane_preprocessor, tfidf_vectorizer, feature_names, output_dir='production_artifacts'):
    """
    Saves the trained model and all transformers to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, 'champion_model.joblib'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'initial_preprocessor.joblib'))
    joblib.dump(extended_cat_preprocessor, os.path.join(output_dir, 'extended_cat_preprocessor.joblib'))
    joblib.dump(lane_preprocessor, os.path.join(output_dir, 'lane_preprocessor.joblib'))
    joblib.dump(tfidf_vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.joblib'))
    print(f"All model artifacts saved to {output_dir}")

def load_model_artifacts(output_dir='production_artifacts'):
    """
    Loads the trained model and all transformers from disk.
    """
    model = joblib.load(os.path.join(output_dir, 'champion_model.joblib'))
    preprocessor = joblib.load(os.path.join(output_dir, 'initial_preprocessor.joblib'))
    extended_cat_preprocessor = joblib.load(os.path.join(output_dir, 'extended_cat_preprocessor.joblib'))
    lane_preprocessor = joblib.load(os.path.join(output_dir, 'lane_preprocessor.joblib'))
    tfidf_vectorizer = joblib.load(os.path.join(output_dir, 'tfidf_vectorizer.joblib'))
    feature_names = joblib.load(os.path.join(output_dir, 'feature_names.joblib'))
    print(f"All model artifacts loaded from {output_dir}")
    return model, preprocessor, extended_cat_preprocessor, lane_preprocessor, tfidf_vectorizer, feature_names

def predict_and_explain(raw_data_point, preprocessor, extended_cat_preprocessor, lane_preprocessor, tfidf_vectorizer, champion_model, feature_names):
    """
    Preprocesses a new raw data point, makes a prediction, and provides SHAP explanations.
    """
    from src.preprocess import engineer_features, apply_transformers

    # Ensure raw_data_point is a DataFrame with same columns as original df
    # Create a dummy DataFrame with all necessary columns for the preprocessor
    # This part needs to be robust for single row predictions
    # Reconstruct original df-like structure for preprocessing compatibility
    dummy_df_cols = joblib.load(os.path.join('production_artifacts', 'feature_names_raw_df_at_fit.joblib'))
    temp_df = pd.DataFrame(columns=dummy_df_cols)
    raw_data_point_df = pd.concat([temp_df, pd.DataFrame([raw_data_point])], ignore_index=True, sort=False)

    # Fill missing values if any, to match the training data structure
    for col in dummy_df_cols:
        if col not in raw_data_point_df.columns: raw_data_point_df[col] = np.nan # Or appropriate default
    raw_data_point_df = raw_data_point_df.ffill().bfill() # Simple forward/backward fill for missing values
    
    # Preprocess the new data point
    df_engineered = engineer_features(raw_data_point_df)
    X_new = apply_transformers(df_engineered, preprocessor, extended_cat_preprocessor, lane_preprocessor, tfidf_vectorizer)

    # Ensure feature order matches training data
    X_new_aligned = X_new[feature_names] # Use feature_names saved at training time

    # Make prediction
    prediction_prob = champion_model.predict_proba(X_new_aligned)[:, 1][0]
    predicted_risk = "HIGH" if prediction_prob > 0.5 else "LOW"

    # Generate SHAP explanation
    explainer = shap.TreeExplainer(champion_model)
    shap_values = explainer(X_new_aligned)
    # If shap_values is a list of Explanation objects, take the first one for single prediction
    if isinstance(shap_values, list): # For models that return list of Explanation objects for multi-output
        shap_explanation = shap_values[0]
    else: # For single Explanation object
        shap_explanation = shap_values

    print(f"\nPredicted Late Delivery Risk: {predicted_risk} (Probability: {prediction_prob:.4f})")
    shap.force_plot(shap_explanation, link="logit", matplotlib=False, show=True)

    # Extract and display NLP feature contributions
    nlp_feature_names = tfidf_vectorizer.get_feature_names_out()
    nlp_shap_contributions = {}

    # Ensure feature_names from shap_explanation align with X_new_aligned columns
    if hasattr(shap_explanation, 'data') and shap_explanation.data is not None:
        current_feature_names = X_new_aligned.columns # Use the columns from the processed X_new_aligned
    else:
        current_feature_names = feature_names # Fallback to global feature names

    # Ensure shap_explanation.values is a 1D array for a single sample
    shap_values_for_current_sample = shap_explanation.values.flatten() if len(shap_explanation.values.shape) > 1 else shap_explanation.values

    for feature_name, shap_value in zip(current_feature_names, shap_values_for_current_sample):
        if feature_name in nlp_feature_names and abs(shap_value) > 0.001:
            nlp_shap_contributions[feature_name] = shap_value

    if nlp_shap_contributions:
        print("\n  Key NLP Feature Contributions (SHAP Values) for this Prediction:")
        sorted_nlp_contributions = sorted(nlp_shap_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        for k, v in sorted_nlp_contributions:
            print(f"    - '{k}': {v:.4f} (Positive value increases late risk, negative decreases)")
    else:
        print("\n  No significant NLP-derived feature contributions detected for this sample.")
