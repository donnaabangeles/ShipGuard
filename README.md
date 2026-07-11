
## Supply Chain Late Delivery Risk Prediction

## Overview

This project develops a machine learning solution to predict `Late_delivery_risk` in a supply chain dataset. By leveraging various features, including numerical data, categorical information, and natural language processing (NLP) on product names, the goal is to identify and explain factors contributing to delivery delays. The solution culminates in the identification of high-risk 'lane combinations' (origin city/country to destination country) and associated products, providing actionable insights for supply chain optimization.

## Key Findings

*   **High-Performing Model:** An **XGBoost Classifier** was selected as the champion model due to its robust performance, achieving an accuracy of **0.9749** and an ROC-AUC score of **0.9950** when trained with comprehensive features, including NLP and lane combinations.

*   **Feature Importance:**
    *   `Shipping_Delay_Variance` (real vs. scheduled shipping days) consistently emerged as the most influential feature in predicting late delivery risk.
    *   `Type_TRANSFER` also showed significant impact.
    *   **NLP features** extracted from `Product Name` (via TF-IDF) contributed meaningfully. For example, terms like 'golf' and 'women' tended to increase risk, while 'nike' and 'comfort' tended to decrease risk.
    *   **Lane Combination Features:** `Customer City`, `Customer Country`, `Market`, and `Order Country` were integrated. `Customer City_Caguas` and `Market_Pacific Asia` were identified by SHAP as having the highest overall impact among these features.

*   **High-Risk Lane Combinations:** The analysis revealed specific high-risk lanes. For example:
    *   **'Caguas, Puerto Rico' to 'Estados Unidos'**: This is the most frequent high-risk lane, with products like 'Nike Men's Dri-FIT Victory Golf Polo' being a top at-risk item.
    *   Other significant high-risk lanes involve 'Caguas, Puerto Rico' shipping to 'México' and 'Francia'.

*   **Logistics Insights (Model-Derived):** The model suggests that origins from island territories (e.g., Puerto Rico) and destinations within large, complex markets (e.g., Pacific Asia, Spain) may face inherent logistical challenges contributing to delays. Specific product names also carry predictive power for risk.

## Project Structure

For a clean, reproducible, and deployable project, the recommended structure is as follows:

```
ShipGuard/
├── data/
│   └── DataCoSupplyChainDataset.csv  # Original dataset
├── notebooks/
│   └── exploratory_analysis.ipynb           # This notebook (or a cleaned version)
├── src/
│   ├── __init__.py
│   ├── preprocess.py                 # Functions for data cleaning, feature engineering, and TF-IDF vectorization
│   ├── model.py                      # Functions for model training, evaluation, loading, prediction, and SHAP explanation generation
│   └── utils.py                      # Any utility functions (e.g., plot configurations, helper functions, high_risk_analysis function)
├── models/
│   ├── champion_model_lane_combinations.joblib # Final trained model
│   ├── tfidf_vectorizer.joblib             # TF-IDF vectorizer for Product Name
│   ├── initial_preprocessor.joblib         # Preprocessor for initial numerical/categorical features
│   ├── extended_cat_preprocessor.joblib    # Preprocessor for Customer Country and Market
│   ├── lane_preprocessor.joblib            # Preprocessor for Customer City and Order Country
│   ├── feature_names.joblib                # List of all feature names for consistent input
│   └── feature_names_raw_df_at_fit.joblib  # List of raw dataframe columns at fit time
├── production_artifacts/                     # Directory for all saved artifacts from modularized code
├── requirements.txt                  # List of Python dependencies
├── README.md                         # Project description, setup, usage, and results
└── LICENSE                           # License information (e.g., MIT, Apache 2.0)
```

## Setup Guide

To set up and run this project, follow these steps:

### 1. Clone the Repository

```bash
git clone <your-github-repo-url>
cd my_supply_chain_project
```

### 2. Set up Python Environment

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all necessary libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Data Acquisition

*   Download the `dataco-smart-supply-chain-for-big-data-analysis.zip` dataset from Kaggle. You will need a Kaggle API token. Place the unzipped `DataCoSupplyChainDataset.csv` file into the `data/` directory.
*   Alternatively, manually place the `DataCoSupplyChainDataset.csv` file directly into the `data/` directory.

### 5. Run the Notebooks

Open `notebooks/main_analysis.ipynb` in your preferred Jupyter environment (e.g., Jupyter Lab, VS Code, Google Colab) and run the cells sequentially to reproduce the analysis, model training, and explanation steps. The notebook also demonstrates the modularized code usage from the `src/` directory.

### 6. Production Readiness

The project includes refactored code into the `src/` directory and saves all necessary model artifacts (champion model, preprocessors, vectorizers, feature names) into the `production_artifacts/` directory. This allows for easy deployment and consistent predictions on new data without retraining.

### 7. Deployment Considerations

With the project structured and refactored, you can deploy the `src/` modules in various environments:

*   **API Endpoint:** Wrap the prediction logic in a REST API (e.g., using Flask or FastAPI) to serve predictions.
*   **Batch Processing:** Integrate into a batch processing pipeline for regular risk assessment.
*   **Dashboard Integration:** Connect to a business intelligence dashboard to visualize high-risk lanes and product alerts in real-time.