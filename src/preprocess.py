import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_clean_data(file_path):
    """
    Loads the dataset, drops specified irrelevant columns, handles duplicates,
    fills missing values for numerical columns, and caps outliers for 'Sales'.
    """
    df = pd.read_csv(file_path, encoding='latin1')

    # Drop irrelevant columns based on initial EDA
    df = df.drop([
        "Category Id", "Customer Email", "Customer Fname", "Customer Lname",
        "Customer Password", "Customer Id", "Department Id", "Latitude", "Longitude",
        "Order Customer Id", "Order Id", "Order Item Id", "Order Item Cardprod Id",
        "Product Category Id", "Product Card Id", "Product Description", "Product Status"
    ], axis=1, errors='ignore') # errors='ignore' prevents error if column already dropped

    df = df.drop_duplicates().reset_index(drop=True)

    # Fill missing numerical values with median
    for col in ['Sales', 'Product Price']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Handle Sales outliers by capping via IQR method
    if 'Sales' in df.columns:
        Q1, Q3 = df['Sales'].quantile(0.25), df['Sales'].quantile(0.75)
        IQR = Q3 - Q1
        df['Sales'] = np.clip(df['Sales'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Ensure 'Product Name' is string type for TF-IDF
    df['Product Name'] = df['Product Name'].fillna('').astype(str)

    return df

def engineer_features(df_original):
    """
    Engineers additional features like Shipping_Delay_Variance and Sales_Volume_Bin.
    """
    df = df_original.copy()
    df['Shipping_Delay_Variance'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']

    # Feature Binning: Segment sales into equal quantiles
    # Use try-except for robustness if 'Sales' column might not exist or qcut fails
    if 'Sales' in df.columns and df['Sales'].nunique() > 1:
        df['Sales_Volume_Bin'] = pd.qcut(df['Sales'], q=3, labels=['Low_Value', 'Medium_Value', 'High_Value'], duplicates='drop')
    else:
        df['Sales_Volume_Bin'] = 'Medium_Value' # Default if not enough unique values

    return df

def create_and_fit_transformers(df):
    """
    Creates and fits the ColumnTransformer and TfidfVectorizer based on the provided DataFrame.
    Returns the fitted transformers and the list of feature names.
    """
    # Define feature groups
    num_pipeline_features = ['Sales', 'Product Price', 'Shipping_Delay_Variance']
    cat_pipeline_features = ['Shipping Mode', 'Type', 'Sales_Volume_Bin']
    extended_cat_features = ['Customer Country', 'Market'] # for section 5.1
    lane_cat_features = ['Customer City', 'Order Country'] # for section 5.3

    # Preprocessor for numerical and initial categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_pipeline_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_pipeline_features)
        ], remainder='passthrough'
    )
    preprocessor.fit(df[num_pipeline_features + cat_pipeline_features])

    # Preprocessor for extended categorical features (Customer Country, Market)
    extended_cat_preprocessor = ColumnTransformer(
        transformers=[
            ('extended_cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), extended_cat_features)
        ], remainder='passthrough'
    )
    extended_cat_preprocessor.fit(df[extended_cat_features])

    # Preprocessor for lane combination features (Customer City, Order Country)
    lane_preprocessor = ColumnTransformer(
        transformers=[
            ('lane_cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), lane_cat_features)
        ], remainder='passthrough'
    )
    lane_preprocessor.fit(df[lane_cat_features])


    # TF-IDF Vectorizer for product names
    tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_vectorizer.fit(df['Product Name'])

    return preprocessor, extended_cat_preprocessor, lane_preprocessor, tfidf_vectorizer

def apply_transformers(df, preprocessor, extended_cat_preprocessor, lane_preprocessor, tfidf_vectorizer):
    """
    Applies the fitted transformers to a DataFrame to produce the final feature matrix.
    """
    num_pipeline_features = ['Sales', 'Product Price', 'Shipping_Delay_Variance']
    cat_pipeline_features = ['Shipping Mode', 'Type', 'Sales_Volume_Bin']
    extended_cat_features = ['Customer Country', 'Market']
    lane_cat_features = ['Customer City', 'Order Country']

    # Apply initial preprocessor
    processed_initial = preprocessor.transform(df[num_pipeline_features + cat_pipeline_features])
    initial_feature_names = num_pipeline_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_pipeline_features))
    X_initial = pd.DataFrame(processed_initial, columns=initial_feature_names, index=df.index)

    # Apply extended categorical preprocessor
    processed_extended_cat = extended_cat_preprocessor.transform(df[extended_cat_features])
    extended_cat_feature_names = list(extended_cat_preprocessor.named_transformers_['extended_cat'].get_feature_names_out(extended_cat_features))
    X_extended_cat = pd.DataFrame(processed_extended_cat, columns=extended_cat_feature_names, index=df.index)

    # Apply lane categorical preprocessor
    processed_lane_cat = lane_preprocessor.transform(df[lane_cat_features])
    lane_cat_feature_names = list(lane_preprocessor.named_transformers_['lane_cat'].get_feature_names_out(lane_cat_features))
    X_lane_cat = pd.DataFrame(processed_lane_cat, columns=lane_cat_feature_names, index=df.index)


    # Apply TF-IDF vectorizer
    tfidf_features = tfidf_vectorizer.transform(df['Product Name'])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=df.index)

    # Combine all features
    X_final = pd.concat([X_initial, X_extended_cat, X_lane_cat, tfidf_df], axis=1)

    return X_final
