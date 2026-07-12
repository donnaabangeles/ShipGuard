import pandas as pd
import numpy as np

def analyze_high_risk_lane_combinations(df_original, X_test_df, champion_model):
    """
    Analyzes high-risk lane combinations (Origin City, Origin Country, Destination Country)
    and identifies the top products associated with late delivery risk in those lanes.

    Args:
        df_original (pd.DataFrame): The original DataFrame containing all raw features.
        X_test_df (pd.DataFrame): The preprocessed test DataFrame used for prediction.
        champion_model: The trained machine learning model to make predictions.

    Returns:
        pd.DataFrame: A DataFrame summarizing high-risk lane combinations, top products, and frequency.
    """
    y_pred_test = champion_model.predict(X_test_df)

    late_delivery_indices = X_test_df.index[y_pred_test == 1]

    high_risk_transactions = df_original.loc[late_delivery_indices].copy()

    lane_risk_products = high_risk_transactions[[
        'Customer City',
        'Customer Country',
        'Order Country',
        'Product Name'
    ]].copy()

    lane_summary = lane_risk_products.groupby(['Customer City', 'Customer Country', 'Order Country'])['Product Name'] \
        .agg(lambda x: ', '.join(x.value_counts().index.tolist()[:3])) \
        .reset_index()

    lane_summary.rename(columns={
        'Customer City': 'Origin City',
        'Customer Country': 'Origin Country',
        'Order Country': 'Destination Country',
        'Product Name': 'Top Products At Risk'
    }, inplace=True)

    lane_counts = lane_risk_products.groupby(['Customer City', 'Customer Country', 'Order Country']).size().reset_index(name='Frequency')

    lane_counts.rename(columns={
        'Customer City': 'Origin City',
        'Customer Country': 'Origin Country',
        'Order Country': 'Destination Country'
    }, inplace=True)

    final_risk_table = pd.merge(lane_summary, lane_counts, on=['Origin City', 'Origin Country', 'Destination Country'])

    final_risk_table.sort_values(by='Frequency', ascending=False, inplace=True)

    return final_risk_table
