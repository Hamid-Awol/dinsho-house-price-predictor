import numpy as np
import pandas as pd

def prepare_features(input_data, feature_names, encoders):
    features = []
    for feature in feature_names:
        if feature in input_data:
            value = input_data[feature]
            if feature in encoders:
                try:
                    value = encoders[feature].transform([str(value)])[0]
                except ValueError:
                    value = encoders[feature].transform([encoders[feature].classes_[0]])[0]
            features.append(value)
        else:
            features.append(0)
    return np.array(features).reshape(1, -1)

def predict_price(model, scaler, input_data, feature_names, encoders):
    features = prepare_features(input_data, feature_names, encoders)
    features_scaled = scaler.transform(features)
    price = model.predict(features_scaled)[0]
    return price

def predict_bulk_prices(model, scaler, df_input, feature_names, encoders, discount_threshold=2, discount_rate=0.10):
    predictions = []
    for idx, row in df_input.iterrows():
        input_data = row.to_dict()
        price = predict_price(model, scaler, input_data, feature_names, encoders)
        predictions.append(price)
    df_results = df_input.copy()
    df_results['predicted_price_etb'] = predictions
    total_houses = len(df_results)
    if total_houses > discount_threshold:
        discount_amount = df_results['predicted_price_etb'].sum() * discount_rate
        discounted_total = df_results['predicted_price_etb'].sum() - discount_amount
        df_results['discount_applied'] = f"{discount_rate*100:.0f}%"
        df_results['discounted_price_etb'] = df_results['predicted_price_etb'] * (1 - discount_rate)
        summary = {'total_houses': total_houses, 'original_total': df_results['predicted_price_etb'].sum(),
                   'discount_rate': discount_rate, 'discount_amount': discount_amount,
                   'final_total': discounted_total, 'bulk_discount_applied': True}
    else:
        df_results['discount_applied'] = '0%'
        df_results['discounted_price_etb'] = df_results['predicted_price_etb']
        summary = {'total_houses': total_houses, 'original_total': df_results['predicted_price_etb'].sum(),
                   'discount_rate': 0, 'discount_amount': 0,
                   'final_total': df_results['predicted_price_etb'].sum(), 'bulk_discount_applied': False}
    return df_results, summary

def get_sample_csv_template():
    sample_data = {'sub_city': ['Bole', 'Yeka', 'Kirkos'], 'area_sqm': [150, 200, 180],
                   'bedrooms': [3, 4, 3], 'bathrooms': [2, 3, 2], 'year_built': [2015, 2010, 2020],
                   'water_source': ['Public Water', 'Private Well', 'Public Water'],
                   'electricity': ['Yes', 'Yes', 'Yes'], 'fence': ['Yes', 'No', 'Yes'],
                   'floor_type': ['G+2', 'G+4', 'G+1'], 'property_condition': ['Good', 'Excellent', 'Good'],
                   'road_access': ['Asphalt', 'Gravel', 'Asphalt'],
                   'distance_to_school_km': [2.5, 1.5, 3.0], 'distance_to_market_km': [1.5, 2.0, 1.0]}
    return pd.DataFrame(sample_data)
