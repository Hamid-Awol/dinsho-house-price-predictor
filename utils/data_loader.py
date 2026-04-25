import pandas as pd
import numpy as np

def load_data():
    """Generate the house price dataset"""
    np.random.seed(42)
    n = 500
    sub_cities = ['Bole', 'Yeka', 'Kirkos', 'Lideta', 'Arada', 'Gullele', 'Nifas Silk', 'Akaki']
    water_sources = ['Public Water', 'Private Well', 'Borehole']
    conditions = ['Excellent', 'Good', 'Fair', 'Poor']
    road_access = ['Asphalt', 'Gravel', 'Dirt']
    floor_types = ['G+0', 'G+1', 'G+2', 'G+3', 'G+4', 'G+5']
    
    data = {
        'sub_city': np.random.choice(sub_cities, n),
        'area_sqm': np.random.randint(50, 500, n),
        'bedrooms': np.random.randint(1, 6, n),
        'bathrooms': np.random.randint(1, 4, n),
        'year_built': np.random.randint(1995, 2024, n),
        'water_source': np.random.choice(water_sources, n),
        'electricity': np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15]),
        'fence': np.random.choice(['Yes', 'No'], n, p=[0.7, 0.3]),
        'floor_type': np.random.choice(floor_types, n),
        'property_condition': np.random.choice(conditions, n, p=[0.15, 0.40, 0.30, 0.15]),
        'road_access': np.random.choice(road_access, n, p=[0.5, 0.3, 0.2]),
        'distance_to_school_km': np.round(np.random.uniform(0.2, 5.0, n), 2),
        'distance_to_market_km': np.round(np.random.uniform(0.1, 4.0, n), 2)
    }
    df = pd.DataFrame(data)
    base_price = df['area_sqm'] * 45000
    df['price_etb'] = (base_price + df['bedrooms']*500000 + df['bathrooms']*300000 + 
                       (df['year_built']-1995)*50000 + np.random.normal(0, 1000000, n)).astype(int)
    df['price_etb'] = df['price_etb'].clip(lower=500000)
    return df

def get_data_summary(df):
    return {
        "total_properties": len(df),
        "avg_price": df['price_etb'].mean(),
        "avg_area": df['area_sqm'].mean(),
        "avg_bedrooms": df['bedrooms'].mean(),
        "price_by_city": df.groupby('sub_city')['price_etb'].mean().sort_values(ascending=False),
        "feature_columns": df.drop('price_etb', axis=1).columns.tolist(),
        "categorical_columns": df.select_dtypes(['object']).columns.tolist(),
        "numerical_columns": df.select_dtypes(['int64', 'float64']).columns.tolist()
    }
