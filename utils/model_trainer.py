import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

def train_model_with_complexity_control(df, test_size=0.2, n_estimators=100, 
                                        max_depth=None, min_samples_split=2,
                                        min_samples_leaf=1, max_features='sqrt',
                                        regularization=None, alpha=1.0,
                                        use_cross_validation=True, cv_folds=5,
                                        random_state=42):
    X = df.drop('price_etb', axis=1)
    y = df['price_etb']
    feature_names = X.columns.tolist()
    encoders = {}
    for col in X.select_dtypes(['object']):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if regularization == 'ridge':
        model = Ridge(alpha=alpha, random_state=random_state)
    elif regularization == 'lasso':
        model = Lasso(alpha=alpha, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      max_features=max_features, random_state=random_state, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    cv_scores = None
    if use_cross_validation and not regularization:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2')
    y_pred = model.predict(X_test_scaled)
    train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    overfitting_score = train_r2 - test_r2
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    return {'model': model, 'scaler': scaler, 'encoders': encoders, 'feature_names': feature_names,
            'metrics': {'train_r2': train_r2, 'test_r2': test_r2, 'rmse': rmse, 'overfitting_score': overfitting_score,
                        'test_size': test_size, 'n_estimators': n_estimators, 'max_depth': max_depth,
                        'cv_scores': cv_scores, 'cv_mean': cv_scores.mean() if cv_scores is not None else None,
                        'cv_std': cv_scores.std() if cv_scores is not None else None},
            'feature_importance': feature_importance}

def save_model(model_data, model_path="models/house_price_model.pkl", 
               scaler_path="models/scaler.pkl", encoders_path="models/encoders.pkl"):
    os.makedirs("models", exist_ok=True)
    with open(model_path, 'wb') as f: pickle.dump(model_data['model'], f)
    with open(scaler_path, 'wb') as f: pickle.dump(model_data['scaler'], f)
    with open(encoders_path, 'wb') as f: pickle.dump(model_data['encoders'], f)
    metadata = {'feature_names': model_data['feature_names'], 'metrics': model_data['metrics']}
    with open("models/metadata.pkl", 'wb') as f: pickle.dump(metadata, f)

def load_model(model_path="models/house_price_model.pkl", scaler_path="models/scaler.pkl",
               encoders_path="models/encoders.pkl"):
    try:
        with open(model_path, 'rb') as f: model = pickle.load(f)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        with open(encoders_path, 'rb') as f: encoders = pickle.load(f)
        with open("models/metadata.pkl", 'rb') as f: metadata = pickle.load(f)
        return {'model': model, 'scaler': scaler, 'encoders': encoders,
                'feature_names': metadata['feature_names'], 'metrics': metadata['metrics']}
    except FileNotFoundError:
        return None
