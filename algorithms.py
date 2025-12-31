import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# --- Univariate Methods ---

def detect_isolation_forest(series, contamination):
    values = series.values.reshape(-1, 1)
    values_clean = np.nan_to_num(values, nan=np.nanmedian(values))
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(values_clean)
    return preds == -1

def detect_density_lof(series, contamination):
    values = series.values.reshape(-1, 1)
    values_clean = np.nan_to_num(values, nan=np.nanmedian(values))
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    preds = lof.fit_predict(values_clean)
    return preds == -1

def detect_roc(series, contamination):
    diff = series.diff().abs()
    threshold = diff.quantile(1 - contamination)
    is_outlier = diff > threshold
    return is_outlier

def detect_iqr(series, factor):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (factor * IQR)
    upper_bound = Q3 + (factor * IQR)
    return (series < lower_bound) | (series > upper_bound)

# --- Multivariate Method ---

def detect_multivariate_robust(df, contamination):
    """
    Detects outliers considering correlations between variables 
    using Robust Covariance (Elliptic Envelope).
    """
    # Drop timestamp and handle NaNs
    data = df.select_dtypes(include=[np.number]).dropna()
    
    if data.shape[1] < 2:
        # Fallback if less than 2 numeric columns
        return pd.Series([False] * len(df), index=df.index)
        
    ee = EllipticEnvelope(contamination=contamination, random_state=42)
    preds = ee.fit_predict(data)
    
    # Re-align with original index (preds is for dropped-na data)
    full_mask = pd.Series([False] * len(df), index=df.index)
    full_mask.loc[data.index] = (preds == -1)
    
    return full_mask

# --- Cascade Logic ---

def apply_cascade_voting(masks_dict, min_votes):
    """
    Combines multiple boolean masks. 
    A point is an outlier if 'min_votes' methods agree it is an outlier.
    """
    # Create a DataFrame where columns are the method names and rows are timestamps
    votes_df = pd.DataFrame(masks_dict)
    
    # Sum True values across rows
    vote_counts = votes_df.sum(axis=1)
    
    # Final boolean mask
    return vote_counts >= min_votes