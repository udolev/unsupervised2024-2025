import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import prince
from config import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL, RANDOM_SEED

def load_data(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess data including encoding and scaling."""
    # Check for missing values
    print("Missing values per column:\n", df.isnull().sum())
    
    # Encode the target variable
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
    print(f"Target classes after encoding: {le.classes_.tolist()}")
    
    # Split features and target
    y = df[TARGET_COL]
    X_full = df.drop(columns=TARGET_COL)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_full[NUMERICAL_COLS]),
        columns=NUMERICAL_COLS,
        index=X_full.index
    )
    
    # Extract categorical features
    X_cat = X_full[CATEGORICAL_COLS]
    
    return X_cat, X_num_scaled, y

def apply_mca(X_cat, n_components):
    """Apply Multiple Correspondence Analysis for dimensionality reduction."""
    mca = prince.MCA(n_components=n_components)
    df_mca = mca.fit_transform(X_cat)
    return df_mca

def combine_features(df_mca, X_num_scaled):
    """Combine MCA-transformed categorical features with numerical features."""
    X_combined = pd.concat([df_mca.reset_index(drop=True), 
                          X_num_scaled.reset_index(drop=True)], axis=1)
    X_combined.columns = X_combined.columns.astype(str)
    return X_combined

def sample_data(df, n_samples):
    """Get a random sample from dataframe."""
    return df.sample(n=min(n_samples, len(df)), random_state=RANDOM_SEED)