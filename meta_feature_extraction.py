import numpy as np

def extract_meta_features(df):
    meta = {
        "Num_Features": 0,
        "Num_Rows": 0,
        "Num_Numeric": 0,
        "Num_Categorical": 0,
        "Missing_Values_Ratio": 0.0,
        "Class_Imbalance": 0.0
    }

    try:
        if df.empty or df.shape[1] < 2:
            raise ValueError("Dataset is empty or too few columns")

        df = df.copy()
        meta["Num_Features"] = df.shape[1] - 1
        meta["Num_Rows"] = df.shape[0]
        meta["Num_Numeric"] = len(df.select_dtypes(include=[np.number]).columns)
        meta["Num_Categorical"] = len(df.select_dtypes(include=['object', 'category']).columns)

        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        meta["Missing_Values_Ratio"] = missing_cells / total_cells if total_cells > 0 else 0.0

        target_col = df.columns[-1]
        if df[target_col].dtype in [np.int64, np.object_, 'category']:
            value_counts = df[target_col].value_counts(normalize=True)
            if len(value_counts) > 1:
                meta["Class_Imbalance"] = value_counts.max()
            else:
                meta["Class_Imbalance"] = 1.0
        else:
            meta["Class_Imbalance"] = 0.0

        return meta

    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return meta  # Return default (zeros) instead of {}
