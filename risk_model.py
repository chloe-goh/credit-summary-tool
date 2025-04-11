import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier

# 1. Load data with explicit checks
try:
    df = pd.read_csv("credit_data.csv")

    # Clean column names by removing (USD M) but keep them accessible
    df.columns = [col.replace(' (USD M)', '') for col in df.columns]

    # Now we can reference clean column names
    print("Available columns:", df.columns.tolist())
    
    # Verify critical columns exist
    required_columns = {'sp_rating', 'total_assets', 'total_debt', 'ebitda'}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")

    # 2. Enhanced rating cleaning
    print("\nOriginal sp_rating values:")
    print(df['sp_rating'].value_counts(dropna=False))
    
    # Standardize ratings
    df['sp_rating_clean'] = (df['sp_rating']
                            .astype(str)
                            .str.upper()
                            .str.strip()
                            .str.replace(' ', '')
                            .str.replace('*', '')
                            )
    
    # 3. Comprehensive PD mapping
    rating_to_pd = {
        'AAA': 0.03, 'AA+': 0.05, 'AA': 0.07, 'AA-': 0.10,
        'A+': 0.15, 'A': 0.20, 'A-': 0.30,
        'BBB+': 0.50, 'BBB': 0.70, 'BBB-': 1.00,
        'BB+': 1.50, 'BB': 2.00, 'BB-': 3.00,
        'B+': 5.00, 'B': 7.50, 'B-': 10.00,
        'CCC+': 20.00, 'CCC': 25.00, 'CC': 35.00, 'C': 50.00,
        'D': 100.00
    }
    
    # Handle any unexpected ratings
    unexpected_ratings = set(df['sp_rating_clean']) - set(rating_to_pd.keys())
    if unexpected_ratings:
        print(f"\nWarning: Unexpected ratings found - defaulting to 50%% PD: {unexpected_ratings}")
    
    df['sp_pd'] = df['sp_rating_clean'].map(lambda x: rating_to_pd.get(x, 50.00)) / 100
    
    # 4. Merton Model with validation
    df['equity_volatility'] = 0.30  # Conservative estimate
    
    # Validate financials
    if (df['total_assets'] <= 0).any() or (df['total_debt'] < 0).any():
        raise ValueError("Invalid financial values detected")
    
    df['distance_to_default'] = (np.log(df['total_assets']/df['total_debt']) + 
                               (0.05 + 0.5*df['equity_volatility']**2)*1) / \
                               df['equity_volatility']
    df['merton_pd'] = norm.cdf(-df['distance_to_default'])
    
    # 5. Machine Learning Prep
    df['leverage_ratio'] = df['total_debt'] / df['ebitda'].replace(0, 0.01)  # Avoid div/0
    df['interest_coverage'] = df['ebitda'] / (df['total_debt'] * 0.05)  # Assume 5% interest
    df['quick_ratio'] = (df['cash_equivalents'] + 0.3*df['total_assets']) / (0.4*df['total_debt'])
    
    # 6. Risk Classification
    X = df[['leverage_ratio', 'interest_coverage', 'quick_ratio', 'sp_pd']].fillna(0)
    y = pd.cut(df['merton_pd'], bins=[0, 0.05, 0.15, 1], labels=['Low', 'Medium', 'High'])
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    df['ml_risk'] = model.predict(X)
    
    # 7. Save with verification
    output_file = "corporate_pd_results_enhanced.csv"
    df.to_csv(output_file, index=False)
    
    print("\nSuccess! Output saved to:", output_file)
    print("Final risk distribution:")
    print(df['ml_risk'].value_counts())
    
except Exception as e:
    print("\nError occurred:", str(e))
    if 'df' in locals():
        print("\nDebug info:")
        print("Data types:\n", df.dtypes)
        print("\nSample sp_rating values:\n", df['sp_rating'].head())
        print("\nNull counts:\n", df.isnull().sum())