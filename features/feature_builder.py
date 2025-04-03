def build_features(df):
    print("Building features...")
    df['feature_sum'] = df.select_dtypes(include='number').sum(axis=1)
    return df