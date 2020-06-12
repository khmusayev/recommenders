from sklearn.utils import shuffle

def split(df, ratio=0.8):
    # split into train and test
    df = shuffle(df)
    cutoff = int(ratio * len(df))
    df_train = df.iloc[:cutoff]
    df_test = df.iloc[cutoff:]
    return df_train, df_test