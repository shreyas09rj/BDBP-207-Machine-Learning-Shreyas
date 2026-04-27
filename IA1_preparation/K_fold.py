# #!/usr/bin/python
# # k_fold
#
# import pandas as pd
# from  sklearn.linear_model import LinearRegression
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
#
#
# def data_input():
#     file_path=("simulated_data_multiple_linear_regression_for_ML.csv")
#     df = pd.read_csv(file_path)
#     return df
# def split_value(df):
#     X = df.drop(columns=["disease_score_fluct"])
#     y = df["disease_score_fluct"]
#     return X,y
#
# def train_model(X,y):
#     scaler = StandardScaler()
#     scaler.fit(X)
#     X = scaler.transform(X)
#     model = LinearRegression()
#     model.fit(X,y)
#     return model
# def evaluate(model, X_test, y_test):
#     K_fold = KFold(n_splits=10)
#     accuracy = cross_val_score(model, X_test, y_test, cv=K_fold)
#     accuracy = accuracy.mean()
#     return accuracy
# def main():
#     df = data_input()
#     X,y = split_value(df)
#     model = train_model(X,y)
#     accuracy = evaluate(model, X, y)
#     print(accuracy)
# if __name__ == "__main__":
#     main()

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
#
#
# def data_input():
#     file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
#     df = pd.read_csv(file_path)
#     return df
#
#
# def split_value(df):
#     X = df.drop(columns=["disease_score_fluct"])
#     y = df["disease_score_fluct"]
#     return X, y
#
#
# def evaluate(X, y):
#
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('model', LinearRegression())
#     ])
#
#     k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
#
#     scores = cross_val_score(pipeline, X, y, cv=k_fold)
#
#     return scores.mean()
#
#
# def main():
#     df = data_input()
#
#     X, y = split_value(df)
#
#     score = evaluate(X, y)
#
#     print("Mean Cross Validation Score:", score)
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#


#!/usr/bin/python
# k_fold with multiple evaluation metrics

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def data_input():
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    df = pd.read_csv(file_path)
    return df


def split_value(df):
    X = df.drop(columns=["disease_score_fluct"])
    y = df["disease_score_fluct"]
    return X, y


def evaluate(X, y):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }

    scores = cross_validate(pipeline, X, y, cv=k_fold, scoring=scoring)

    return scores


def main():
    df = data_input()

    X, y = split_value(df)

    scores = evaluate(X, y)

    print("R2 Score (Mean):", scores['test_r2'].mean())
    print("MSE (Mean):", -scores['test_neg_mse'].mean())
    print("MAE (Mean):", -scores['test_neg_mae'].mean())

    print("\nIndividual Fold R2 Scores:")
    print(scores['test_r2'])


if __name__ == "__main__":
    main()


