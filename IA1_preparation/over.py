import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("Heart.csv")

print(df.head())
print(df.info())


df = df.drop(columns=["Unnamed: 0"])



numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include='object').columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values after filling:")
print(df.isnull().sum())



plt.figure()
sns.countplot(x="AHD", data=df)
plt.title("Target Distribution")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()



for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Shape after removing outliers:", df.shape)


df.to_csv("heart_cleaned.csv", index=False)

print("Clean dataset saved as heart_cleaned.csv")


df["AHD"] = df["AHD"].map({"No":0, "Yes":1})


X = df.drop("AHD", axis=1)
y = df["AHD"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=999
)


categorical_cols = X_train.select_dtypes(include='object').columns

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Fit encoder on training data
encoder.fit(X_train[categorical_cols])

# Transform data
X_train_encoded = encoder.transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

# Convert to DataFrame
encoded_train_df = pd.DataFrame(
    X_train_encoded,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_train.index
)

encoded_test_df = pd.DataFrame(
    X_test_encoded,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_test.index
)

# Drop original categorical columns
X_train = X_train.drop(columns=categorical_cols)
X_test = X_test.drop(columns=categorical_cols)

# Concatenate encoded columns
X_train = pd.concat([X_train, encoded_train_df], axis=1)
X_test = pd.concat([X_test, encoded_test_df], axis=1)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_params = {
    "C":[0.01,0.1,1,10],
    "solver":["lbfgs","liblinear"]
}

log_grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    log_params,
    cv=10
)

log_grid.fit(X_train_scaled, y_train)

best_log_model = log_grid.best_estimator_

log_pred = best_log_model.predict(X_test_scaled)

log_acc = accuracy_score(y_test, log_pred)

print("Best Logistic Parameters:", log_grid.best_params_)
print("Logistic Regression Accuracy:", log_acc)


rf_params = {
    "n_estimators":[100,200,300],
    "max_depth":[4,6,8,10],
    "min_samples_split":[2,5,10],
"min_samples_leaf":[1,2,4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=10
)

rf_grid.fit(X_train, y_train)

best_rf_model = rf_grid.best_estimator_

rf_pred = best_rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("Best RF Parameters:", rf_grid.best_params_)
print("Random Forest Accuracy:", rf_acc)


desc_param = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3,5,7,10],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4]
}


desc_grid = GridSearchCV(
    DecisionTreeClassifier(),
    desc_param,
    cv=10,
)

desc_grid.fit(X_train, y_train)

best_desc_model = desc_grid.best_estimator_

desc_pred = best_desc_model.predict(X_test)

desc_acc = accuracy_score(y_test, desc_pred)

print("Best DT Parameters:", desc_grid.best_params_)
print("Descision Tree Accuracy:", desc_acc)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

log_cv = cross_val_score(best_log_model, X_train_scaled, y_train,
                         cv=kfold, scoring="accuracy")

rf_cv = cross_val_score(best_rf_model, X_train, y_train,
                        cv=kfold, scoring="accuracy")

dt_cv = cross_val_score(best_desc_model, X_train, y_train, cv=kfold, scoring="accuracy")

print("Logistic Regression CV Mean:", log_cv.mean())
print("Random Forest CV Mean:", rf_cv.mean())
print("Decision Tree CV Mean:", dt_cv.mean())



results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Decision Tree"],
    "Accuracy": [log_cv.mean(), rf_cv.mean(), dt_cv.mean()],
})

print(results)

sns.barplot(x="Model", y="Accuracy", data=results)
plt.title("Model Comparison")
plt.show()
