import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def input_data():
    filepath=("sonar data.csv" )
    data =pd.read_csv(filepath)
    return data
def split_data(data):
    X = data.iloc[:, :-1]
    y_ = data.iloc[:, -1]
    y = LabelEncoder().fit_transform(y_)
    return X, y
def train_model(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr
def predict(model, X):
    prediction = model.predict(X)
    return prediction
def evaluate(model, X, y):
    prediction = model.predict(X)
    accuracy = accuracy_score(y, prediction)
    report = classification_report(y, prediction)

    return accuracy , report
def main():
    data = input_data()
    X, y = split_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    model = train_model(X_train, y_train)
    # prediction = predict(model, X_test)
    accuracy, report = evaluate(model, X_test, y_test)
    print(accuracy)
    print(report)
if __name__ == "__main__":
    main()

