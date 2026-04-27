import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

def input_data():
    filepath="simulated_data_multiple_linear_regression_for_ML.csv"
    df = pd.read_csv(filepath)
    return df
def split_data(df):
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    return X,y

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return mse,r2



def main():
    df = input_data()
    X,y = split_data(df)
    X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)
    model = train_model(X_train,y_train)
    mse,r2 = evaluate_model(model,X_test,y_test)
    print(mse,r2)






if __name__ == "__main__":
    main()