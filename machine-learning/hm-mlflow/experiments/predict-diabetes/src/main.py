import mlflow
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    mlflow.set_tracking_uri("https://mlflow.hongbomiao.com")
    mlflow.sklearn.autolog()

    db = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(db.data, db.target)

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(x_train, y_train)

    predictions = rf.predict(x_test)
    print(predictions)


if __name__ == "__main__":
    main()
