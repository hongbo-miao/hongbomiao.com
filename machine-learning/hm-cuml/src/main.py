import logging

import cudf
from cuml.ensemble import RandomForestClassifier
from cuml.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def main() -> None:
    # Load the iris dataset
    iris = load_iris()
    x = iris.data
    y = iris.target

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Convert to cuDF DataFrames
    x_train_cudf = cudf.DataFrame(x_train)
    x_test_cudf = cudf.DataFrame(x_test)
    y_train_cudf = cudf.Series(y_train)
    y_test_cudf = cudf.Series(y_test)

    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_cudf)
    x_test_scaled = scaler.transform(x_test_cudf)

    # Create and train the model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train_scaled, y_train_cudf)

    # Make predictions
    y_pred_cudf = rf_classifier.predict(x_test_scaled)

    # Convert predictions back to CPU for evaluation
    y_pred = y_pred_cudf.values_host
    y_test = y_test_cudf.values_host

    # Print results
    logger.info("cuML Results:")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=iris.target_names))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
