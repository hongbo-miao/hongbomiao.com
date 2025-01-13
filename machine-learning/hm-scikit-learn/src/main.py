import logging

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Create and train the model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train_scaled, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(x_test_scaled)

    # logger.info results
    logger.info("Scikit-learn Results:")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=iris.target_names))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
