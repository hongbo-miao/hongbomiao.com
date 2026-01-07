import logging

import optuna
from optuna import Study, Trial

logger = logging.getLogger(__name__)


def objective(trial: Trial) -> float:
    x: float = trial.suggest_float("x", -100, 100)
    y: int = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


def main() -> None:
    study: Study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name="my-study",
    )
    study.optimize(objective, n_trials=100)
    logger.info(f"Best value: {study.best_value} (params: {study.best_params})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
