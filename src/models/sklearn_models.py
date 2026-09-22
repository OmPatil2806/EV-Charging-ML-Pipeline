"""Classical ML models: Linear Regression, Ridge, Random Forest, Gradient
Boosting - each tuned with GridSearchCV over the grid defined in config.yaml.
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV


def get_model_grid(cfg: dict) -> dict:
    models_cfg = cfg["models"]
    seed = cfg["random_seed"]

    return {
        "linear_regression": (LinearRegression(), {}),
        "ridge": (Ridge(random_state=seed), {"alpha": models_cfg["ridge"]["alpha"]}),
        "random_forest": (
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            {
                "n_estimators": models_cfg["random_forest"]["n_estimators"],
                "max_depth": models_cfg["random_forest"]["max_depth"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=seed),
            {
                "n_estimators": models_cfg["gradient_boosting"]["n_estimators"],
                "learning_rate": models_cfg["gradient_boosting"]["learning_rate"],
                "max_depth": models_cfg["gradient_boosting"]["max_depth"],
            },
        ),
    }


def train_sklearn_models(X_train, y_train, cfg: dict, cv: int = 3) -> dict:
    trained = {}
    for name, (estimator, param_grid) in get_model_grid(cfg).items():
        if param_grid:
            search = GridSearchCV(
                estimator, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            search.fit(X_train, y_train)
            trained[name] = search.best_estimator_
            print(f"{name}: best params {search.best_params_}, cv RMSE {-search.best_score_:.3f}")
        else:
            estimator.fit(X_train, y_train)
            trained[name] = estimator
            print(f"{name}: fitted (no hyperparameters to tune)")
    return trained
