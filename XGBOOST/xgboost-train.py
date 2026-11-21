
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(r"C:\Users\ip471\OneDrive\Documents\GitHub\Poster-Competition")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "XGBoost"
OUTPUT_DIR = BASE_DIR / "outputs"

IV_DIR = MODEL_DIR / "IV_training"
HV_DIR = MODEL_DIR / "HV_training"

IV_DIR.mkdir(parents=True, exist_ok=True)
HV_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42

FEATURES = [
    "S",
    "K",
    "T",
    "r",
    "sigma",
    "log_moneyness",
    "sqrt_T",
    "option_type_encoded"
]


def add_features(df):
    df = df.copy()

    df["log_moneyness"] = np.log(df["S"] / df["K"])
    df["sqrt_T"] = np.sqrt(df["T"])

    df["option_type_encoded"] = df["option_type"].map({
        "call": 1,
        "put": 0
    })

    if df["option_type_encoded"].isna().any():
        raise ValueError("Unexpected option_type values found.")

    return df


def build_model():
    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


def train_model(df):
    df = add_features(df)

    X = df[FEATURES].astype(float)
    y = np.log1p(df["target_price"].values)

    model = build_model()
    model.fit(X, y)

    return model


def predict(model, df):
    df = add_features(df)

    X = df[FEATURES].astype(float)

    predictions = model.predict(X)

    return np.maximum(np.expm1(predictions), 0)


def validate_data(iv_train_df, hv_train_df, test_df):

    assert len(iv_train_df) == len(hv_train_df)

    assert (
        iv_train_df["contract_id"].tolist()
        == hv_train_df["contract_id"].tolist()
    )

    assert np.allclose(
        iv_train_df["target_price"].values,
        hv_train_df["target_price"].values
    )

    assert test_df["contract_id"].is_unique

    for column in ["sigma_HV", "sigma_IV"]:
        assert test_df[column].notna().all()
        assert np.isfinite(test_df[column]).all()
        assert (test_df[column] > 0).all()

    assert test_df["target_price"].notna().all()
    assert (test_df["target_price"] > 0).all()

    print("Dataset validation successful.")


def run():

    iv_train_df = pd.read_csv(
        DATA_DIR / "real_iv_train.csv"
    )

    hv_train_df = pd.read_csv(
        DATA_DIR / "real_hv_train.csv"
    )

    test_df = pd.read_csv(
        DATA_DIR / "real_test.csv"
    )

    validate_data(
        iv_train_df,
        hv_train_df,
        test_df
    )

    all_predictions = []
    metrics_rows = []

    experiments = [
        {
            "name": "IV",
            "train_df": iv_train_df,
            "test_sigma": "sigma_IV",
            "model_dir": IV_DIR
        },
        {
            "name": "HV",
            "train_df": hv_train_df,
            "test_sigma": "sigma_HV",
            "model_dir": HV_DIR
        }
    ]

    for experiment in experiments:

        name = experiment["name"]
        train_df = experiment["train_df"]
        test_sigma = experiment["test_sigma"]
        model_dir = experiment["model_dir"]

        print(f"\nTraining XGBoost using {name}...")

        model = train_model(train_df)

        model_path = model_dir / f"xgboost_{name}.pkl"
        joblib.dump(model, model_path)

        experiment_test = test_df.copy()
        experiment_test["sigma"] = experiment_test[test_sigma]

        predictions = predict(model, experiment_test)

        actual = experiment_test["target_price"].values

        metrics = evaluate(actual, predictions)

        metrics_rows.append({
            "model": f"XGBoost_{name}",
            "training_contracts": len(train_df),
            "test_contracts": len(experiment_test),
            **metrics
        })

        results = experiment_test[
            [
                "contract_id",
                "ticker",
                "option_type",
                "target_price"
            ]
        ].copy()

        results["volatility"] = name
        results["model"] = f"XGBoost_{name}"
        results["sigma"] = experiment_test["sigma"].values
        results["predicted_price"] = predictions
        results["error"] = (
            results["target_price"] -
            results["predicted_price"]
        )
        results["absolute_error"] = results["error"].abs()

        all_predictions.append(results)

        print(f"\n{name} results:")
        print(pd.Series(metrics))

        print(f"\nModel saved to: {model_path}")

    final_predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    metrics_df = pd.DataFrame(metrics_rows)

    final_predictions.to_csv(
        OUTPUT_DIR / "XGBoost_predictions.csv",
        index=False
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "XGBoost_metrics.csv",
        index=False
    )

    print("\nFinal XGBoost metrics:")
    print(metrics_df.to_string(index=False))

    print("\nPrediction results:")
    print(final_predictions.head())

    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
