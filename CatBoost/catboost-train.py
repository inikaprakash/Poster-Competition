
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor



BASE_DIR = Path(
    r"C:\Users\ip471\OneDrive\Documents\GitHub\Poster-Competition"
)

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "CatBoost"
OUTPUT_DIR = BASE_DIR / "outputs"

IV_MODEL_DIR = MODEL_DIR / "IV_training"
HV_MODEL_DIR = MODEL_DIR / "HV_training"

IV_MODEL_DIR.mkdir(parents=True, exist_ok=True)
HV_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



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

RANDOM_STATE = 42



def prepare_features(df, volatility_column=None):

    df = df.copy()

    if volatility_column is not None:
        df["sigma"] = df[volatility_column]

    df["log_moneyness"] = np.log(df["S"] / df["K"])
    df["sqrt_T"] = np.sqrt(df["T"])

    df["option_type_encoded"] = (
        df["option_type"].map({"call": 1, "put": 0})
    )

    if df[FEATURES].isnull().any().any():
        raise ValueError("Missing values found in model features.")

    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError("Non-finite values found in model features.")

    return df




def build_model():

    return CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        l2_leaf_reg=5,
        random_seed=RANDOM_STATE,
        verbose=100,
        allow_writing_files=False
    )



def train_model(train_df):

    X = train_df[FEATURES]
    y = train_df["target_price"]

    model = build_model()
    model.fit(X, y)

    return model



def predict(model, df):

    X = df[FEATURES]

    predictions = model.predict(X)

    return np.asarray(predictions).flatten()




def evaluate(y_true, y_pred):

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }




def run_experiment(
    train_df,
    test_df,
    volatility_name,
    volatility_column,
    model_dir
):

    print(f"\nTraining CatBoost using {volatility_name}...")

    train_df = prepare_features(train_df)

    test_df = prepare_features(
        test_df,
        volatility_column=volatility_column
    )


    model = train_model(train_df)

    model_path = model_dir / f"CatBoost_{volatility_name}.cbm"
    model.save_model(str(model_path))


    predictions = predict(model, test_df)

    y_true = test_df["target_price"].to_numpy()


    metrics = evaluate(y_true, predictions)

    metrics_row = {
        "model": f"CatBoost_{volatility_name}",
        "training_contracts": len(train_df),
        "test_contracts": len(test_df),
        **metrics
    }


    results = test_df.copy()
    results["model"] = f"CatBoost_{volatility_name}"
    results["predicted_price"] = predictions
    results["absolute_error"] = np.abs(y_true - predictions)

    print(f"\n{volatility_name} results:")
    print(metrics_row)

    return results, metrics_row




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


    if not iv_train_df["contract_id"].equals(
        hv_train_df["contract_id"]
    ):
        raise ValueError(
            "IV and HV training datasets do not contain "
            "the same contracts in the same order."
        )


    if test_df["contract_id"].duplicated().any():
        raise ValueError("Duplicate contracts found in test dataset.")


    if not np.allclose(
        iv_train_df["target_price"],
        hv_train_df["target_price"]
    ):
        raise ValueError(
            "IV and HV training targets do not match."
        )

    all_predictions = []
    metrics_rows = []



    iv_predictions, iv_metrics = run_experiment(
        train_df=iv_train_df,
        test_df=test_df,
        volatility_name="IV",
        volatility_column="sigma_IV",
        model_dir=IV_MODEL_DIR
    )

    all_predictions.append(iv_predictions)
    metrics_rows.append(iv_metrics)



    hv_predictions, hv_metrics = run_experiment(
        train_df=hv_train_df,
        test_df=test_df,
        volatility_name="HV",
        volatility_column="sigma_HV",
        model_dir=HV_MODEL_DIR
    )

    all_predictions.append(hv_predictions)
    metrics_rows.append(hv_metrics)



    final_predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    metrics_df = pd.DataFrame(metrics_rows)

    final_predictions.to_csv(
        OUTPUT_DIR / "CatBoost_predictions.csv",
        index=False
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "CatBoost_metrics.csv",
        index=False
    )

    print("\n=========================")
    print("FINAL CATBOOST RESULTS")
    print("=========================")

    print(metrics_df.to_string(index=False))

    print("\nPrediction sample:")
    print(
        final_predictions[
            [
                "contract_id",
                "option_type",
                "target_price",
                "predicted_price",
                "absolute_error",
                "model"
            ]
        ].head()
    )

    print("\nFiles saved successfully.")



if __name__ == "__main__":
    run()
