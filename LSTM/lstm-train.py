
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


BASE_DIR = Path(r"C:\Users\ip471\OneDrive\Documents\GitHub\Poster-Competition")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "LSTM"
OUTPUT_DIR = BASE_DIR / "outputs"

IV_DIR = MODEL_DIR / "IV_training"
HV_DIR = MODEL_DIR / "HV_training"

IV_DIR.mkdir(parents=True, exist_ok=True)
HV_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 256

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
    model = Sequential([
        tf.keras.Input(shape=(len(FEATURES), 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def prepare_data(df, scaler_X=None, scaler_y=None, fit_scalers=False):

    df = add_features(df)

    X = df[FEATURES].values.astype(float)
    y = df["target_price"].values.reshape(-1, 1)

    if fit_scalers:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y)

    else:
        X = scaler_X.transform(X)
        y = scaler_y.transform(y)

    X = X.reshape(
        (X.shape[0], X.shape[1], 1)
    )

    return X, y, scaler_X, scaler_y


def train_model(df):

    X, y, scaler_X, scaler_y = prepare_data(
        df,
        fit_scalers=True
    )

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    model = build_model()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    history = model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[early_stopping],
        shuffle=True,
        verbose=1
    )

    return model, scaler_X, scaler_y, history


def predict(model, scaler_X, scaler_y, df):

    df = add_features(df)

    X = df[FEATURES].values.astype(float)

    X = scaler_X.transform(X)

    X = X.reshape(
        (X.shape[0], X.shape[1], 1)
    )

    predictions = model.predict(
        X,
        verbose=0
    )

    predictions = scaler_y.inverse_transform(predictions)

    return np.maximum(predictions.flatten(), 0)


def evaluate(y_true, y_pred):

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


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
    assert np.isfinite(test_df["target_price"]).all()
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

        print(f"\nTraining LSTM using {name}...")

        model, scaler_X, scaler_y, history = train_model(
            train_df
        )

        model_path = model_dir / f"lstm_{name}.keras"
        scaler_x_path = model_dir / f"lstm_{name}_scaler_X.pkl"
        scaler_y_path = model_dir / f"lstm_{name}_scaler_y.pkl"

        model.save(model_path)

        joblib.dump(scaler_X, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)

        experiment_test = test_df.copy()

        experiment_test["sigma"] = experiment_test[test_sigma]

        predictions = predict(
            model,
            scaler_X,
            scaler_y,
            experiment_test
        )

        actual = experiment_test["target_price"].values

        metrics = evaluate(actual, predictions)

        metrics_rows.append({
            "model": f"LSTM_{name}",
            "training_contracts": len(train_df),
            "test_contracts": len(experiment_test),
            "epochs_completed": len(history.history["loss"]),
            "best_validation_loss": min(history.history["val_loss"]),
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
        results["model"] = f"LSTM_{name}"
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
        OUTPUT_DIR / "LSTM_predictions.csv",
        index=False
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "LSTM_metrics.csv",
        index=False
    )

    print("\nFinal LSTM metrics:")
    print(metrics_df.to_string(index=False))

    print("\nPrediction results:")
    print(final_predictions.head())

    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
