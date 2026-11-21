
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import norm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)




BASE_DIR = Path(
    r"C:\Users\ip471\OneDrive\Documents\GitHub\Poster-Competition"
)

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)




def black_scholes_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type
):

    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)

        elif option_type == "put":
            return max(K - S, 0.0)

        return np.nan

    if (
        S <= 0
        or K <= 0
        or sigma <= 0
        or not np.isfinite([S, K, T, r, sigma]).all()
    ):
        return np.nan

    sqrt_T = np.sqrt(T)

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (
        sigma * sqrt_T
    )

    d2 = d1 - sigma * sqrt_T

    if option_type == "call":

        return (
            S * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

    elif option_type == "put":

        return (
            K * np.exp(-r * T) * norm.cdf(-d2)
            - S * norm.cdf(-d1)
        )

    return np.nan




def evaluate(y_true, y_pred):

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred)
    }




def run_experiment(
    test_df,
    volatility_name,
    volatility_column
):

    df = test_df.copy()


    df["sigma"] = df[volatility_column]

    df["predicted_price"] = df.apply(
        lambda row: black_scholes_price(
            S=row["S"],
            K=row["K"],
            T=row["T"],
            r=row["r"],
            sigma=row["sigma"],
            option_type=row["option_type"]
        ),
        axis=1
    )

    df = df.dropna(
        subset=["predicted_price", "target_price"]
    ).copy()

    if len(df) == 0:
        raise ValueError(
            f"No valid predictions for {volatility_name}."
        )

    y_true = df["target_price"].to_numpy()
    y_pred = df["predicted_price"].to_numpy()


    metrics = evaluate(y_true, y_pred)

    metrics_row = {
        "model": f"BlackScholes_{volatility_name}",
        "test_contracts": len(df),
        **metrics
    }


    df["model"] = f"BlackScholes_{volatility_name}"


    df["absolute_error"] = np.abs(
        y_true - y_pred
    )

    print(f"\nBlack-Scholes using {volatility_name}")
    print("-" * 40)

    for metric, value in metrics.items():
        print(f"{metric:<10}: {value:.6f}")

    print(f"Test contracts: {len(df)}")

    return df, metrics_row




def main():

    test_file = DATA_DIR / "real_test.csv"

    test_df = pd.read_csv(test_file)


    required_columns = [
        "contract_id",
        "S",
        "K",
        "T",
        "r",
        "option_type",
        "target_price",
        "sigma_IV",
        "sigma_HV"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in test dataset: {missing_columns}"
        )


    if test_df["contract_id"].duplicated().any():
        raise ValueError(
            "Duplicate contracts found in test dataset."
        )

    all_predictions = []
    metrics_rows = []



    iv_predictions, iv_metrics = run_experiment(
        test_df=test_df,
        volatility_name="IV",
        volatility_column="sigma_IV"
    )

    all_predictions.append(iv_predictions)
    metrics_rows.append(iv_metrics)


    hv_predictions, hv_metrics = run_experiment(
        test_df=test_df,
        volatility_name="HV",
        volatility_column="sigma_HV"
    )

    all_predictions.append(hv_predictions)
    metrics_rows.append(hv_metrics)

   

    final_predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    metrics_df = pd.DataFrame(metrics_rows)

    predictions_file = (
        OUTPUT_DIR / "BlackScholes_predictions.csv"
    )

    metrics_file = (
        OUTPUT_DIR / "BlackScholes_metrics.csv"
    )

    final_predictions.to_csv(
        predictions_file,
        index=False
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )



    print("\n==============================")
    print("BLACK-SCHOLES FINAL RESULTS")
    print("==============================")

    print(metrics_df.to_string(index=False))

    print("\nPrediction sample:")

    print(
        final_predictions[
            [
                "contract_id",
                "option_type",
                "target_price",
                "sigma",
                "predicted_price",
                "absolute_error",
                "model"
            ]
        ].head()
    )

    print("\nPredictions saved to:")
    print(predictions_file)

    print("\nMetrics saved to:")
    print(metrics_file)




if __name__ == "__main__":
    main()
