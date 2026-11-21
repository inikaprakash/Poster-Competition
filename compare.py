
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import wilcoxon




BASE_DIR = Path(
    r"C:\Users\ip471\OneDrive\Documents\GitHub\Poster-Competition"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)




MODEL_FILES = {
    "Black-Scholes": "BlackScholes_predictions.csv",
    "Random Forest": "RandomForest_predictions.csv",
    "XGBoost": "XGBoost_predictions.csv",
    "CatBoost": "CatBoost_predictions.csv",
    "LSTM": "LSTM_predictions.csv"
}

BOOTSTRAP_ITERATIONS = 10000
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95




def get_prediction_column(df):

    possible_columns = [
        "predicted_price",
        "pred",
        "prediction",
        "predicted"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify prediction column. "
        f"Available columns: {list(df.columns)}"
    )




def load_paired_predictions(model_name, filename):

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "contract_id",
        "ticker",
        "target_price",
        "model"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {missing}"
        )

    prediction_column = get_prediction_column(df)

   
    model_labels = df["model"].astype(str).str.upper()

    iv_df = df[
        model_labels.str.endswith("_IV")
    ].copy()

    hv_df = df[
        model_labels.str.endswith("_HV")
    ].copy()

    if iv_df.empty or hv_df.empty:
        raise ValueError(
            f"{model_name}: IV or HV predictions not found."
        )

  
    iv_df = iv_df[
        [
            "contract_id",
            "ticker",
            "target_price",
            prediction_column
        ]
    ].rename(
        columns={
            "target_price": "target_IV",
            prediction_column: "predicted_IV"
        }
    )

    hv_df = hv_df[
        [
            "contract_id",
            "ticker",
            "target_price",
            prediction_column
        ]
    ].rename(
        columns={
            "target_price": "target_HV",
            prediction_column: "predicted_HV"
        }
    )

 
    if iv_df["contract_id"].duplicated().any():
        raise ValueError(
            f"{model_name}: duplicate IV contract IDs."
        )

    if hv_df["contract_id"].duplicated().any():
        raise ValueError(
            f"{model_name}: duplicate HV contract IDs."
        )

  
    iv_ids = set(iv_df["contract_id"])
    hv_ids = set(hv_df["contract_id"])

    if iv_ids != hv_ids:
        raise ValueError(
            f"{model_name}: IV and HV predictions "
            "do not contain identical contract IDs."
        )


    paired = pd.merge(
        iv_df,
        hv_df,
        on="contract_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_IV", "_HV")
    )


    if not (
        paired["ticker_IV"] == paired["ticker_HV"]
    ).all():
        raise ValueError(
            f"{model_name}: ticker mismatch between IV and HV."
        )

    paired["ticker"] = paired["ticker_IV"]

  
    if not np.allclose(
        paired["target_IV"],
        paired["target_HV"],
        rtol=1e-8,
        atol=1e-10
    ):
        raise ValueError(
            f"{model_name}: IV and HV actual prices differ."
        )

    paired["target_price"] = paired["target_IV"]

  
    numeric_columns = [
        "target_price",
        "predicted_IV",
        "predicted_HV"
    ]

    if not np.isfinite(
        paired[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{model_name}: non-finite predictions found."
        )

    if (paired["target_price"] <= 0).any():
        raise ValueError(
            f"{model_name}: non-positive market prices found."
        )

    print(
        f"{model_name}: {len(paired)} matched contracts "
        f"across {paired['ticker'].nunique()} equities."
    )

    return paired




def calculate_errors(df):

    df = df.copy()

    df["absolute_error_IV"] = np.abs(
        df["target_price"] - df["predicted_IV"]
    )

    df["absolute_error_HV"] = np.abs(
        df["target_price"] - df["predicted_HV"]
    )

    df["squared_error_IV"] = (
        df["target_price"] - df["predicted_IV"]
    ) ** 2

    df["squared_error_HV"] = (
        df["target_price"] - df["predicted_HV"]
    ) ** 2

    return df




def paired_cluster_bootstrap(
    df,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=RANDOM_STATE
):

    rng = np.random.default_rng(seed)


    ticker_groups = {
        ticker: group.index.to_numpy()
        for ticker, group in df.groupby("ticker")
    }

    tickers = list(ticker_groups.keys())
    number_of_tickers = len(tickers)

    if number_of_tickers < 2:
        raise ValueError(
            "At least two ticker clusters are required."
        )


    abs_iv = df["absolute_error_IV"].to_numpy()
    abs_hv = df["absolute_error_HV"].to_numpy()

    sq_iv = df["squared_error_IV"].to_numpy()
    sq_hv = df["squared_error_HV"].to_numpy()

    index_positions = {
        index: position
        for position, index in enumerate(df.index)
    }

    ticker_positions = {
        ticker: np.array([
            index_positions[index]
            for index in indices
        ])
        for ticker, indices in ticker_groups.items()
    }

    mae_differences = []
    rmse_differences = []

    for _ in range(iterations):


        sampled_tickers = rng.choice(
            tickers,
            size=number_of_tickers,
            replace=True
        )

        sampled_positions = np.concatenate([
            ticker_positions[ticker]
            for ticker in sampled_tickers
        ])

        mae_iv = np.mean(abs_iv[sampled_positions])
        mae_hv = np.mean(abs_hv[sampled_positions])

        mae_differences.append(
            mae_iv - mae_hv
        )

        rmse_iv = np.sqrt(
            np.mean(sq_iv[sampled_positions])
        )

        rmse_hv = np.sqrt(
            np.mean(sq_hv[sampled_positions])
        )

        rmse_differences.append(
            rmse_iv - rmse_hv
        )

    alpha = 1 - CONFIDENCE_LEVEL

    mae_ci = np.percentile(
        mae_differences,
        [
            100 * alpha / 2,
            100 * (1 - alpha / 2)
        ]
    )

    rmse_ci = np.percentile(
        rmse_differences,
        [
            100 * alpha / 2,
            100 * (1 - alpha / 2)
        ]
    )

    return {
        "MAE_CI_lower": mae_ci[0],
        "MAE_CI_upper": mae_ci[1],
        "RMSE_CI_lower": rmse_ci[0],
        "RMSE_CI_upper": rmse_ci[1]
    }




def wilcoxon_test(iv_errors, hv_errors):

    differences = (
        np.asarray(iv_errors) - np.asarray(hv_errors)
    )


    if np.allclose(differences, 0):
        return 1.0

    result = wilcoxon(
        iv_errors,
        hv_errors,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )

    return result.pvalue



def holm_adjust(p_values):

    p_values = np.asarray(p_values, dtype=float)

    number_of_tests = len(p_values)

    sorted_indices = np.argsort(p_values)

    sorted_p = p_values[sorted_indices]

    adjusted_sorted = np.maximum.accumulate(
        (number_of_tests - np.arange(number_of_tests))
        * sorted_p
    )

    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0
    )

    adjusted = np.empty(number_of_tests)

    adjusted[sorted_indices] = adjusted_sorted

    return adjusted




def analyse_model(model_name, filename):

    paired = load_paired_predictions(
        model_name,
        filename
    )

    paired = calculate_errors(paired)


    mae_iv = paired["absolute_error_IV"].mean()
    mae_hv = paired["absolute_error_HV"].mean()

    rmse_iv = np.sqrt(
        paired["squared_error_IV"].mean()
    )

    rmse_hv = np.sqrt(
        paired["squared_error_HV"].mean()
    )

    mae_difference = mae_iv - mae_hv
    rmse_difference = rmse_iv - rmse_hv


    bootstrap_results = paired_cluster_bootstrap(
        paired
    )


    mae_p = wilcoxon_test(
        paired["absolute_error_IV"],
        paired["absolute_error_HV"]
    )

 
    rmse_p = wilcoxon_test(
        paired["squared_error_IV"],
        paired["squared_error_HV"]
    )

    results = {
        "model": model_name,
        "contracts": len(paired),
        "tickers": paired["ticker"].nunique(),

        "MAE_IV": mae_iv,
        "MAE_HV": mae_hv,
        "MAE_difference_IV_minus_HV": mae_difference,

        "MAE_CI_lower": bootstrap_results["MAE_CI_lower"],
        "MAE_CI_upper": bootstrap_results["MAE_CI_upper"],

        "MAE_Wilcoxon_p": mae_p,

        "RMSE_IV": rmse_iv,
        "RMSE_HV": rmse_hv,
        "RMSE_difference_IV_minus_HV": rmse_difference,

        "RMSE_CI_lower": bootstrap_results["RMSE_CI_lower"],
        "RMSE_CI_upper": bootstrap_results["RMSE_CI_upper"],

        "RMSE_Wilcoxon_p": rmse_p
    }

    return results, paired




def main():

    all_results = []
    all_paired_predictions = []



    for model_name, filename in MODEL_FILES.items():


        if not (OUTPUT_DIR / filename).exists():
            if model_name == "LSTM":
                print(
                    "\nLSTM prediction file not found. "
                    "Skipping LSTM."
                )
                continue

            raise FileNotFoundError(
                f"Required file missing: {filename}"
            )

        print("\n" + "=" * 55)
        print(f"ANALYSING: {model_name}")
        print("=" * 55)

        results, paired = analyse_model(
            model_name,
            filename
        )

        all_results.append(results)

        paired["model_type"] = model_name

        all_paired_predictions.append(paired)

    if not all_results:
        raise ValueError("No model results were generated.")



    results_df = pd.DataFrame(all_results)


    p_values = []

    for _, row in results_df.iterrows():

        p_values.append(row["MAE_Wilcoxon_p"])
        p_values.append(row["RMSE_Wilcoxon_p"])

    adjusted_p_values = holm_adjust(p_values)

    results_df["MAE_Holm_adjusted_p"] = (
        adjusted_p_values[0::2]
    )

    results_df["RMSE_Holm_adjusted_p"] = (
        adjusted_p_values[1::2]
    )



    results_df["MAE_significant"] = (
        results_df["MAE_Holm_adjusted_p"] < 0.05
    )

    results_df["RMSE_significant"] = (
        results_df["RMSE_Holm_adjusted_p"] < 0.05
    )



    results_df["MAE_CI_excludes_zero"] = (
        (results_df["MAE_CI_lower"] > 0)
        | (results_df["MAE_CI_upper"] < 0)
    )

    results_df["RMSE_CI_excludes_zero"] = (
        (results_df["RMSE_CI_lower"] > 0)
        | (results_df["RMSE_CI_upper"] < 0)
    )



    results_df.to_csv(
        OUTPUT_DIR / "Statistical_Significance_Results.csv",
        index=False
    )

    paired_predictions = pd.concat(
        all_paired_predictions,
        ignore_index=True
    )

    paired_predictions.to_csv(
        OUTPUT_DIR / "Paired_Model_Errors.csv",
        index=False
    )



    display_columns = [
        "model",
        "MAE_difference_IV_minus_HV",
        "MAE_CI_lower",
        "MAE_CI_upper",
        "MAE_Wilcoxon_p",
        "MAE_Holm_adjusted_p",
        "RMSE_difference_IV_minus_HV",
        "RMSE_CI_lower",
        "RMSE_CI_upper",
        "RMSE_Wilcoxon_p",
        "RMSE_Holm_adjusted_p"
    ]

    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE RESULTS")
    print("=" * 100)

    print(
        results_df[display_columns]
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("SIGNIFICANCE INTERPRETATION")
    print("=" * 100)

    for _, row in results_df.iterrows():

        print(f"\n{row['model']}")

        print(
            "MAE: "
            + (
                "Statistically significant"
                if row["MAE_significant"]
                else "Not statistically significant"
            )
        )

        print(
            "RMSE: "
            + (
                "Statistically significant"
                if row["RMSE_significant"]
                else "Not statistically significant"
            )
        )

    print("\nFiles saved:")
    print(
        OUTPUT_DIR / "Statistical_Significance_Results.csv"
    )
    print(
        OUTPUT_DIR / "Paired_Model_Errors.csv"
    )




if __name__ == "__main__":
    main()
