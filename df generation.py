import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

BASE_DIR = Path(r"C:\Users\ip471\OneDrive\Documents\GitHub\Poster-Competition")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "MSFT", "AAPL", "GOOGL", "NVDA", "JPM", "BAC", "GS",
    "JNJ", "PFE", "UNH", "AMZN", "WMT", "HD", "XOM",
    "CVX", "TSLA", "F"
]

RISK_FREE_RATE = 0.04
HV_WINDOW = 30
TRADING_DAYS = 252
IV_NEIGHBOURS = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_RELATIVE_SPREAD = 0.5

FEATURE_COLUMNS = [
    "S",
    "K",
    "T",
    "r",
    "sigma",
    "option_type"
]

TARGET_COLUMN = "target_price"


def calculate_historical_volatility(ticker_object):
    history = ticker_object.history(
        period="3mo",
        auto_adjust=True
    )

    if history.empty:
        return None, None

    close = history["Close"].dropna()

    if len(close) < HV_WINDOW + 1:
        return None, None

    returns = np.log(close / close.shift(1)).dropna()
    recent_returns = returns.tail(HV_WINDOW)

    if len(recent_returns) < HV_WINDOW:
        return None, None

    hv = recent_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    spot = float(close.iloc[-1])

    if not np.isfinite(hv) or hv <= 0:
        return None, None

    return spot, float(hv)


def download_real_option_data():
    rows = []
    valuation_date = pd.Timestamp.now().normalize()
    snapshot_time = pd.Timestamp.now()

    print("\nDownloading real option-chain data...")

    for ticker in TICKERS:
        print(f"\nProcessing {ticker}")

        try:
            tk = yf.Ticker(ticker)
            spot, hv = calculate_historical_volatility(tk)

            if spot is None:
                print("Insufficient historical price data.")
                continue

            expiries = tk.options

            for expiry in expiries:
                try:
                    expiry_date = pd.to_datetime(expiry).normalize()
                    days_to_expiry = (expiry_date - valuation_date).days
                    T = days_to_expiry / 365

                    if T <= 0:
                        continue

                    chain = tk.option_chain(expiry)

                    option_tables = [
                        ("call", chain.calls),
                        ("put", chain.puts)
                    ]

                    for option_type, table in option_tables:
                        for _, option in table.iterrows():

                            contract_symbol = option.get(
                                "contractSymbol", None
                            )

                            strike = option["strike"]
                            bid = option["bid"]
                            ask = option["ask"]
                            market_iv = option["impliedVolatility"]

                            if pd.isna(contract_symbol):
                                continue
                            if pd.isna(strike):
                                continue
                            if pd.isna(bid) or pd.isna(ask):
                                continue
                            if pd.isna(market_iv):
                                continue

                            strike = float(strike)
                            bid = float(bid)
                            ask = float(ask)
                            market_iv = float(market_iv)

                            if strike <= 0 or bid <= 0 or ask <= 0:
                                continue
                            if ask < bid:
                                continue
                            if not (0.01 < market_iv < 5.0):
                                continue

                            midpoint = (bid + ask) / 2

                            if midpoint <= 0:
                                continue

                            relative_spread = (ask - bid) / midpoint

                            if relative_spread > MAX_RELATIVE_SPREAD:
                                continue

                            rows.append({
                                "ticker": ticker,
                                "contract_symbol": str(contract_symbol),
                                "quote_date": valuation_date,
                                "snapshot_time": snapshot_time,
                                "expiry": expiry_date,
                                "option_type": option_type,
                                "S": spot,
                                "K": strike,
                                "T": float(T),
                                "r": RISK_FREE_RATE,
                                "HV": hv,
                                "market_iv_peer": market_iv,
                                "bid": bid,
                                "ask": ask,
                                "target_price": midpoint,
                                "relative_spread": relative_spread
                            })

                except Exception as e:
                    print(f"Expiry error {expiry}: {e}")

        except Exception as e:
            print(f"Ticker error {ticker}: {e}")

    real_df = pd.DataFrame(rows)

    if real_df.empty:
        raise ValueError("No valid option data was downloaded.")

    real_df = real_df.drop_duplicates(
        subset=["contract_symbol"]
    ).copy()

    print(f"\nDownloaded contracts: {len(real_df):,}")

    return real_df


def add_iv_surface(target_df, reference_df, exclude_self=False):
    target_df = target_df.copy()
    target_df["IV_surface"] = np.nan

    for (ticker, option_type), targets in target_df.groupby(
        ["ticker", "option_type"]
    ):

        references = reference_df[
            (reference_df["ticker"] == ticker) &
            (reference_df["option_type"] == option_type)
        ].copy()

        if len(references) < 3:
            continue

        reference_coordinates = np.column_stack([
            np.log(references["K"].values / references["S"].values),
            references["T"].values
        ])

        target_coordinates = np.column_stack([
            np.log(targets["K"].values / targets["S"].values),
            targets["T"].values
        ])

        scaler = StandardScaler()

        scaled_ref_coords = scaler.fit_transform(
            reference_coordinates
        )

        scaled_target_coords = scaler.transform(
            target_coordinates
        )

        distances = pairwise_distances(
            scaled_target_coords,
            scaled_ref_coords
        )

        if exclude_self:
            ref_ids = references["contract_id"].values
            target_ids = targets["contract_id"].values

            self_mask = (
                target_ids[:, None] == ref_ids[None, :]
            )

            distances[self_mask] = np.inf

        iv_values = references["market_iv_peer"].values
        surface_values = []

        k_neighbours = min(
            IV_NEIGHBOURS,
            len(references) - (1 if exclude_self else 0)
        )

        if k_neighbours < 2:
            continue

        for i in range(len(targets)):

            nearest_indices = np.argsort(distances[i])[:k_neighbours]

            neighbour_distances = distances[i, nearest_indices]
            neighbour_ivs = iv_values[nearest_indices]

            valid = (
                np.isfinite(neighbour_distances) &
                np.isfinite(neighbour_ivs)
            )

            neighbour_distances = neighbour_distances[valid]
            neighbour_ivs = neighbour_ivs[valid]

            if len(neighbour_ivs) < 2:
                surface_values.append(np.nan)
                continue

            weights = (
                1 / np.maximum(neighbour_distances, 1e-6) ** 2
            )

            estimated_iv = (
                np.sum(weights * neighbour_ivs) /
                np.sum(weights)
            )

            surface_values.append(float(estimated_iv))

        target_df.loc[targets.index, "IV_surface"] = surface_values

    target_df = target_df.dropna(
        subset=["IV_surface"]
    ).copy()

    return target_df


def validate_datasets(hv_train, hv_test, iv_train, iv_test):
    datasets = {
        "HV train": hv_train,
        "HV test": hv_test,
        "IV train": iv_train,
        "IV test": iv_test
    }

    for name, dataset in datasets.items():
        print(f"\n{name}")
        print(f"Rows: {len(dataset)}")
        print(f"Missing values:\n{dataset.isna().sum()}")

        assert dataset["sigma"].notna().all()
        assert np.isfinite(dataset["sigma"]).all()
        assert dataset["target_price"].notna().all()
        assert np.isfinite(dataset["target_price"]).all()
        assert (dataset["target_price"] > 0).all()
        assert (dataset["sigma"] > 0).all()

    assert hv_train["contract_id"].tolist() == iv_train["contract_id"].tolist()
    assert hv_test["contract_id"].tolist() == iv_test["contract_id"].tolist()

    assert np.allclose(
        hv_train["target_price"],
        iv_train["target_price"]
    )

    assert np.allclose(
        hv_test["target_price"],
        iv_test["target_price"]
    )

    print("\nDataset validation completed successfully.")


def create_real_train_test():

    real_df = download_real_option_data()

    real_df["contract_id"] = real_df["contract_symbol"].astype(str)

    real_df["strata"] = (
        real_df["ticker"].astype(str) + "_" +
        real_df["option_type"].astype(str)
    )

    strata_counts = real_df["strata"].value_counts()

    valid_strata = strata_counts[
        strata_counts >= 5
    ].index

    real_df = real_df[
        real_df["strata"].isin(valid_strata)
    ].copy()

    if real_df.empty:
        raise ValueError(
            "No ticker/option-type groups have sufficient observations."
        )

    train_df, test_df = train_test_split(
        real_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=real_df["strata"]
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df.drop(columns=["strata"], inplace=True)
    test_df.drop(columns=["strata"], inplace=True)

    print("\nConstructing independent IV surfaces...")

    train_df = add_iv_surface(
        train_df,
        train_df,
        exclude_self=True
    )

    test_df = add_iv_surface(
        test_df,
        train_df,
        exclude_self=False
    )

    if train_df.empty or test_df.empty:
        raise ValueError(
            "No contracts had sufficient neighbouring IV observations."
        )

    combined_df = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    combined_df.to_csv(
        DATA_DIR / "real_option_data_with_volatility.csv",
        index=False
    )

    hv_train = train_df.copy()
    hv_test = test_df.copy()

    hv_train["sigma"] = hv_train["HV"]
    hv_test["sigma"] = hv_test["HV"]

    iv_train = train_df.copy()
    iv_test = test_df.copy()

    iv_train["sigma"] = iv_train["IV_surface"]
    iv_test["sigma"] = iv_test["IV_surface"]

    volatility_columns = [
        "HV",
        "IV_surface",
        "market_iv_peer"
    ]

    for dataset in [hv_train, hv_test, iv_train, iv_test]:
        dataset.drop(
            columns=volatility_columns,
            inplace=True
        )

    hv_train.to_csv(
        DATA_DIR / "real_hv_train.csv",
        index=False
    )

    hv_test.to_csv(
        DATA_DIR / "real_hv_test.csv",
        index=False
    )

    iv_train.to_csv(
        DATA_DIR / "real_iv_train.csv",
        index=False
    )

    iv_test.to_csv(
        DATA_DIR / "real_iv_test.csv",
        index=False
    )

    validate_datasets(
        hv_train,
        hv_test,
        iv_train,
        iv_test
    )

    print(f"\nSaved datasets to {DATA_DIR}")
    print(f"Train contracts: {len(train_df):,}")
    print(f"Test contracts: {len(test_df):,}")

    return hv_train, hv_test, iv_train, iv_test


if __name__ == "__main__":
    create_real_train_test()