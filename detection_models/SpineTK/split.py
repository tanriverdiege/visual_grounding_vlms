from sklearn.model_selection import train_test_split
import pandas as pd


def make_split(
    baseline_directory: str,
    val_fraction: float = 0.08 / 0.7,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Both fractions are of the whole dataset. The defaults reproduce the
    # original ~70/10/20 split (test 0.2, then val 10/70 of the rest).

    with open(f"{baseline_directory}/metadata.json") as meta_file:
        df = pd.read_json(meta_file, orient="index")
        X_train_p, X_test = train_test_split(
            df, test_size=test_fraction, random_state=100, shuffle=True
        )
        X_train, X_val = train_test_split(
            X_train_p,
            test_size=val_fraction / (1 - test_fraction),
            random_state=100, shuffle=True
        )

    return X_train, X_val, X_test


if __name__ == "__main__":
    BASELINE_DIRECTORY = "/data/datasets/csxa/spinetk_baseline"
    X_train, X_val, X_test = make_split(BASELINE_DIRECTORY)
    print("train num:", len(X_train))
    print("val num:", len(X_val))
    print("test num:", len(X_test))
