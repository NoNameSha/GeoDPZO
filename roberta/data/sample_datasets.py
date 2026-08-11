import os
import pandas as pd
import numpy as np

ROOT = "./k-shot-1k-test"
TARGET_DATASETS = ["SST-2", "sst-5", "trec", "RTE", "SNLI", "MNLI"]
TARGET_N = 1024


def robust_read(path):
    sep = "\t" if path.endswith(".tsv") else ","

    try:
        df = pd.read_csv(path, sep=sep)
        if df.shape[1] == 1:
            raise Exception
    except:
        df = pd.read_csv(
            path,
            sep=sep,
            engine="python",
            on_bad_lines="skip"
        )

    return df, sep


def detect_label_col(df):
    for c in ["label", "labels", "gold_label"]:
        if c in df.columns:
            return c

    first_col = df.columns[0]
    if pd.api.types.is_numeric_dtype(df[first_col]):
        return first_col

    raise ValueError("Cannot detect label column")


# ===== stratified sampling =====
def stratified_sample(df, label_col, target_n):
    counts = df[label_col].value_counts()
    total = len(df)

    result = []

    for label, count in counts.items():
        ratio = count / total
        n = int(round(ratio * target_n))

        subset = df[df[label_col] == label]
        n = min(n, len(subset))

        result.append(subset.sample(n=n, random_state=42))

    result = pd.concat(result)

    if len(result) < target_n:
        remain = target_n - len(result)
        extra = df.drop(result.index).sample(n=remain, random_state=42)
        result = pd.concat([result, extra])

    return result.sample(frac=1, random_state=42)


def process_dataset(dataset_name):
    dataset_path = os.path.join(ROOT, dataset_name, "512-42") ### different seed

    if not os.path.isdir(dataset_path):
        print(f"Skip {dataset_name}")
        return

    print(f"\nProcessing {dataset_name}")

    file_path = None
    for fname in ["train.csv", "train.tsv"]:
        candidate = os.path.join(dataset_path, fname)
        if os.path.exists(candidate):
            file_path = candidate
            break

    if file_path is None:
        print("No train file")
        return

    df, sep = robust_read(file_path)

    if "snli" in dataset_name.lower():
        if "gold_label" not in df.columns:
            raise ValueError("SNLI missing gold_label")

        df = df[df["gold_label"] != "-"]
        label_col = "gold_label"

    else:
        label_col = detect_label_col(df)

    print("  Original size:", len(df))
    print("  Label distribution:\n", df[label_col].value_counts())

    sampled_df = stratified_sample(df, label_col, TARGET_N)

    print("  Sampled size:", len(sampled_df))
    print("  New distribution:\n", sampled_df[label_col].value_counts())

    save_path = file_path.replace("train", "train_sampled_1024")
    sampled_df.to_csv(save_path, sep=sep, index=False)

    print("Saved:", save_path)



if __name__ == "__main__":
    for dataset in TARGET_DATASETS:
        process_dataset(dataset)
