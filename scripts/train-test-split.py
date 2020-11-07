from argparse import ArgumentParser
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd


def folds_iter(fold_n, y, seed):
    skf = StratifiedKFold(n_splits=fold_n, random_state=seed, shuffle=True)
    x_fake = np.zeros(len(y))
    for train_index, test_index in skf.split(x_fake, y):
        yield train_index, test_index


def main(args):
    df_labels = pd.read_csv(args.csv_path / "train_labels.csv")
    labels = df_labels.iloc[:, 1:].values.argmax(axis=1)
    train_index, test_index = next(folds_iter(args.folds_n, labels, args.seed))
    train_filenames = df_labels.iloc[train_index].filename
    test_filenames = df_labels.iloc[test_index].filename

    for fn in ("train_annotations.csv", "train_labels.csv", "train_metadata.csv"):
        df = pd.read_csv(args.csv_path / fn)
        file_type = fn.split("_")[-1]
        df_train = df[df.filename.isin(train_filenames)]
        df_train.to_csv(args.output / f"train_{file_type}", index=False)
        df_test = df[df.filename.isin(test_filenames)]
        df_test.to_csv(args.output / f"test_{file_type}", index=False)

    df_train_submission_fmt = pd.read_csv(args.output / "train_labels.csv")
    df_train_submission_fmt.iloc[:, 1:] = 0
    df_train_submission_fmt.to_csv(args.output / "train_submission_format.csv", index=False)

    df_test_submission_fmt = pd.read_csv(args.output / "test_labels.csv")
    df_test_submission_fmt.iloc[:, 1:] = 0
    df_test_submission_fmt.to_csv(args.output / "submission_format.csv", index=False)
    print(f"Split train {len(train_filenames)} test {len(test_filenames)}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("csv_path", type=lambda x: Path(x))
    parser.add_argument("output", type=lambda x: Path(x))
    parser.add_argument("--folds-n", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    main(parser.parse_args())
