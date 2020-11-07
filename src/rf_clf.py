import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_df(df):
    df_copy = df.copy()
    df_copy["pred_class"] = df_copy.loc[:, "prob0":"prob3"].values.argmax(axis=1)
    gr = df_copy.groupby("filename")
    return pd.DataFrame({
        "pred_class0": gr["pred_class"].aggregate(lambda x: x[x == 0].count()),
        "pred_class1": gr["pred_class"].aggregate(lambda x: x[x == 1].count()),
        "pred_class2": gr["pred_class"].aggregate(lambda x: x[x == 2].count()),
        "pred_class3": gr["pred_class"].aggregate(lambda x: x[x == 3].count()),
        "prob0_mean": gr["prob0"].mean(),
        "prob0_max": gr["prob0"].max(),
        "prob0_min": gr["prob0"].min(),
        "prob0_var": gr["prob0"].var(),
        "prob0_std": gr["prob0"].std(),

        "prob1_mean": gr["prob1"].mean(),
        "prob1_max": gr["prob1"].max(),
        "prob1_min": gr["prob1"].min(),
        "prob1_var": gr["prob1"].var(),
        "prob1_std": gr["prob1"].std(),

        "prob2_mean": gr["prob2"].mean(),
        "prob2_max": gr["prob2"].max(),
        "prob2_min": gr["prob2"].min(),
        "prob2_var": gr["prob2"].var(),
        "prob2_std": gr["prob2"].std(),

        "prob3_mean": gr["prob3"].mean(),
        "prob3_max": gr["prob3"].max(),
        "prob3_min": gr["prob3"].min(),
        "prob3_var": gr["prob3"].var(),
        "prob3_std": gr["prob3"].std(),

        "t_w_mean": gr["t_w"].mean(),
        "t_w_max": gr["t_w"].max(),
        "t_w_min": gr["t_w"].min(),

        "t_h_mean": gr["t_h"].mean(),
        "t_h_max": gr["t_h"].max(),
        "t_h_min": gr["t_h"].min(),

        "t_r": gr["t_r"].median(),
        "t_g": gr["t_g"].median(),
        "t_b": gr["t_b"].median(),

        "t_n": gr["t_i"].max(),

        "pred_n": gr["filename"].count(),
        "label": gr["label"].max(),
    }).fillna(0)


_RF_PARAMS = {'bootstrap': True,
              'max_depth': 45,
              'max_features': 'auto',
              'min_samples_leaf': 3,
              'min_samples_split': 12,
              'n_estimators': 100,
              'random_state': 42,
              }


def train_random_forest(df):
    df_rf_train = build_df(df)
    clf = RandomForestClassifier(**_RF_PARAMS)
    clf.fit(df_rf_train.iloc[:, 0:-1], df_rf_train.iloc[:, -1])
    return clf


def predict_random_forest(clf, df):
    df_rf = build_df(df)
    cls = clf.predict(df_rf.iloc[:, 0:-1])
    ret = pd.get_dummies(cls)
    ret.insert(0, "filename", df_rf.index)
    return ret
