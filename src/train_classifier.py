"""
Train and evaluate a Random Forest Classifier to predict a song's popularity tier
(Low: 0-33.32, Medium: 33.33-66.65, High: 66.66-100) from Spotify audio/metadata
features, and visualize feature importance + a sample decision tree.

Expects ../data/processed_tracks_new.csv (produced by data_preparation.py) with
a numeric "popularity" column (0-100).
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.tree import plot_tree

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed_tracks_new.csv")

# Tier boundaries for discretizing the continuous 0-100 popularity score
LOW_MAX = 33.33
MED_MAX = 66.66
TIER_LABELS = ["0-33.32", "33.33-66.65", "66.66-100"]

# Cap on how many Low/Medium-tier samples to keep, to reduce class imbalance
# against the naturally rarer High-popularity tier (see writeup, Section 3.2).
MAX_PER_MAJORITY_CLASS = 40000


def load_and_prepare_data(path: str):
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    x = df.drop(columns=["popularity"])
    y = df["popularity"].copy()

    med_count, low_count = 0, 0
    drop_idx = []
    for i in y.index:
        if y[i] >= MED_MAX:
            y.loc[i] = 3
        elif y[i] >= LOW_MAX:
            y.loc[i] = 2
            med_count += 1
            if med_count < MAX_PER_MAJORITY_CLASS:
                drop_idx.append(i)
        else:
            y.loc[i] = 1
            low_count += 1
            if low_count < MAX_PER_MAJORITY_CLASS:
                drop_idx.append(i)

    x = x.drop(index=drop_idx)
    y = y.drop(index=drop_idx)
    return x, y


def visualize_classification_report(report: dict):
    metrics = ["precision", "recall", "f1-score"]
    labels, data = [], {m: [] for m in metrics}
    for cls, values in report.items():
        if cls.isdigit() or cls in ("macro avg", "weighted avg"):
            labels.append(cls)
            for m in metrics:
                data[m].append(values.get(m))

    metrics_df = pd.DataFrame(data, index=labels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric in zip(axes, metrics):
        sns.barplot(ax=ax, x=metrics_df.index, y=metric, data=metrics_df)
        ax.set_title(f"{metric.capitalize()} by Class")
        ax.set_ylim(0, 1.1)
    plt.tight_layout()


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    order = sorted(range(len(importances)), key=lambda i: importances[i])
    plt.figure()
    plt.barh(range(len(importances)), [importances[i] for i in order])
    plt.yticks(range(len(importances)), [feature_names[i] for i in order])
    plt.title("Feature Importance in Spotify Musics")


def main():
    x, y = load_and_prepare_data(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = RandomForestClassifier(n_estimators=150, random_state=0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    visualize_classification_report(report)

    plot_feature_importance(model, x.columns.tolist())

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TIER_LABELS).plot()

    plt.figure(figsize=(12, 8))
    plot_tree(
        model.estimators_[0],
        feature_names=x.columns.tolist(),
        class_names=TIER_LABELS,
        filled=True,
        max_depth=2,
    )

    plt.show()


if __name__ == "__main__":
    main()
