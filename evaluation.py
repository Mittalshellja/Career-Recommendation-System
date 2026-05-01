import os
import numpy as np
import pandas as pd

from main import generate_recommendations
from resume_parser import extract_text_from_file

# ===============================
# LOAD DATASET (FOLDER STRUCTURE)
# ===============================
def load_dataset(base_path):
    data = []

    for role in os.listdir(base_path):
        role_path = os.path.join(base_path, role)

        if not os.path.isdir(role_path):
            continue

        for file in os.listdir(role_path):
            if file.endswith(".pdf"):
                data.append({
                    "path": os.path.join(role_path, file),
                    "true_role": role
                })

    return data


# ===============================
# READ PDF
# ===============================
def read_resume(path):
    try:
        with open(path, "rb") as f:
            text, error = extract_text_from_file(f)
            if error:
                return ""
            return text
    except:
        return ""


# ===============================
# RUN EVALUATION
# ===============================
def evaluate(base_path):

    dataset = load_dataset(base_path)
    results = []

    for sample in dataset:

        resume_text = read_resume(sample["path"])

        if not resume_text:
            continue

        output = generate_recommendations(
            resume_text,
            field_of_interest="",
            experience_years=0,
            min_score=0.0,
            top_k=5
        )

        preds = [r["role"] for r in output["recommendations"]]

        if not preds:
            continue

        results.append({
            "true": sample["true_role"].lower(),
            "pred": [p.lower() for p in preds]
        })

    return results


# ===============================
# METRICS
# ===============================
def compute_metrics(results):

    total = len(results)

    # Accuracy
    correct = sum(1 for r in results if r["true"] == r["pred"][0])
    accuracy = correct / total

    # Top-3 Accuracy
    top3 = sum(1 for r in results if r["true"] in r["pred"][:3])
    top3_acc = top3 / total

    # Precision / Recall / F1
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true = [r["true"] for r in results]
    y_pred = [r["pred"][0] for r in results]

    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # NDCG
    def dcg(rel):
        return sum(r / np.log2(i+2) for i, r in enumerate(rel))

    def ndcg(results, k):
        scores = []

        for r in results:
            rel = [1 if role == r["true"] else 0 for role in r["pred"][:k]]
            ideal = sorted(rel, reverse=True)

            if sum(ideal) == 0:
                continue

            scores.append(dcg(rel) / dcg(ideal))

        return sum(scores) / len(scores)

    ndcg3 = ndcg(results, 3)
    ndcg5 = ndcg(results, 5)

    return {
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f1, 3),
        "Top3 Accuracy": round(top3_acc, 3),
        "NDCG@3": round(ndcg3, 3),
        "NDCG@5": round(ndcg5, 3),
    }


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    base_path = "validation_Labeled"

    results = evaluate(base_path)

    metrics = compute_metrics(results)

    print("\n===== FINAL METRICS =====\n")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save to CSV (for Streamlit UI)
    pd.DataFrame([metrics]).to_csv("labeled_metrics.csv", index=False)