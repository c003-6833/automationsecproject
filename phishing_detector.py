"""
Machine Learning-Based Phishing URL Detection
A URL Structure and Domain-Age Approach

Usage:
    python phishing_detector.py                      # train + evaluate, save results
    python phishing_detector.py --predict "http://secure-login.example-bank.xyz/a@b" --age 12
"""
import argparse
import re
import os
from urllib.parse import urlparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "dataset_phishing.csv"
OUT_DIR = "results"
SEED = 42
IP_REGEX = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


# ---------------------------------------------------------------------------
# 1. Feature extraction (lexical features: computed from the URL string only)
# ---------------------------------------------------------------------------
def get_hostname(url):
    """Return the hostname of a URL (adds a scheme if it is missing)."""
    if "://" not in url:
        url = "http://" + url
    return (urlparse(url).hostname or "").lower()


def lexical_features(url):
    """Turn one URL string into a dictionary of numeric features."""
    host = get_hostname(url)
    parsed = urlparse(url if "://" in url else "http://" + url)
    labels = [p for p in host.replace("www.", "", 1).split(".") if p]

    return {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(parsed.path),
        "num_digits": sum(c.isdigit() for c in url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_special_chars": sum(not c.isalnum() for c in url),
        "num_slashes": url.count("/"),
        "num_query_params": url.count("=") + url.count("?"),
        "num_subdomains": max(len(labels) - 2, 0),
        "has_ip_address": int(bool(IP_REGEX.match(host))),
        "has_at_symbol": int("@" in url),
        "uses_https": int(url.lower().startswith("https")),
    }


def build_features(urls, domain_age_days):
    """Lexical features + domain age. A missing WHOIS record is stored as -1
    and flagged, so it is handled the same way every time."""
    X = pd.DataFrame([lexical_features(u) for u in urls])
    age = pd.Series(domain_age_days).reset_index(drop=True)
    X["domain_age_days"] = age.where(age > 0, -1)
    X["domain_age_missing"] = (age <= 0).astype(int)
    return X


LEXICAL_COLS = list(lexical_features("http://a.com").keys())


# ---------------------------------------------------------------------------
# 2. Data loading and cleaning
# ---------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, usecols=["url", "domain_age", "status"])
    before = len(df)
    df = df.drop_duplicates(subset="url").dropna().reset_index(drop=True)
    print(f"[data] {before} rows loaded, {before - len(df)} duplicate/invalid removed -> {len(df)} kept")
    print("[data] class balance:\n" + df["status"].value_counts().to_string())
    y = (df["status"] == "phishing").astype(int)          # 1 = phishing
    X = build_features(df["url"], df["domain_age"])
    groups = df["url"].map(get_hostname).str.replace(r"^www\.", "", regex=True)
    return df, X, y, groups


# ---------------------------------------------------------------------------
# 3. Models and metrics
# ---------------------------------------------------------------------------
def make_rf():
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def blacklist_baseline(train_hosts, train_y, test_hosts):
    """Static blacklist: flag a URL only if its hostname was already known
    to be phishing in the training data. Everything else is 'legitimate'."""
    known_bad = set(train_hosts[train_y == 1])
    return test_hosts.isin(known_bad).astype(int).values


# ---------------------------------------------------------------------------
# 4. Experiment
# ---------------------------------------------------------------------------
def run_split(name, X, y, hosts, train_idx, test_idx, results):
    Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    print(f"\n================ {name} ================")
    print(f"train = {len(Xtr)}  test = {len(Xte)}")
    fitted = {}

    for label, cols in [("Lexical only", LEXICAL_COLS), ("Lexical + domain age", list(X.columns))]:
        for mname, make in [("Random Forest", make_rf), ("Logistic Regression", make_lr)]:
            model = make().fit(Xtr[cols], ytr)
            m = metrics(yte, model.predict(Xte[cols]))
            results.append({"split": name, "features": label, "model": mname, **m})
            fitted[(label, mname)] = (model, cols)
            print(f"{mname:20s} | {label:21s} | acc={m['accuracy']:.4f} "
                  f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")

    m = metrics(yte, blacklist_baseline(hosts.iloc[train_idx], ytr, hosts.iloc[test_idx]))
    results.append({"split": name, "features": "Known-host list", "model": "Blacklist baseline", **m})
    print(f"{'Blacklist baseline':20s} | {'Known-host list':21s} | acc={m['accuracy']:.4f} "
          f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")
    return fitted


def main():
    parser = argparse.ArgumentParser(description="ML phishing URL detector")
    parser.add_argument("--predict", help="URL to classify with the saved model")
    parser.add_argument("--age", type=int, default=-1,
                        help="Domain age in days (WHOIS); -1 if unknown")
    args = parser.parse_args()

    if args.predict:
        model, cols = joblib.load(os.path.join(OUT_DIR, "rf_model.joblib"))
        X = build_features([args.predict], [args.age])[cols]
        p = model.predict_proba(X)[0, 1]
        print(f"{args.predict}\n-> {'PHISHING' if p >= 0.5 else 'LEGITIMATE'} (phishing probability {p:.2f})")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    df, X, y, hosts = load_data()
    results = []

    # --- A) Standard random 80/20 split (URL level) ------------------------
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
    fitted = run_split("Random split (80/20)", X, y, hosts, tr, te, results)

    # 5-fold cross-validation on the training set (overfitting check)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("\n[cross-validation, 5-fold, training set only, accuracy]")
    for mname, make in [("Random Forest", make_rf), ("Logistic Regression", make_lr)]:
        s = cross_val_score(make(), X.iloc[tr], y.iloc[tr], cv=cv, scoring="accuracy")
        print(f"{mname:20s} mean={s.mean():.4f} std={s.std():.4f}  folds={np.round(s, 4)}")

    # --- B) Domain-level split: test domains never seen in training --------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_g, te_g = next(gss.split(X, y, groups=hosts))
    run_split("Domain-level split (unseen domains)", X, y, hosts, tr_g, te_g, results)

    # --- Final model: Random Forest, lexical + domain age -------------------
    rf, cols = fitted[("Lexical + domain age", "Random Forest")]
    joblib.dump((rf, cols), os.path.join(OUT_DIR, "rf_model.joblib"))
    Xte, yte = X.iloc[te][cols], y.iloc[te]
    pred = rf.predict(Xte)

    # Confusion matrix
    cm = confusion_matrix(yte, pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n[confusion matrix: Random Forest, random split]\nTN={tn} FP={fp} FN={fn} TP={tp}")
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Random Forest - confusion matrix")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=200); plt.close(fig)

    # Feature importance (impurity-based and permutation-based)
    imp = pd.Series(rf.feature_importances_, index=cols)
    perm = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=SEED, n_jobs=-1)
    imp_df = pd.DataFrame({"impurity_importance": imp,
                           "permutation_importance": perm.importances_mean}).sort_values(
                           "impurity_importance", ascending=False)
    imp_df.to_csv(f"{OUT_DIR}/feature_importance.csv")
    print("\n[feature importance, top 10]\n" + imp_df.head(10).round(4).to_string())
    top = imp_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(top.index, top["impurity_importance"], color="#2f5597")
    ax.set_xlabel("Importance (Random Forest)"); ax.set_title("Top 10 features")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/feature_importance.png", dpi=200); plt.close(fig)

    # Model comparison chart
    res = pd.DataFrame(results)
    res.to_csv(f"{OUT_DIR}/all_results.csv", index=False)
    labels = [("Random Forest", "Lexical only"), ("Random Forest", "Lexical + domain age"),
              ("Logistic Regression", "Lexical + domain age"), ("Blacklist baseline", "Known-host list")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.38
    for k, split in enumerate(res["split"].unique()):
        vals = [res[(res.split == split) & (res.model == m) & (res.features == f)]["accuracy"].iloc[0]
                for m, f in labels]
        bars = ax.bar(np.arange(4) + (k - 0.5) * width, vals, width, label=split,
                      color=["#2f5597", "#c55a11"][k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.90, ls="--", c="grey", lw=1); ax.text(3.45, 0.915, "90% target", fontsize=8, ha="right")
    ax.set_xticks(range(4), ["RF\nlexical only", "RF\nlexical + age", "Log. Reg.\nlexical + age", "Blacklist\nbaseline"])
    ax.set_ylim(0, 1.2); ax.set_ylabel("Accuracy"); ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.set_title("Accuracy by model and split")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/model_comparison.png", dpi=200); plt.close(fig)

    # Error review: false positives and false negatives
    test_urls = df["url"].iloc[te].reset_index(drop=True)
    errs = pd.DataFrame({"url": test_urls, "actual": yte.values, "predicted": pred})
    fps = errs[(errs.actual == 0) & (errs.predicted == 1)]
    fns = errs[(errs.actual == 1) & (errs.predicted == 0)]
    errs[errs.actual != errs.predicted].to_csv(f"{OUT_DIR}/misclassified_urls.csv", index=False)
    print("\n[sample false positives (legitimate flagged as phishing)]")
    for u in fps["url"].head(5): print("  ", u[:110])
    print("[sample false negatives (phishing missed)]")
    for u in fns["url"].head(5): print("  ", u[:110])

    # Domain-age summary for the report
    print("\n[median domain age in days, known records only]")
    known = df[df["domain_age"] > 0]
    print(known.groupby("status")["domain_age"].median().to_string())


if __name__ == "__main__":
    main()
