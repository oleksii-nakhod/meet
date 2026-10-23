"""Train and test the turn-end classifier (eval/turnend_features.py). Train: AMI meeting series never used in any
evaluation; test: the 12 meetings behind AMI stimulus sets 1 and 2.
Reports AUROC (hold vs shift) for VAP alone and for models with prosodic + lexical features, overall and for
mid-sentence holds (no sentence-final punctuation), plus holds caught at <= 10% of shifts wrongly held
(threshold fixed on the training data). Saves the logistic model to out/turnend/model.json."""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eval.turnend_features import FEATS, OUT  # noqa: E402

tr = json.load(open(os.path.join(OUT, "train.json")))
te = json.load(open(os.path.join(OUT, "test.json")))
X = lambda rows, f: np.array([[r[k] for k in f] for r in rows], float)  # noqa: E731
# target: 1 = HOLD (the speaker will go on: don't answer), score = p(hold)
y_tr = np.array([1 - r["label"] for r in tr]); y_te = np.array([1 - r["label"] for r in te])
mid = np.array([r["label"] == 1 or r["punc"] not in (".", "?", "!") for r in te])
print(f"train {len(tr)} events ({y_tr.mean():.0%} holds), test {len(te)} ({y_te.mean():.0%} holds)")


def at_fpr(scores_tr, ytr, scores, y, fpr=0.10):
    th = np.quantile(scores_tr[ytr == 0], 1 - fpr)  # threshold so that 10% of training shifts are held
    return (scores[y == 1] > th).mean(), (scores[y == 0] > th).mean()


models = {}
vap = ["p_now_1", "p_now_2", "p_fut_1", "p_fut_2"]
for name, feats, kind in [("VAP p_now only", ["p_now_2"], "raw"), ("VAP (4 outputs), logistic", vap, "lr"),
                          ("VAP + prosody + words, logistic", FEATS, "lr"),
                          ("VAP + prosody + words, boosted trees", FEATS, "gb")]:
    a, b = X(tr, feats), X(te, feats)
    if kind == "raw":
        s_tr, s_te = a[:, 0], b[:, 0]
    elif kind == "lr":
        sc = StandardScaler().fit(a)
        m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(a), y_tr)
        s_tr, s_te = m.predict_proba(sc.transform(a))[:, 1], m.predict_proba(sc.transform(b))[:, 1]
        models[name] = (m, sc, feats)
    else:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15).fit(a, y_tr)
        s_tr, s_te = m.predict_proba(a)[:, 1], m.predict_proba(b)[:, 1]
    caught, false = at_fpr(s_tr, y_tr, s_te, y_te)
    print(f"{name:40s} AUROC {roc_auc_score(y_te, s_te):.3f} | mid-sentence holds {roc_auc_score(y_te[mid], s_te[mid]):.3f}"
          f" | at <=10% train FPR: holds caught {caught:.0%}, shifts wrongly held {false:.0%}")

m, sc, feats = models["VAP + prosody + words, logistic"]
w = dict(zip(feats, np.round(m.coef_[0], 3)))
print("logistic weights (standardized):", w)
json.dump({"feats": feats, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": m.coef_[0].tolist(),
           "intercept": float(m.intercept_[0])}, open(os.path.join(OUT, "model.json"), "w"), indent=1)
