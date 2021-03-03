import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import warnings
from models import ResNet34
from models import DenseNet121


def return_model(mode, **kwargs):
    """returns model"""
    if mode == 'ResNet':
        model = ResNet34()
    elif mode == 'DenseNet':
        model = DenseNet121()
    else:
        raise ValueError("Invalid mode!")
    return model


def one_iteration(clf, X, y, X_test, y_test, mean_score, tol=0.0, c=None, metric='accuracy'):
    """runs one iteration of TMC-Shapley"""

    if metric == 'auc':
        def score_func(clf, a, b):
            return roc_auc_score(b, clf.predict_proba(a)[:, 1])
    elif metric == 'accuracy':
        def score_func(clf, a, b):
            return clf.score(a, b)
    else:
        raise ValueError("Wrong metric!")
    if c is None:
        c = {i: np.array([i]) for i in range(len(X))}
    idxs, marginal_contribs = np.random.permutation(len(c.keys())), np.zeros(len(X))
    new_score = np.max(np.bincount(y)) * 1. / len(y) if np.mean(y // 1 == y / 1) == 1 else 0.
    start = 0
    if start:
        X_batch, y_batch = \
            np.concatenate([X[c[idx]] for idx in idxs[:start]]), np.concatenate([y[c[idx]] for idx in idxs[:start]])
    else:
        X_batch, y_batch = np.zeros((0,) + tuple(X.shape[1:])), np.zeros(0).astype(int)
    for n, idx in enumerate(idxs[start:]):
        try:
            clf = clone(clf)
        except:
            clf.fit(np.zeros((0,) + X.shape[1:]), y)
        old_score = new_score
        X_batch, y_batch = np.concatenate([X_batch, X[c[idx]]]), np.concatenate([y_batch, y[c[idx]]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X_batch, y_batch)
                temp_score = score_func(clf, X_test, y_test)
                if temp_score > -1 and temp_score < 1.:  # Removing measningless r2 scores
                    new_score = temp_score
            except:
                continue
        marginal_contribs[c[idx]] = (new_score - old_score) / len(c[idx])
        if np.abs(new_score - mean_score) / mean_score < tol:
            break
    return marginal_contribs, idxs


def error(mem):
    """calculates error of TMC-Shapley"""
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0) / np.reshape(np.arange(1, len(mem) + 1), (-1, 1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)


