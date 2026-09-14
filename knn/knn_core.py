"""Testable, dependency-free KNN pipeline for the shirt-size lab."""
import csv
import math
import random
from collections import Counter, defaultdict


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"height_cm", "weight_kg", "shirt_size"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain height_cm, weight_kg and shirt_size columns.")
        rows = []
        for line, raw in enumerate(reader, 2):
            try:
                h, w = float(raw["height_cm"]), float(raw["weight_kg"])
            except (TypeError, ValueError):
                raise ValueError(f"Row {line}: height and weight must be numeric.")
            label = (raw["shirt_size"] or "").strip()
            if not math.isfinite(h) or not math.isfinite(w):
                raise ValueError(f"Row {line}: NaN/Infinity is not allowed.")
            if not label:
                raise ValueError(f"Row {line}: shirt_size is missing.")
            rows.append((h, w, label))
    if len(rows) < 20 or len(set(r[2] for r in rows)) < 2:
        raise ValueError("At least 20 rows and two classes are required.")
    return rows


def stratified_split(rows, test_ratio=.2, seed=21):
    groups = defaultdict(list)
    for row in rows:
        groups[row[2]].append(row)
    train, test = [], []
    for label in sorted(groups):
        group = groups[label][:]
        random.Random(f"{seed}:{label}").shuffle(group)
        n_test = max(1, round(len(group) * test_ratio))
        test.extend(group[:n_test]); train.extend(group[n_test:])
    random.Random(seed).shuffle(train); random.Random(seed + 1).shuffle(test)
    return train, test


def stratified_folds(rows, folds=5, seed=7):
    buckets = [[] for _ in range(folds)]
    groups = defaultdict(list)
    for row in rows:
        groups[row[2]].append(row)
    for label in sorted(groups):
        group = groups[label][:]
        random.Random(f"{seed}:{label}").shuffle(group)
        for i, row in enumerate(group):
            buckets[i % folds].append(row)
    return buckets


def fit_scaler(rows):
    means = tuple(sum(r[j] for r in rows) / len(rows) for j in (0, 1))
    stds = tuple(math.sqrt(sum((r[j] - means[j]) ** 2 for r in rows) / len(rows)) or 1.0 for j in (0, 1))
    return means, stds


def transform(rows, means, stds):
    return [((r[0] - means[0]) / stds[0], (r[1] - means[1]) / stds[1], r[2]) for r in rows]


def transform_point(point, means, stds):
    return ((point[0] - means[0]) / stds[0], (point[1] - means[1]) / stds[1])


def classify(train, point, k, details=False):
    if not train or k < 1 or k > len(train):
        raise ValueError("k must be between 1 and the number of training rows.")
    ranked = sorted((math.hypot(r[0] - point[0], r[1] - point[1]), r[2], r[0], r[1]) for r in train)
    near = ranked[:k]
    votes = Counter(r[1] for r in near)
    # Deterministic tie: nearest tied class, then alphabetical label.
    top = max(votes.values()); tied = {lab for lab, count in votes.items() if count == top}
    prediction = next(lab for _, lab, _, _ in near if lab in tied)
    return (prediction, near, votes) if details else prediction


def confusion(actual, predicted, labels):
    matrix = {a: {p: 0 for p in labels} for a in labels}
    for a, p in zip(actual, predicted): matrix[a][p] += 1
    return matrix


def metrics(matrix, labels):
    total = sum(sum(row.values()) for row in matrix.values())
    per_class = {}; correct = 0
    for label in labels:
        tp = matrix[label][label]; correct += tp
        fp = sum(matrix[a][label] for a in labels if a != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=sum(matrix[label].values()))
    return dict(accuracy=correct / total if total else 0.0,
                macro_precision=sum(x["precision"] for x in per_class.values()) / len(labels),
                macro_recall=sum(x["recall"] for x in per_class.values()) / len(labels),
                macro_f1=sum(x["f1"] for x in per_class.values()) / len(labels), per_class=per_class)


def select_k(train, candidates=(3, 5, 7, 9), folds=5):
    fold_rows = stratified_folds(train, folds)
    labels = sorted(set(r[2] for r in train)); scores = {}
    for k in candidates:
        fold_scores = []
        for i in range(folds):
            validation = fold_rows[i]
            training = [r for j, fold in enumerate(fold_rows) if j != i for r in fold]
            means, stds = fit_scaler(training)
            tr = transform(training, means, stds); va = transform(validation, means, stds)
            pred = [classify(tr, row, k) for row in va]
            fold_scores.append(metrics(confusion([r[2] for r in va], pred, labels), labels)["macro_f1"])
        scores[k] = sum(fold_scores) / len(fold_scores)
    best = max(candidates, key=lambda k: (scores[k], -k))
    return best, scores


def train_evaluate(rows, candidates=(3, 5, 7, 9)):
    train_raw, test_raw = stratified_split(rows)
    best_k, cv = select_k(train_raw, candidates)
    means, stds = fit_scaler(train_raw)
    train = transform(train_raw, means, stds); test = transform(test_raw, means, stds)
    labels = sorted(set(r[2] for r in rows))
    predicted = [classify(train, row, best_k) for row in test]
    matrix = confusion([r[2] for r in test], predicted, labels)
    result = metrics(matrix, labels)
    result.update(best_k=best_k, cv=cv, labels=labels, matrix=matrix,
                  means=means, stds=stds, train=train, test=test,
                  train_raw=train_raw, test_raw=test_raw)
    return result
