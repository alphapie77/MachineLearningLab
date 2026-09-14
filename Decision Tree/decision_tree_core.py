"""Categorical ID3 decision tree and evaluation utilities, built from scratch."""
import csv
import math
import random
from collections import Counter


def load_csv(path, target='Play'):
    with open(path, newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        if target not in columns or len(columns) < 2:
            raise ValueError(f'CSV needs feature columns and a {target} target column.')
        rows = []
        for row in reader:
            if not any(row.values()):
                continue
            if any(row.get(column, '').strip() == '' for column in columns):
                raise ValueError(f'Missing value at CSV line {reader.line_num}.')
            rows.append({column: row[column].strip() for column in columns})
    if len(rows) < 10:
        raise ValueError('Use at least 10 complete rows for train/test evaluation.')
    if len({row[target] for row in rows}) != 2:
        raise ValueError(f'{target} must contain exactly two classes.')
    return rows, [c for c in columns if c != target], target


def entropy(labels):
    counts, total = Counter(labels), len(labels)
    return -sum((n / total) * math.log2(n / total) for n in counts.values()) if total else 0.0


def information_gain(rows, feature, target):
    total = len(rows)
    groups = {}
    for row in rows:
        groups.setdefault(row[feature], []).append(row)
    remainder = sum(len(group) / total * entropy([r[target] for r in group])
                    for group in groups.values())
    return entropy([r[target] for r in rows]) - remainder


def majority(rows, target):
    counts = Counter(row[target] for row in rows)
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def build_tree(rows, features, target, max_depth=5, depth=0):
    counts = Counter(row[target] for row in rows)
    prediction = majority(rows, target)
    node = {'prediction': prediction, 'counts': dict(counts), 'depth': depth}
    if len(counts) == 1 or not features or depth >= max_depth:
        node['type'] = 'leaf'
        return node
    gains = {feature: information_gain(rows, feature, target) for feature in features}
    feature = max(features, key=lambda f: (gains[f], f))
    if gains[feature] <= 1e-12:
        node['type'] = 'leaf'
        return node
    node.update(type='node', feature=feature, gain=gains[feature], children={})
    for value in sorted({row[feature] for row in rows}):
        subset = [row for row in rows if row[feature] == value]
        node['children'][value] = build_tree(subset, [f for f in features if f != feature],
                                              target, max_depth, depth + 1)
    return node


def predict(tree, row):
    node, path = tree, []
    while node['type'] == 'node':
        feature, value = node['feature'], row.get(node['feature'])
        path.append((feature, value))
        if value not in node['children']:
            return node['prediction'], node, path
        node = node['children'][value]
    return node['prediction'], node, path


def positive_probability(node, positive):
    total = sum(node['counts'].values())
    return node['counts'].get(positive, 0) / total if total else 0.0


def stratified_split(rows, target, test_ratio=.2, seed=42):
    rng, train, test = random.Random(seed), [], []
    by_class = {}
    for row in rows:
        by_class.setdefault(row[target], []).append(row)
    for group in by_class.values():
        group = group[:]
        rng.shuffle(group)
        count = max(1, round(len(group) * test_ratio))
        test.extend(group[:count]); train.extend(group[count:])
    rng.shuffle(train); rng.shuffle(test)
    return train, test


def stratified_folds(rows, target, k=5, seed=43):
    if len(rows) < k * 2:
        raise ValueError(f'At least {k * 2} training rows are needed for {k}-fold CV.')
    rng, folds = random.Random(seed), [[] for _ in range(k)]
    by_class = {}
    for row in rows:
        by_class.setdefault(row[target], []).append(row)
    if min(map(len, by_class.values())) < k:
        raise ValueError(f'Each class needs at least {k} training rows for {k}-fold CV.')
    for group in by_class.values():
        group = group[:]; rng.shuffle(group)
        for i, row in enumerate(group):
            folds[i % k].append(row)
    return folds


def accuracy(rows, tree, target):
    return sum(predict(tree, row)[0] == row[target] for row in rows) / len(rows)


def select_depth(train, features, target, depths=(1, 2, 3, 4, 5), k=5):
    folds, scores = stratified_folds(train, target, k), {}
    for depth in depths:
        values = []
        for i in range(k):
            validation = folds[i]
            fitting = [row for j, fold in enumerate(folds) if j != i for row in fold]
            values.append(accuracy(validation, build_tree(fitting, features, target, depth), target))
        scores[depth] = sum(values) / len(values)
    # Prefer the simpler tree when scores tie.
    return max(depths, key=lambda d: (scores[d], -d)), scores


def evaluate(rows, tree, target, positive='Yes'):
    negative = next(label for label in {r[target] for r in rows} if label != positive)
    actual = [r[target] for r in rows]
    predicted_nodes = [predict(tree, r)[:2] for r in rows]
    predicted = [item[0] for item in predicted_nodes]
    scores = [positive_probability(item[1], positive) for item in predicted_nodes]
    tp = sum(a == positive and p == positive for a, p in zip(actual, predicted))
    tn = sum(a == negative and p == negative for a, p in zip(actual, predicted))
    fp = sum(a == negative and p == positive for a, p in zip(actual, predicted))
    fn = sum(a == positive and p == negative for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pairs = [(sp, sn) for sp, ap in zip(scores, actual) if ap == positive
                      for sn, an in zip(scores, actual) if an == negative]
    auc = sum(1 if sp > sn else .5 if sp == sn else 0 for sp, sn in pairs) / len(pairs)
    return {'accuracy': (tp + tn) / len(rows), 'precision': precision, 'recall': recall,
            'f1': f1, 'auc': auc, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'positive': positive, 'negative': negative}


def tree_stats(node):
    if node['type'] == 'leaf':
        return 1, node['depth']
    stats = [tree_stats(child) for child in node['children'].values()]
    return sum(s[0] for s in stats), max(s[1] for s in stats)


def tree_text(node, prefix=''):
    if node['type'] == 'leaf':
        return f"{prefix}Predict {node['prediction']}  counts={node['counts']}\n"
    text = f"{prefix}{node['feature']}  gain={node['gain']:.6f}  majority={node['prediction']}\n"
    for value, child in node['children'].items():
        text += f'{prefix}  if {node["feature"]} = {value}:\n' + tree_text(child, prefix + '    ')
    return text
