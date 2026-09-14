"""K-means math in plain Python, independent of the graphical interface."""
import csv
import itertools
import math
import random


def validate_points(coords):
    if not coords:
        raise ValueError('Add at least one data point.')
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in coords):
        raise ValueError('Every point must contain two finite numbers (x, y).')


def load_points_csv(path):
    labels, coords = [], []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'x', 'y'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV needs x and y columns; label is optional.')
        for row in reader:
            if not any(row.values()):
                continue
            try:
                point = (float(row['x']), float(row['y']))
            except (ValueError, TypeError):
                raise ValueError(f'Invalid x or y at CSV line {reader.line_num}.') from None
            labels.append((row.get('label') or '').strip() or f'P{len(labels)+1}')
            coords.append(point)
    validate_points(coords)
    if len(set(labels)) != len(labels):
        raise ValueError('Point IDs must be unique.')
    return labels, coords


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def assign_clusters(coords, centroids):
    distances = [[euclidean(p, c) for c in centroids] for p in coords]
    return [d.index(min(d)) for d in distances], distances


def update_centroids(coords, assignments, k, old_centroids):
    result = [None] * k
    for i in range(k):
        members = [p for p, a in zip(coords, assignments) if a == i]
        if members:
            result[i] = tuple(sum(p[j] / len(members) for p in members) for j in (0, 1))
    # A bad user seed can leave a cluster empty. Move each empty centroid to the
    # data point farthest from the active centroids so it can join the next pass.
    for i, center in enumerate(result):
        if center is None:
            active = [c for c in result if c is not None]
            available = [p for p in dict.fromkeys(coords) if p not in active]
            result[i] = max(available, key=lambda p: min(euclidean(p, c) for c in active))
    return result


def kmeans_full(coords, init_centroids, max_iter=100):
    validate_points(coords)
    validate_points(init_centroids)
    if not 1 <= len(init_centroids) <= len(set(coords)):
        raise ValueError('k must be between 1 and the number of unique data points.')
    if max_iter < 1:
        raise ValueError('max_iter must be positive.')
    centroids, previous, history = list(init_centroids), None, []
    for iteration in range(1, max_iter + 1):
        assignments, distances = assign_clusters(coords, centroids)
        updated = update_centroids(coords, assignments, len(centroids), centroids)
        stable = assignments == previous and updated == centroids
        history.append(dict(iter=iteration, centroids_before=centroids[:],
                            dist_table=distances, assignments=assignments[:],
                            centroids_after=updated[:], converged=stable))
        centroids = updated
        if stable:
            break
        previous = assignments
    return centroids, assignments, history


def wcss(coords, centroids, assignments):
    return sum(euclidean(p, centroids[a]) ** 2 for p, a in zip(coords, assignments))


def initial_centroids(coords, k):
    """Deterministic farthest-point seeds; users may edit these before training."""
    unique = list(dict.fromkeys(coords))
    chosen = [unique[0]]
    while len(chosen) < min(k, len(unique)):
        chosen.append(max(unique, key=lambda p: min(euclidean(p, c) for c in chosen)))
    return chosen + [chosen[0]] * (k - len(chosen))


def elbow_method(coords, k_max=None):
    validate_points(coords)
    unique_count = len(set(coords))
    k_max = unique_count if k_max is None else min(k_max, unique_count)
    rng, results = random.Random(42), {}
    for k in range(1, k_max + 1):
        # Exhaustive for small problems; bounded deterministic restarts otherwise.
        if math.comb(len(coords), k) <= 100:
            seeds = ([coords[i] for i in combo]
                     for combo in itertools.combinations(range(len(coords)), k))
        else:
            seeds = [initial_centroids(coords, k)] + [rng.sample(coords, k) for _ in range(19)]
        best = math.inf
        for seed in seeds:
            c, a, _ = kmeans_full(coords, seed)
            best = min(best, wcss(coords, c, a))
        results[k] = best
    return results
