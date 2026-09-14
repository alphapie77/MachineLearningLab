# =====================================================================================
#  NON-LINEAR SVM CLASSIFIER  --  built from scratch (no sklearn / no libsvm)
#  Only numpy is used for array math and matplotlib for plotting; the SVM optimisation
#  itself (Simplified SMO, hyperparameter search, cross validation, metrics, ROC) is
#  implemented manually below so every equation can be shown and explained.
#
#  THEORY / EQUATIONS USED (also reproduced inside the GUI "Model information" panel)
#  -------------------------------------------------------------------------------
#  Because the two classes are NOT linearly separable (concentric-ring dataset),
#  we solve the DUAL soft-margin SVM problem, which lets us swap the dot product
#  x_i.x_j for a KERNEL function K(x_i,x_j) -> this is the "kernel trick" and is
#  what allows a straight hyperplane in a higher-dimensional feature space to
#  correspond to a CURVED decision boundary in the original 2-D space.
#
#  Dual objective (maximise over alpha):
#       max_alpha   sum_i alpha_i  -  (1/2) sum_i sum_j alpha_i alpha_j y_i y_j K(x_i,x_j)
#       subject to  0 <= alpha_i <= C   and   sum_i alpha_i y_i = 0     (KKT constraints)
#
#  RBF (Gaussian) kernel used here:
#       K(x, x') = exp( -gamma * ||x - x'||^2 )
#
#  Decision function (no explicit w in kernel space; expressed via support vectors):
#       f(x) = sum_i alpha_i * y_i * K(x_i, x)  +  b
#       predicted class = sign(f(x))
#
#  We find alpha (and b) with the SIMPLIFIED SMO ALGORITHM (Platt, 1998 /
#  simplified version from CS229 course notes): repeatedly pick a pair
#  (alpha_i, alpha_j), keep every other alpha fixed, and solve that 2-variable
#  sub-problem analytically (this is why SMO needs no external QP solver):
#
#     E_i = f(x_i) - y_i                                  (prediction error)
#     if KKT violated for i:
#        pick random j != i
#        L, H = box constraints for alpha_j given alpha_i + alpha_j = const
#        eta  = 2K(i,j) - K(i,i) - K(j,j)
#        alpha_j_new = clip( alpha_j - y_j(E_i-E_j)/eta , L, H )
#        alpha_i_new = alpha_i + y_i*y_j*(alpha_j_old - alpha_j_new)
#        update b from the KKT stationarity condition
#  Support vectors = training points with alpha_i > 0 (they alone define f(x)).
# =====================================================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import csv, os, math, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent))
from ui_theme import setup_theme, polish
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

RNG_SEED = 42
BASE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------
# 1) SYNTHETIC DATASET  --  500 points, 2 classes, concentric rings (NOT linearly separable)
# --------------------------------------------------------------------------------
def generate_circles_data(n=500, r_in=2.0, r_out=4.5, noise=0.8, seed=RNG_SEED):
    """
    Two concentric rings with a modest amount of noise - enough overlap to
    make the problem non-trivial (a straight line fails completely, ~50%),
    while the RBF-kernel SVM realistically reaches ~95% test accuracy.
    """
    rng = np.random.RandomState(seed)
    n1 = n // 2
    n2 = n - n1
    ang1 = rng.uniform(0, 2*np.pi, n1)
    ang2 = rng.uniform(0, 2*np.pi, n2)
    r1 = r_in + rng.randn(n1) * noise
    r2 = r_out + rng.randn(n2) * noise
    X1 = np.column_stack([r1*np.cos(ang1), r1*np.sin(ang1)])
    X2 = np.column_stack([r2*np.cos(ang2), r2*np.sin(ang2)])
    X = np.vstack([X1, X2])
    y = np.array([1]*n1 + [-1]*n2)
    idx = np.arange(n)
    rng.shuffle(idx)
    return X[idx], y[idx]


# --------------------------------------------------------------------------------
# 2) PRE-PROCESSING  --  manual z-score standardisation
# --------------------------------------------------------------------------------
def load_csv_data(path):
    """ Reads a CSV with columns x,y,label (label = +1 / -1). """
    X, y = [], []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'x', 'y', 'label'}.issubset(reader.fieldnames or []): raise ValueError('CSV needs x, y and label columns.')
        for row in reader:
            try: point, label = [float(row['x']), float(row['y'])], int(row['label'])
            except (ValueError, TypeError): raise ValueError(f'Invalid number at CSV line {reader.line_num}.') from None
            if not all(math.isfinite(v) for v in point) or label not in (-1, 1): raise ValueError(f'Line {reader.line_num}: finite x,y and label -1/+1 required.')
            X.append(point); y.append(label)
    if len(y) < 30 or min(y.count(-1), y.count(1)) < 6: raise ValueError('Use at least 30 rows and at least 6 per class.')
    return np.array(X, float), np.array(y, int)


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


def train_test_split(X, y, test_ratio=0.2, seed=1):
    rng = np.random.RandomState(seed)
    train_parts, test_parts = [], []
    for label in (-1, 1):
        idx = np.where(y == label)[0]; rng.shuffle(idx); n_test = max(1, round(len(idx)*test_ratio))
        test_parts.append(idx[:n_test]); train_parts.append(idx[n_test:])
    train_idx, test_idx = np.concatenate(train_parts), np.concatenate(test_parts); rng.shuffle(train_idx); rng.shuffle(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# --------------------------------------------------------------------------------
# 3) RBF KERNEL   K(x,x') = exp(-gamma * ||x-x'||^2)
#    ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b   (expanded manually, then vectorised)
# --------------------------------------------------------------------------------
def rbf_kernel_matrix(A, B, gamma):
    A2 = np.sum(A**2, axis=1).reshape(-1, 1)
    B2 = np.sum(B**2, axis=1).reshape(1, -1)
    sq_dist = np.maximum(A2 + B2 - 2*A.dot(B.T), 0.0)
    return np.exp(-gamma * sq_dist)


# --------------------------------------------------------------------------------
# 4) SIMPLIFIED SMO  --  see header comment for the equations behind every line
# --------------------------------------------------------------------------------
def smo_train(X, y, C=1.0, gamma=0.5, tol=1e-3, max_passes=5, seed=0):
    rng = np.random.RandomState(seed)
    m = len(y)
    K = rbf_kernel_matrix(X, X, gamma)
    alpha = np.zeros(m)
    b = 0.0
    passes = 0
    while passes < max_passes:
        num_changed = 0
        for i in range(m):
            f_i = np.sum(alpha * y * K[:, i]) + b
            E_i = f_i - y[i]
            if (y[i]*E_i < -tol and alpha[i] < C) or (y[i]*E_i > tol and alpha[i] > 0):
                j = i
                while j == i:
                    j = rng.randint(0, m)
                f_j = np.sum(alpha * y * K[:, j]) + b
                E_j = f_j - y[j]
                ai_old, aj_old = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L = max(0.0, aj_old - ai_old); H = min(C, C + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - C); H = min(C, ai_old + aj_old)
                if L == H:
                    continue
                eta = 2*K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                aj_new = aj_old - y[j]*(E_i - E_j)/eta
                aj_new = min(H, max(L, aj_new))
                if abs(aj_new - aj_old) < 1e-5:
                    continue
                ai_new = ai_old + y[i]*y[j]*(aj_old - aj_new)
                b1 = b - E_i - y[i]*(ai_new-ai_old)*K[i, i] - y[j]*(aj_new-aj_old)*K[i, j]
                b2 = b - E_j - y[i]*(ai_new-ai_old)*K[i, j] - y[j]*(aj_new-aj_old)*K[j, j]
                if 0 < ai_new < C:
                    b = b1
                elif 0 < aj_new < C:
                    b = b2
                else:
                    b = (b1 + b2)/2
                alpha[i], alpha[j] = ai_new, aj_new
                num_changed += 1
        passes = passes + 1 if num_changed == 0 else 0
    return alpha, b


def decision_function_kernel(X_train, y_train, alpha, b, X_query, gamma):
    """ f(x) = sum_i alpha_i * y_i * K(x_i, x) + b """
    K = rbf_kernel_matrix(X_train, X_query, gamma)
    return (alpha * y_train).dot(K) + b


def predict_kernel(X_train, y_train, alpha, b, X_query, gamma):
    return np.where(decision_function_kernel(X_train, y_train, alpha, b, X_query, gamma) >= 0, 1, -1)


# --------------------------------------------------------------------------------
# 5) K-FOLD CROSS VALIDATION  --  manual grid search over (C, gamma)
# --------------------------------------------------------------------------------
def k_fold_indices(n, k=3, seed=2):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    return np.array_split(idx, k)


def cross_validate(X, y, C, gamma, k=3, max_passes=3):
    folds = k_fold_indices(len(y), k)
    accs = []
    for i in range(k):
        val = folds[i]
        tr = np.hstack([folds[j] for j in range(k) if j != i])
        mu, sigma = standardize_fit(X[tr]); fitting = standardize_apply(X[tr], mu, sigma); validation = standardize_apply(X[val], mu, sigma)
        alpha, b = smo_train(fitting, y[tr], C=C, gamma=gamma, max_passes=max_passes)
        p = predict_kernel(fitting, y[tr], alpha, b, validation, gamma)
        accs.append(np.mean(p == y[val]))
    return float(np.mean(accs))


def grid_search(X, y, C_grid, gamma_grid, k=3, max_passes=3, progress=None):
    results = {}
    total = len(C_grid) * len(gamma_grid)
    for C in C_grid:
        for g in gamma_grid:
            results[(C, g)] = cross_validate(X, y, C, g, k=k, max_passes=max_passes)
            if progress:
                progress(len(results), total, C, g, results[(C, g)])
    best = max(results, key=results.get)
    return best, results


# --------------------------------------------------------------------------------
# 6) EVALUATION METRICS  --  identical manual definitions as the linear version
# --------------------------------------------------------------------------------
def confusion_matrix_manual(y_true, y_pred, pos=1, neg=-1):
    TP = int(np.sum((y_true == pos) & (y_pred == pos)))
    TN = int(np.sum((y_true == neg) & (y_pred == neg)))
    FP = int(np.sum((y_true == neg) & (y_pred == pos)))
    FN = int(np.sum((y_true == pos) & (y_pred == neg)))
    return TP, FP, FN, TN


def precision_recall_f1(TP, FP, FN, TN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    accuracy = (TP+TN)/(TP+FP+FN+TN) if (TP+FP+FN+TN) > 0 else 0.0
    return accuracy, precision, recall, f1


def roc_curve_manual(y_true, scores, pos=1, neg=-1):
    thresholds = np.sort(np.unique(scores))[::-1]
    P = np.sum(y_true == pos); N = np.sum(y_true == neg)
    tprs, fprs = [0.0], [0.0]
    for th in thresholds:
        pred = np.where(scores >= th, pos, neg)
        tp = np.sum((y_true == pos) & (pred == pos))
        fp = np.sum((y_true == neg) & (pred == pos))
        tprs.append(tp/P if P > 0 else 0.0)
        fprs.append(fp/N if N > 0 else 0.0)
    tprs.append(1.0); fprs.append(1.0)
    fprs, tprs = np.array(fprs), np.array(tprs)
    order = np.argsort(fprs)
    return fprs[order], tprs[order]


def auc_trapezoidal(fprs, tprs):
    return float(np.sum((fprs[1:]-fprs[:-1]) * (tprs[1:]+tprs[:-1]) / 2.0))


# =====================================================================================
#  GUI APPLICATION
# =====================================================================================
class KernelSVMApp:
    C_GRID = [0.5, 1, 5]
    GAMMA_GRID = [0.2, 0.5, 1.0]

    def __init__(self, root):
        self.root = root
        setup_theme(root)
        root.title('Non-Linear SVM Classifier  (RBF kernel, from scratch, SMO)')
        root.geometry('1300x900')
        root.configure(bg='#eaf3ff')
        try:
            root.state('zoomed')
        except tk.TclError:
            try:
                root.attributes('-zoomed', True)
            except tk.TclError:
                pass

        outer = tk.Canvas(root, bg='#eaf3ff', highlightthickness=0)
        sb = ttk.Scrollbar(root, command=outer.yview)
        outer.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        outer.pack(fill='both', expand=True)
        self.body = tk.Frame(outer, bg='#eaf3ff')
        self.body_win = outer.create_window((0, 0), window=self.body, anchor='nw')

        def _sync(e):
            outer.configure(scrollregion=outer.bbox('all'))
            outer.itemconfig(self.body_win, width=e.width)
        outer.bind('<Configure>', _sync)
        self.body.bind('<Configure>', lambda e: outer.configure(scrollregion=outer.bbox('all')))

        tk.Label(self.body, text='NON-LINEAR SVM CLASSIFIER (RBF KERNEL)', bg='#7a1f5c', fg='white',
                 font=('Segoe UI', 20, 'bold'), pady=16).pack(fill='x')
        tk.Label(self.body, text='Pure Python / NumPy implementation  \u2022  Simplified SMO (dual optimisation)  \u2022  grid-search CV over (C, gamma)',
                 bg='#eaf3ff', fg='#7a1f5c', font=('Segoe UI', 10, 'italic')).pack(fill='x', pady=(0, 6))

        bar = tk.Frame(self.body, bg='white')
        bar.pack(fill='x', padx=15, pady=10)
        tk.Button(bar, text='Load CSV', command=self.load_csv, bg='#f6e8f1').pack(side='left', padx=5, pady=8)
        tk.Button(bar, text='Generate New Data (n=500)', command=self.new_data, bg='#f6e8f1').pack(side='left', padx=5)
        self.train_btn = tk.Button(bar, text='Train / Evaluate (CV + SMO)', command=self.train, bg='#7a1f5c', fg='white',
                  font=('Segoe UI', 10, 'bold')); self.train_btn.pack(side='left', padx=5)
        tk.Button(bar, text='Save Report (TXT/CSV)', command=self.save_report, bg='#f6e8f1').pack(side='left', padx=5)
        tk.Button(bar, text='Bangla Lab Guide', command=self.show_guide, bg='#f6e8f1').pack(side='left', padx=5)
        self.progress_text = tk.StringVar(value='Ready')
        tk.Label(bar, textvariable=self.progress_text, bg='white', fg='#666').pack(side='left', padx=10)

        tk.Label(self.body, text='1) Non-linear decision boundary diagram (curved hyperplane in feature space)', bg='#f0d9e8',
                 fg='#7a1f5c', font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        plots = tk.Frame(self.body, bg='#eaf3ff')
        plots.pack(fill='both', padx=15, pady=5)

        self.fig_main = Figure(figsize=(7.6, 5.6), dpi=100)
        self.ax_main = self.fig_main.add_subplot(111)
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=plots)
        self.canvas_main.get_tk_widget().pack(side='left', fill='both', expand=True)

        self.fig_roc = Figure(figsize=(4.4, 5.6), dpi=100)
        self.ax_roc = self.fig_roc.add_subplot(111)
        self.canvas_roc = FigureCanvasTkAgg(self.fig_roc, master=plots)
        self.canvas_roc.get_tk_widget().pack(side='left', fill='both', expand=True, padx=(10, 0))

        tk.Label(self.body, text='2) Model information / hyperparameter search / manual calculation', bg='#f0d9e8',
                 fg='#7a1f5c', font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        self.info = tk.Text(self.body, height=18, bg='white', fg='black', font=('Consolas', 10))
        self.info.pack(fill='x', padx=15, pady=5)

        tk.Label(self.body, text='3) Evaluation: confusion matrix, accuracy, precision, recall, F1, ROC-AUC',
                 bg='#f0d9e8', fg='#7a1f5c', font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(
            fill='x', padx=15, pady=(10, 3))
        self.result = tk.Text(self.body, height=11, bg='white', fg='black', font=('Consolas', 10))
        self.result.pack(fill='x', padx=15, pady=5)

        tk.Label(self.body, text='4) Classify a new data point (x, y)', bg='#f0d9e8', fg='#7a1f5c',
                 font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        p = tk.Frame(self.body, bg='white')
        p.pack(fill='x', padx=15, pady=10)
        tk.Label(p, text='x =', bg='white', font=('Segoe UI', 11)).grid(row=0, column=0, padx=8, pady=14)
        self.ex = tk.Entry(p, width=10, font=('Segoe UI', 11))
        self.ex.grid(row=0, column=1)
        tk.Label(p, text='y =', bg='white', font=('Segoe UI', 11)).grid(row=0, column=2, padx=8)
        self.ey = tk.Entry(p, width=10, font=('Segoe UI', 11))
        self.ey.grid(row=0, column=3)
        tk.Button(p, text='Classify Point', command=self.classify_point, bg='#7a1f5c', fg='white',
                  font=('Segoe UI', 10, 'bold')).grid(row=0, column=4, padx=15)
        self.out = tk.Label(p, text='Prediction: \u2014', bg='white', fg='#087f5b', font=('Segoe UI', 13, 'bold'))
        self.out.grid(row=0, column=5, padx=15)

        self.X = self.y = None
        self.Xtr = self.ytr = self.Xte = self.yte = None
        self.mu = self.sigma = None
        self.alpha = self.b = None
        self.best_C = self.best_gamma = None
        self.last_point = None
        self.report = ''
        self.last_classification = ''
        root.after_idle(lambda: polish(root))

        default_csv = BASE / 'svm_circle_data.csv'
        if os.path.exists(default_csv):
            self.load_csv(default_csv)
        else:
            self.new_data()

    # ------------------------------------------------------------------
    def load_csv(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])
            if not path:
                return
        try:
            self.X, self.y = load_csv_data(path)
        except Exception as e:
            messagebox.showerror('Error loading CSV', str(e))
            return
        self._prepare_after_load()

    def new_data(self):
        seed = np.random.randint(0, 100000)
        self.X, self.y = generate_circles_data(500, seed=seed)
        self._prepare_after_load()

    def _prepare_after_load(self):
        self.Xtr_raw, self.ytr, self.Xte_raw, self.yte = train_test_split(self.X, self.y)
        self.mu, self.sigma = standardize_fit(self.Xtr_raw)
        self.Xtr = standardize_apply(self.Xtr_raw, self.mu, self.sigma); self.Xte = standardize_apply(self.Xte_raw, self.mu, self.sigma)
        self.alpha = self.b = None
        self.last_point = None
        self.report = ''
        self.last_classification = ''
        self.info.delete('1.0', 'end')
        self.info.insert('end', f'Generated {len(self.y)} samples arranged as two concentric rings (2 classes, '
                                 f'{np.sum(self.y==1)} vs {np.sum(self.y==-1)}) \u2014 NOT linearly separable.\n'
                                 f'Train set: {len(self.ytr)}   Test set: {len(self.yte)}\n'
                                 f'Standardised with z-score: mu={np.round(self.mu,3)}, sigma={np.round(self.sigma,3)}\n\n'
                                 f'Click "Train / Evaluate" to grid-search C in {self.C_GRID} and gamma in {self.GAMMA_GRID}\n'
                                 f'(3-fold CV), then train the final RBF-kernel SVM with Simplified SMO.\n')
        self.result.delete('1.0', 'end')
        self.out.config(text='Prediction: \u2014')
        self.plot_data_only()

    # ------------------------------------------------------------------
    def plot_data_only(self):
        self.ax_main.clear()
        self.ax_main.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], c='#2e6bd6', label='Class +1 (inner ring)', edgecolor='white', s=45)
        self.ax_main.scatter(self.X[self.y == -1, 0], self.X[self.y == -1, 1], c='#e0473c', label='Class -1 (outer ring)', edgecolor='white', s=45)
        self.ax_main.set_xlabel('x'); self.ax_main.set_ylabel('y')
        self.ax_main.set_title('Raw data (not yet trained)')
        self.ax_main.legend(loc='upper left')
        self.ax_main.grid(alpha=0.25)
        self.ax_main.set_aspect('equal', adjustable='box')
        self.fig_main.tight_layout()
        self.canvas_main.draw()
        self.ax_roc.clear()
        self.ax_roc.set_title('ROC curve (after training)')
        self.ax_roc.set_xlabel('False Positive Rate'); self.ax_roc.set_ylabel('True Positive Rate')
        self.fig_roc.tight_layout()
        self.canvas_roc.draw()

    # ------------------------------------------------------------------
    def train(self):
        if self.X is None:
            return
        self.train_btn.config(state='disabled')
        self.progress_text.set('Starting cross-validation…')
        self.info.insert('end', '\n>>> Training in progress, please wait ...\n')
        self.info.see('end')
        self.root.update()

        def show_progress(done, total, C, gamma, score):
            self.progress_text.set(f'CV {done}/{total} · C={C:g}, gamma={gamma:g} · {score*100:.1f}%')
            self.root.update_idletasks(); self.root.update()
        (best_C, best_g), cv_results = grid_search(
            self.Xtr_raw, self.ytr, self.C_GRID, self.GAMMA_GRID,
            k=3, max_passes=3, progress=show_progress)
        self.progress_text.set('Training final model…'); self.root.update_idletasks(); self.root.update()
        self.best_C, self.best_gamma = best_C, best_g

        self.alpha, self.b = smo_train(self.Xtr, self.ytr, C=best_C, gamma=best_g, max_passes=6)
        scores = decision_function_kernel(self.Xtr, self.ytr, self.alpha, self.b, self.Xte, best_g)
        pred = np.where(scores >= 0, 1, -1)
        TP, FP, FN, TN = confusion_matrix_manual(self.yte, pred)
        acc, prec, rec, f1 = precision_recall_f1(TP, FP, FN, TN)
        fprs, tprs = roc_curve_manual(self.yte, scores)
        auc = auc_trapezoidal(fprs, tprs)

        sv_mask = self.alpha > 1e-5
        n_sv = int(np.sum(sv_mask))

        cv_lines = '\n'.join(f'   C={C:<5g} gamma={g:<5g} -> mean CV accuracy = {a:.4f}' for (C, g), a in cv_results.items())

        # manual worked example: kernel value between first two support vectors + f(x) for one test point
        sv_idx = np.where(sv_mask)[0]
        ex_i = 0
        ex_x = self.Xte[ex_i]
        ex_true = self.yte[ex_i]
        top_k = sv_idx[:3] if len(sv_idx) >= 3 else sv_idx
        terms = []
        f_manual = 0.0
        for k in top_k:
            xi = self.Xtr[k]
            sqd = np.sum((xi - ex_x)**2)
            kval = np.exp(-best_g * sqd)
            term = self.alpha[k] * self.ytr[k] * kval
            f_manual += term
            terms.append(f"alpha={self.alpha[k]:.4f}, y={self.ytr[k]:+d}, ||x_i-x||^2={sqd:.4f}, "
                          f"K=exp(-{best_g:.2f}*{sqd:.4f})={kval:.4f}  -> term={term:.4f}")
        f_full = float(decision_function_kernel(self.Xtr, self.ytr, self.alpha, self.b, ex_x.reshape(1, -1), best_g)[0])
        pred_ex = 1 if f_full >= 0 else -1

        self.info.delete('1.0', 'end')
        self.info.insert('end',
            "DUAL OBJECTIVE (kernel soft-margin SVM):\n"
            "  max_alpha sum_i alpha_i - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j K(x_i,x_j)\n"
            "  subject to: 0 <= alpha_i <= C ,  sum_i alpha_i y_i = 0\n\n"
            "RBF KERNEL:   K(x, x') = exp( -gamma * ||x - x'||^2 )\n\n"
            "SIMPLIFIED SMO: repeatedly optimise a pair (alpha_i, alpha_j) analytically\n"
            "  E_i = f(x_i)-y_i ;  eta = 2K(i,j)-K(i,i)-K(j,j) ;  alpha_j <- clip(alpha_j - y_j(E_i-E_j)/eta, L, H)\n\n"
            f"HYPERPARAMETER SEARCH (3-fold cross validation over C x gamma):\n{cv_lines}\n"
            f"  --> Selected C* = {best_C}, gamma* = {best_g}  (highest mean CV accuracy)\n\n"
            f"FINAL TRAINED MODEL (on {len(self.ytr)} training points, standardised features):\n"
            f"  b = {self.b:.4f}\n"
            f"  Support vectors (alpha_i > 0): {n_sv} out of {len(self.ytr)} training points\n\n"
            "MANUAL CALCULATION EXAMPLE (decision function for the first test point, using its first support vectors):\n"
            f"  query x = ({ex_x[0]:.4f}, {ex_x[1]:.4f}),  true class y = {ex_true:+d}\n"
            + '\n'.join('  ' + t for t in terms) + '\n'
            f"  f(x) \u2248 sum of these terms + ... + b({self.b:.4f})  =  {f_full:.4f}   (using ALL {n_sv} support vectors)\n"
            f"  sign(f(x)) = {pred_ex:+d}  ->  predicted class = {pred_ex:+d}  ({'CORRECT' if pred_ex==ex_true else 'WRONG'})\n"
        )

        self.result.delete('1.0', 'end')
        self.result.insert('end',
            f"Accuracy : {acc:.4f}\nPrecision: {prec:.4f}\nRecall   : {rec:.4f}\nF1-score : {f1:.4f}\nROC-AUC  : {auc:.4f}\n\n"
            f"Confusion Matrix (test set, n={len(self.yte)})\n"
            f"                     Predicted +1     Predicted -1\n"
            f"   Actual  +1       {TP:8d}         {FN:8d}\n"
            f"   Actual  -1       {FP:8d}         {TN:8d}\n\n"
            f"  precision = TP/(TP+FP) = {TP}/{TP+FP} = {prec:.4f}\n"
            f"  recall    = TP/(TP+FN) = {TP}/{TP+FN} = {rec:.4f}\n"
            f"  f1        = 2*P*R/(P+R)          = {f1:.4f}\n"
        )

        self.plot_trained(sv_mask, fprs, tprs, auc)
        self.report = ('=== NON-LINEAR (RBF KERNEL) SVM CLASSIFIER REPORT ===\n\n'
                        + self.info.get('1.0', 'end') + '\n' + self.result.get('1.0', 'end'))
        self.progress_text.set(f'Complete · C={best_C:g}, gamma={best_g:g}, AUC={auc:.3f}')
        self.train_btn.config(state='normal')

    # ------------------------------------------------------------------
    def plot_trained(self, sv_mask, fprs, tprs, auc):
        Xs_all = np.vstack([self.Xtr, self.Xte])
        x_min, x_max = Xs_all[:, 0].min() - 1, Xs_all[:, 0].max() + 1
        y_min, y_max = Xs_all[:, 1].min() - 1, Xs_all[:, 1].max() + 1

        self.ax_main.clear()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        Z = np.concatenate([
            decision_function_kernel(self.Xtr, self.ytr, self.alpha, self.b, part, self.best_gamma)
            for part in np.array_split(grid_pts, 8)
        ]).reshape(xx.shape)

        self.ax_main.contourf(xx, yy, Z, levels=[Z.min()-1, 0, Z.max()+1], colors=['#ffe1de', '#dce8ff'], alpha=0.55)
        self.ax_main.contour(xx, yy, Z, levels=[-1, 0, 1], colors='black', linestyles=['--', '-', '--'], linewidths=[1.2, 2.2, 1.2])

        self.ax_main.scatter(self.Xtr[self.ytr == 1, 0], self.Xtr[self.ytr == 1, 1], c='#2e6bd6', label='Class +1 (train)', edgecolor='white', s=40)
        self.ax_main.scatter(self.Xtr[self.ytr == -1, 0], self.Xtr[self.ytr == -1, 1], c='#e0473c', label='Class -1 (train)', edgecolor='white', s=40)
        self.ax_main.scatter(self.Xte[:, 0], self.Xte[:, 1], facecolors='none', edgecolors='#333333', s=60, linewidths=1.2, label='Test points')
        self.ax_main.scatter(self.Xtr[sv_mask, 0], self.Xtr[sv_mask, 1], facecolors='none', edgecolors='#111827',
                              s=90, linewidths=1.1, alpha=.8, label='Support vectors')

        if self.last_point is not None:
            self.ax_main.scatter([self.last_point[0]], [self.last_point[1]], marker='*', s=320,
                                  c='#ffb703', edgecolor='black', linewidths=1.2, label='New point', zorder=5)

        self.ax_main.set_xlim(x_min, x_max); self.ax_main.set_ylim(y_min, y_max)
        self.ax_main.set_xlabel('x (standardised)'); self.ax_main.set_ylabel('y (standardised)')
        self.ax_main.set_title(f'RBF-kernel SVM  (C={self.best_C}, gamma={self.best_gamma})  \u2014  non-linear decision boundary')
        self.ax_main.legend(loc='upper left', fontsize=8)
        self.ax_main.grid(alpha=0.2)
        self.ax_main.set_aspect('equal', adjustable='box')
        self.fig_main.tight_layout()
        self.canvas_main.draw()

        self.ax_roc.clear()
        self.ax_roc.plot(fprs, tprs, color='#7a1f5c', linewidth=2.2, label=f'ROC (AUC = {auc:.3f})')
        self.ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random guess')
        self.ax_roc.fill_between(fprs, tprs, alpha=0.15, color='#7a1f5c')
        self.ax_roc.set_xlabel('False Positive Rate'); self.ax_roc.set_ylabel('True Positive Rate')
        self.ax_roc.set_title('ROC Curve')
        self.ax_roc.legend(loc='lower right', fontsize=8)
        self.ax_roc.grid(alpha=0.25)
        self.fig_roc.tight_layout()
        self.canvas_roc.draw()

    # ------------------------------------------------------------------
    def classify_point(self):
        if self.alpha is None:
            messagebox.showinfo('Train first', 'Please click "Train / Evaluate" before classifying a new point.')
            return
        try:
            x_raw = float(self.ex.get())
            y_raw = float(self.ey.get())
            if not math.isfinite(x_raw) or not math.isfinite(y_raw):
                raise ValueError
        except ValueError:
            messagebox.showerror('Invalid input', 'Please enter finite numeric values for x and y.')
            return
        point_raw = np.array([x_raw, y_raw])
        point_std = standardize_apply(point_raw, self.mu, self.sigma)
        f_val = float(decision_function_kernel(self.Xtr, self.ytr, self.alpha, self.b,
                                                point_std.reshape(1, -1), self.best_gamma)[0])
        cls = 1 if f_val >= 0 else -1
        self.last_point = point_std
        self.out.config(text=f'Prediction: Class {cls:+d}   (f(x)={f_val:.4f})')
        self.last_classification = (f"\nNEW POINT CLASSIFICATION\n"
                                     f"  input (x, y) = ({x_raw}, {y_raw})  ->  standardised = ({point_std[0]:.4f}, {point_std[1]:.4f})\n"
                                     f"  f(x) = sum_i alpha_i y_i K(x_i,x) + b = {f_val:.4f}  ->  predicted class = {cls:+d}\n")
        self.report = self.report + self.last_classification if self.report else self.last_classification

        sv_mask = self.alpha > 1e-5
        scores = decision_function_kernel(self.Xtr, self.ytr, self.alpha, self.b, self.Xte, self.best_gamma)
        fprs, tprs = roc_curve_manual(self.yte, scores)
        auc = auc_trapezoidal(fprs, tprs)
        self.plot_trained(sv_mask, fprs, tprs, auc)

    # ------------------------------------------------------------------
    def save_report(self):
        if not self.report:
            messagebox.showinfo('Nothing to save', 'Please train the model first (click "Train / Evaluate").')
            return
        path = filedialog.asksaveasfilename(defaultextension='.txt',
                                             filetypes=[('Text', '*.txt'), ('CSV', '*.csv')])
        if not path:
            return
        if path.lower().endswith('.csv'):
            with open(path, 'w', newline='', encoding='utf8') as f:
                w = csv.writer(f)
                w.writerow(['section', 'content'])
                for line in self.report.split('\n'):
                    w.writerow(['report', line])
        else:
            with open(path, 'w', encoding='utf8') as f:
                f.write(self.report)
        messagebox.showinfo('Saved', 'Report saved successfully.')

    def show_guide(self):
        window = tk.Toplevel(self.root); window.title('RBF SVM — Bangla Lab Guide'); window.geometry('900x700')
        text = tk.Text(window, wrap='word', font=('Segoe UI', 11), padx=15, pady=15)
        scroll = ttk.Scrollbar(window, command=text.yview); text.config(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y'); text.pack(fill='both', expand=True)
        text.insert('1.0', (BASE/'LAB_GUIDE_BN.txt').read_text(encoding='utf-8')); text.config(state='disabled')


if __name__ == '__main__':
    App = tk.Tk()
    KernelSVMApp(App)
    App.mainloop()
