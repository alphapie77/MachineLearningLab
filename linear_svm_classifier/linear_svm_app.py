# =====================================================================================
#  LINEAR SVM CLASSIFIER  --  built from scratch (no sklearn / no libsvm)
#  Only numpy is used for array math and matplotlib for plotting; the SVM algorithm
#  itself (training, hyperparameter search, cross validation, metrics, ROC curve) is
#  implemented manually below so every equation can be shown and explained.
#
#  THEORY / EQUATIONS USED (also reproduced inside the GUI "Model information" panel)
#  -------------------------------------------------------------------------------
#  Primal soft-margin SVM objective (what we are minimising):
#       J(w,b) = (1/2)||w||^2  +  C * sum_i max(0, 1 - y_i (w.x_i + b))
#            |----penalty----|    |---------hinge loss over all points---------|
#
#  We solve this directly (no dual, no kernel trick needed for the LINEAR case)
#  using the PEGASOS algorithm (Primal Estimated sub-GrAdient SOlver for SVM,
#  Shalev-Shwartz et al. 2007), a stochastic sub-gradient descent method:
#
#       let lambda = 1 / (C * n)
#       for t = 1, 2, 3, ... :
#           pick a random training point (x_i, y_i)
#           eta_t = 1 / (lambda * t)                      <- learning rate schedule
#           if y_i (w.x_i + b) < 1:                        <- point violates margin
#                w <- (1 - eta_t*lambda) * w + eta_t * y_i * x_i
#                b <- b + eta_t * y_i
#           else:                                          <- point is safe / outside margin
#                w <- (1 - eta_t*lambda) * w
#
#  Decision function :   f(x) = w . x + b
#  Predicted class   :   sign(f(x))   ->  +1 or -1
#  Separating hyperplane :  w.x + b = 0
#  Margin boundaries      :  w.x + b = +1   and   w.x + b = -1
#  Geometric margin width  :  2 / ||w||   (this is what we are trying to maximise)
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
# 1) SYNTHETIC DATASET  --  500 points, 2 classes, roughly linearly separable
# --------------------------------------------------------------------------------
def generate_linear_data(n=500, seed=RNG_SEED):
    """
    Two well-separated Gaussian blobs (large mean separation, small
    covariance / spread) so the two classes stay cleanly apart with no
    overlapping points - this keeps the hyperplane + margin diagram easy
    to read at a glance.
    """
    rng = np.random.RandomState(seed)
    n1 = n // 2
    n2 = n - n1
    mean1, cov1 = np.array([3.0, 3.0]), [[1.0, 0.2], [0.2, 1.0]]
    mean2, cov2 = np.array([-3.0, -3.0]), [[1.0, -0.2], [-0.2, 1.0]]
    X1 = rng.multivariate_normal(mean1, cov1, n1)
    X2 = rng.multivariate_normal(mean2, cov2, n2)
    X = np.vstack([X1, X2])
    y = np.array([1] * n1 + [-1] * n2)
    idx = np.arange(n)
    rng.shuffle(idx)
    return X[idx], y[idx]


# --------------------------------------------------------------------------------
# 2) PRE-PROCESSING  --  manual z-score standardisation:  z = (x - mean) / std
# --------------------------------------------------------------------------------
def load_csv_data(path):
    """ Reads a CSV with columns x,y,label (label = +1 / -1). """
    X, y = [], []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'x', 'y', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV needs x, y and label columns.')
        for row in reader:
            try: point, label = [float(row['x']), float(row['y'])], int(row['label'])
            except (ValueError, TypeError): raise ValueError(f'Invalid number at CSV line {reader.line_num}.') from None
            if not all(math.isfinite(v) for v in point) or label not in (-1, 1):
                raise ValueError(f'Line {reader.line_num}: x,y must be finite and label must be -1 or +1.')
            X.append(point); y.append(label)
    if len(y) < 20 or min(y.count(-1), y.count(1)) < 5:
        raise ValueError('Use at least 20 rows and at least 5 samples in each class.')
    return np.array(X, dtype=float), np.array(y, dtype=int)


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


# --------------------------------------------------------------------------------
# 3) TRAIN / TEST SPLIT  --  manual random split (80% train / 20% test)
# --------------------------------------------------------------------------------
def train_test_split(X, y, test_ratio=0.2, seed=1):
    rng = np.random.RandomState(seed)
    train_parts, test_parts = [], []
    for label in (-1, 1):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_parts.append(idx[:n_test]); train_parts.append(idx[n_test:])
    train_idx, test_idx = np.concatenate(train_parts), np.concatenate(test_parts)
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# --------------------------------------------------------------------------------
# 4) PEGASOS TRAINING  --  see equations in the header comment above
# --------------------------------------------------------------------------------
def pegasos_train(X, y, C=1.0, epochs=100, seed=0):
    n, d = X.shape
    lam = 1.0 / (C * n)                       # lambda = 1/(C*n)
    w = np.zeros(d)
    b = 0.0
    rng = np.random.RandomState(seed)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in order:
            t += 1
            eta = 1.0 / (lam * t)
            margin = y[i] * (np.dot(w, X[i]) + b)     # y_i (w.x_i + b)
            if margin < 1:                             # inside margin / misclassified
                w = (1 - eta * lam) * w + eta * y[i] * X[i]
                b = b + eta * y[i]
            else:                                      # safely outside margin
                w = (1 - eta * lam) * w
    return w, b


def decision_function(X, w, b):
    """ f(x) = w.x + b """
    return X.dot(w) + b


def predict(X, w, b):
    return np.where(decision_function(X, w, b) >= 0, 1, -1)


# --------------------------------------------------------------------------------
# 5) K-FOLD CROSS VALIDATION  --  manual implementation, used to pick best C
# --------------------------------------------------------------------------------
def k_fold_indices(n, k=5, seed=2):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    return np.array_split(idx, k)


def cross_validate_C(X, y, C, k=5, epochs=30):
    folds = k_fold_indices(len(y), k)
    accs = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        mu, sigma = standardize_fit(X[train_idx])
        fitting = standardize_apply(X[train_idx], mu, sigma)
        validation = standardize_apply(X[val_idx], mu, sigma)
        w, b = pegasos_train(fitting, y[train_idx], C=C, epochs=epochs)
        p = predict(validation, w, b)
        accs.append(np.mean(p == y[val_idx]))
    return float(np.mean(accs))


def grid_search_C(X, y, C_grid, k=5, epochs=30):
    results = {C: cross_validate_C(X, y, C, k=k, epochs=epochs) for C in C_grid}
    best_C = max(results, key=results.get)
    return best_C, results


# --------------------------------------------------------------------------------
# 6) EVALUATION METRICS  --  all computed manually (no sklearn.metrics)
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
    return accuracy, precision, recall, f1


def roc_curve_manual(y_true, scores, pos=1, neg=-1):
    """
    For every possible threshold t (each unique score value), classify as
    positive if score >= t, then compute:
        TPR = TP / P      (True Positive Rate  = Recall = Sensitivity)
        FPR = FP / N       (False Positive Rate = 1 - Specificity)
    Sweeping t from +inf down to -inf traces the ROC curve.
    """
    thresholds = np.sort(np.unique(scores))[::-1]
    P = np.sum(y_true == pos)
    N = np.sum(y_true == neg)
    tprs, fprs = [0.0], [0.0]
    for th in thresholds:
        pred = np.where(scores >= th, pos, neg)
        tp = np.sum((y_true == pos) & (pred == pos))
        fp = np.sum((y_true == neg) & (pred == pos))
        tprs.append(tp / P if P > 0 else 0.0)
        fprs.append(fp / N if N > 0 else 0.0)
    tprs.append(1.0)
    fprs.append(1.0)
    fprs, tprs = np.array(fprs), np.array(tprs)
    order = np.argsort(fprs)
    return fprs[order], tprs[order]


def auc_trapezoidal(fprs, tprs):
    """ Manual trapezoidal-rule numerical integration:  AUC = sum of trapezoid areas """
    return float(np.sum((fprs[1:] - fprs[:-1]) * (tprs[1:] + tprs[:-1]) / 2.0))


# =====================================================================================
#  GUI APPLICATION
# =====================================================================================
class LinearSVMApp:
    C_GRID = [0.01, 0.1, 1, 10, 100]

    def __init__(self, root):
        self.root = root
        setup_theme(root)
        root.title('Linear SVM Classifier  (from scratch, Pegasos)')
        root.geometry('1300x900')
        root.configure(bg='#eaf3ff')
        try:
            root.state('zoomed')
        except tk.TclError:
            try:
                root.attributes('-zoomed', True)
            except tk.TclError:
                pass

        # scrollable body -------------------------------------------------------
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

        tk.Label(self.body, text='LINEAR SVM CLASSIFIER', bg='#174a7e', fg='white',
                 font=('Segoe UI', 20, 'bold'), pady=16).pack(fill='x')
        tk.Label(self.body, text='Pure Python / NumPy implementation  \u2022  Pegasos sub-gradient descent  \u2022  k-fold CV for hyperparameter C',
                 bg='#eaf3ff', fg='#174a7e', font=('Segoe UI', 10, 'italic')).pack(fill='x', pady=(0, 6))

        bar = tk.Frame(self.body, bg='white')
        bar.pack(fill='x', padx=15, pady=10)
        tk.Button(bar, text='Load CSV', command=self.load_csv, bg='#eef4ff').pack(side='left', padx=5, pady=8)
        tk.Button(bar, text='Generate New Data (n=500)', command=self.new_data, bg='#eef4ff').pack(side='left', padx=5)
        tk.Button(bar, text='Train / Evaluate (CV + Pegasos)', command=self.train, bg='#174a7e', fg='white',
                  font=('Segoe UI', 10, 'bold')).pack(side='left', padx=5)
        tk.Button(bar, text='Save Report (TXT/CSV)', command=self.save_report, bg='#eef4ff').pack(side='left', padx=5)
        tk.Button(bar, text='Bangla Lab Guide', command=self.show_guide, bg='#eef4ff').pack(side='left', padx=5)

        # --- diagram + ROC side by side -----------------------------------------
        tk.Label(self.body, text='1) Maximum-margin hyperplane diagram', bg='#cfe3ff', fg='#174a7e',
                 font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
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

        # --- info box ------------------------------------------------------------
        tk.Label(self.body, text='2) Model information / hyperparameter search / manual calculation', bg='#cfe3ff',
                 fg='#174a7e', font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        self.info = tk.Text(self.body, height=16, bg='white', fg='black', font=('Consolas', 10))
        self.info.pack(fill='x', padx=15, pady=5)

        # --- metrics box ----------------------------------------------------------
        tk.Label(self.body, text='3) Evaluation: confusion matrix, accuracy, precision, recall, F1, ROC-AUC',
                 bg='#cfe3ff', fg='#174a7e', font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(
            fill='x', padx=15, pady=(10, 3))
        self.result = tk.Text(self.body, height=11, bg='white', fg='black', font=('Consolas', 10))
        self.result.pack(fill='x', padx=15, pady=5)

        # --- classify new point ----------------------------------------------------
        tk.Label(self.body, text='4) Classify a new data point (x, y)', bg='#cfe3ff', fg='#174a7e',
                 font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        p = tk.Frame(self.body, bg='white')
        p.pack(fill='x', padx=15, pady=10)
        tk.Label(p, text='x =', bg='white', font=('Segoe UI', 11)).grid(row=0, column=0, padx=8, pady=14)
        self.ex = tk.Entry(p, width=10, font=('Segoe UI', 11))
        self.ex.grid(row=0, column=1)
        tk.Label(p, text='y =', bg='white', font=('Segoe UI', 11)).grid(row=0, column=2, padx=8)
        self.ey = tk.Entry(p, width=10, font=('Segoe UI', 11))
        self.ey.grid(row=0, column=3)
        tk.Button(p, text='Classify Point', command=self.classify_point, bg='#174a7e', fg='white',
                  font=('Segoe UI', 10, 'bold')).grid(row=0, column=4, padx=15)
        self.out = tk.Label(p, text='Prediction: \u2014', bg='white', fg='#087f5b', font=('Segoe UI', 13, 'bold'))
        self.out.grid(row=0, column=5, padx=15)

        # model state
        self.X = self.y = None
        self.Xtr = self.ytr = self.Xte = self.yte = None
        self.mu = self.sigma = None
        self.w = self.b = None
        self.best_C = None
        self.last_point = None
        self.report = ''
        self.last_classification = ''
        root.after_idle(lambda: polish(root))

        default_csv = BASE / 'svm_linear_data.csv'
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
        self.X, self.y = generate_linear_data(500, seed=seed)
        self._prepare_after_load()

    def _prepare_after_load(self):
        self.Xtr_raw, self.ytr, self.Xte_raw, self.yte = train_test_split(self.X, self.y)
        self.mu, self.sigma = standardize_fit(self.Xtr_raw)
        self.Xtr = standardize_apply(self.Xtr_raw, self.mu, self.sigma)
        self.Xte = standardize_apply(self.Xte_raw, self.mu, self.sigma)
        self.w = self.b = None
        self.last_point = None
        self.report = ''
        self.last_classification = ''
        self.info.delete('1.0', 'end')
        self.info.insert('end', f'Generated {len(self.y)} samples (2 classes, {np.sum(self.y==1)} vs {np.sum(self.y==-1)}).\n'
                                 f'Train set: {len(self.ytr)}   Test set: {len(self.yte)}\n'
                                 f'Standardised with z-score: mu={np.round(self.mu,3)}, sigma={np.round(self.sigma,3)}\n\n'
                                 f'Click "Train / Evaluate" to run 5-fold cross validation over C in {self.C_GRID}\n'
                                 f'and train the final Pegasos linear SVM.\n')
        self.result.delete('1.0', 'end')
        self.out.config(text='Prediction: \u2014')
        self.plot_data_only()

    # ------------------------------------------------------------------
    def plot_data_only(self):
        self.ax_main.clear()
        self.ax_main.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], c='#2e6bd6', label='Class +1', edgecolor='white', s=45)
        self.ax_main.scatter(self.X[self.y == -1, 0], self.X[self.y == -1, 1], c='#e0473c', label='Class -1', edgecolor='white', s=45)
        self.ax_main.set_xlabel('x'); self.ax_main.set_ylabel('y')
        self.ax_main.set_title('Raw data (not yet trained)')
        self.ax_main.legend(loc='upper left')
        self.ax_main.grid(alpha=0.25)
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
        # 1) hyperparameter search via k-fold CV --------------------------------
        best_C, cv_results = grid_search_C(self.Xtr_raw, self.ytr, self.C_GRID, k=5, epochs=30)
        self.best_C = best_C
        # 2) final training on the full training set with the best C ------------
        self.w, self.b = pegasos_train(self.Xtr, self.ytr, C=best_C, epochs=150)
        # 3) evaluate on the held-out test set -----------------------------------
        scores = decision_function(self.Xte, self.w, self.b)
        pred = np.sign(scores)
        TP, FP, FN, TN = confusion_matrix_manual(self.yte, pred)
        acc, prec, rec, f1 = precision_recall_f1(TP, FP, FN, TN)
        fprs, tprs = roc_curve_manual(self.yte, scores)
        auc = auc_trapezoidal(fprs, tprs)

        # ---- info panel: equations + CV table + manual worked example ----------
        cv_lines = '\n'.join(f'   C = {C:<7g} -> mean CV accuracy = {a:.4f}' for C, a in cv_results.items())
        sv_mask = (self.ytr * (self.Xtr.dot(self.w) + self.b)) <= 1.0 + 1e-6
        n_sv = int(np.sum(sv_mask))
        margin_width = 2.0 / np.linalg.norm(self.w)

        ex_i = 0
        ex_x = self.Xte[ex_i]
        ex_true = self.yte[ex_i]
        ex_raw = self.w[0] * ex_x[0] + self.w[1] * ex_x[1]
        ex_f = ex_raw + self.b
        ex_pred = 1 if ex_f >= 0 else -1

        self.info.delete('1.0', 'end')
        self.info.insert('end',
            "OBJECTIVE (primal soft-margin SVM):\n"
            "  J(w,b) = (1/2)||w||^2 + C * sum_i max(0, 1 - y_i(w.x_i + b))\n\n"
            "PEGASOS UPDATE RULE (per random sample, lambda = 1/(C*n)):\n"
            "  eta_t = 1/(lambda*t)\n"
            "  if y_i(w.x_i+b) < 1 :  w <- (1-eta*lambda)w + eta*y_i*x_i ,  b <- b + eta*y_i\n"
            "  else               :  w <- (1-eta*lambda)w\n\n"
            f"HYPERPARAMETER SEARCH  (5-fold cross validation over C):\n{cv_lines}\n"
            f"  --> Selected C* = {best_C}  (highest mean CV accuracy)\n\n"
            f"FINAL TRAINED MODEL (on {len(self.ytr)} training points, standardised features):\n"
            f"  w = [{self.w[0]:.4f}, {self.w[1]:.4f}]      b = {self.b:.4f}\n"
            f"  Hyperplane   :  {self.w[0]:.4f}*x + {self.w[1]:.4f}*y + {self.b:.4f} = 0\n"
            f"  Margin width :  2 / ||w|| = {margin_width:.4f}   (this is what training maximises)\n"
            f"  Support vectors (points with y_i*f(x_i) <= 1) : {n_sv}\n\n"
            "MANUAL CALCULATION EXAMPLE (first test point, standardised coordinates):\n"
            f"  x = ({ex_x[0]:.4f}, {ex_x[1]:.4f}),  true class y = {ex_true:+d}\n"
            f"  f(x) = w.x + b = ({self.w[0]:.4f})*({ex_x[0]:.4f}) + ({self.w[1]:.4f})*({ex_x[1]:.4f}) + ({self.b:.4f})\n"
            f"       = {self.w[0]*ex_x[0]:.4f} + {self.w[1]*ex_x[1]:.4f} + {self.b:.4f} = {ex_f:.4f}\n"
            f"  sign(f(x)) = {ex_pred:+d}  ->  predicted class = {ex_pred:+d}"
            f"  ({'CORRECT' if ex_pred==ex_true else 'WRONG'})\n"
        )

        # ---- metrics panel -------------------------------------------------------
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
        self.report = ('=== LINEAR SVM CLASSIFIER REPORT ===\n\n'
                        + self.info.get('1.0', 'end') + '\n' + self.result.get('1.0', 'end'))

    # ------------------------------------------------------------------
    def plot_trained(self, sv_mask, fprs, tprs, auc):
        w, b = self.w, self.b
        Xs_all = np.vstack([self.Xtr, self.Xte])
        x_min, x_max = Xs_all[:, 0].min() - 1, Xs_all[:, 0].max() + 1
        y_min, y_max = Xs_all[:, 1].min() - 1, Xs_all[:, 1].max() + 1

        self.ax_main.clear()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        Z = (w[0] * xx + w[1] * yy + b)
        self.ax_main.contourf(xx, yy, Z, levels=[Z.min()-1, 0, Z.max()+1],
                               colors=['#ffe1de', '#dce8ff'], alpha=0.55)

        self.ax_main.scatter(self.Xtr[self.ytr == 1, 0], self.Xtr[self.ytr == 1, 1], c='#2e6bd6', label='Class +1 (train)', edgecolor='white', s=40)
        self.ax_main.scatter(self.Xtr[self.ytr == -1, 0], self.Xtr[self.ytr == -1, 1], c='#e0473c', label='Class -1 (train)', edgecolor='white', s=40)
        self.ax_main.scatter(self.Xte[:, 0], self.Xte[:, 1], facecolors='none', edgecolors='#333333', s=60, linewidths=1.2, label='Test points')
        self.ax_main.scatter(self.Xtr[sv_mask, 0], self.Xtr[sv_mask, 1], facecolors='none', edgecolors='black',
                              s=140, linewidths=1.6, label='Support vectors')

        if abs(w[1]) > 1e-9:
            xs = np.linspace(x_min, x_max, 200)
            self.ax_main.plot(xs, (0 - w[0]*xs - b)/w[1], 'k-', linewidth=2.2, label='Hyperplane  w.x+b=0')
            self.ax_main.plot(xs, (1 - w[0]*xs - b)/w[1], 'k--', linewidth=1.2, label='Margin  w.x+b=\u00b11')
            self.ax_main.plot(xs, (-1 - w[0]*xs - b)/w[1], 'k--', linewidth=1.2)
        else:
            x0 = -b / w[0]
            self.ax_main.axvline(x0, color='k', linewidth=2.2, label='Hyperplane')

        if self.last_point is not None:
            self.ax_main.scatter([self.last_point[0]], [self.last_point[1]], marker='*', s=320,
                                  c='#ffb703', edgecolor='black', linewidths=1.2, label='New point', zorder=5)

        self.ax_main.set_xlim(x_min, x_max); self.ax_main.set_ylim(y_min, y_max)
        self.ax_main.set_xlabel('x (standardised)'); self.ax_main.set_ylabel('y (standardised)')
        self.ax_main.set_title(f'Linear SVM  (C = {self.best_C})  \u2014  maximum margin hyperplane')
        self.ax_main.legend(loc='upper left', fontsize=8)
        self.ax_main.grid(alpha=0.2)
        self.fig_main.tight_layout()
        self.canvas_main.draw()

        self.ax_roc.clear()
        color = '#4f46e5'
        self.ax_roc.step(fprs, tprs, where='post', color=color, linewidth=2.8,
                         label=f'Linear SVM · AUC {auc:.3f}')
        self.ax_roc.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', linewidth=1.2,
                         label='Random classifier')
        self.ax_roc.fill_between(fprs, tprs, step='post', alpha=0.12, color=color)
        self.ax_roc.set_xlabel('False Positive Rate (1 − specificity)')
        self.ax_roc.set_ylabel('True Positive Rate (recall)')
        self.ax_roc.set_title('Test-set ROC', fontsize=13, fontweight='bold', pad=12)
        self.ax_roc.set_xlim(-.03, 1.03); self.ax_roc.set_ylim(-.03, 1.05)
        self.ax_roc.set_aspect('equal', adjustable='box')
        self.ax_roc.set_xticks(np.linspace(0, 1, 6)); self.ax_roc.set_yticks(np.linspace(0, 1, 6))
        self.ax_roc.legend(loc='lower right', fontsize=8, frameon=True)
        self.ax_roc.grid(alpha=0.18, color='#94a3b8')
        self.ax_roc.text(.04, .93, 'Perfect ranking' if auc > .999 else 'Higher is better',
                         transform=self.ax_roc.transAxes, color='#087f5b', fontsize=9,
                         fontweight='bold', va='top')
        self.fig_roc.tight_layout()
        self.canvas_roc.draw()

    # ------------------------------------------------------------------
    def classify_point(self):
        if self.w is None:
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
        f_val = float(self.w[0]*point_std[0] + self.w[1]*point_std[1] + self.b)
        cls = 1 if f_val >= 0 else -1
        self.last_point = point_std
        self.out.config(text=f'Prediction: Class {cls:+d}   (f(x)={f_val:.4f})')
        self.last_classification = (f"\nNEW POINT CLASSIFICATION\n"
                                     f"  input (x, y) = ({x_raw}, {y_raw})  ->  standardised = ({point_std[0]:.4f}, {point_std[1]:.4f})\n"
                                     f"  f(x) = w.x + b = {f_val:.4f}  ->  predicted class = {cls:+d}\n")
        self.report = self.report + self.last_classification if self.report else self.last_classification
        # redraw with the new point overlaid
        sv_mask = (self.ytr * (self.Xtr.dot(self.w) + self.b)) <= 1.0 + 1e-6
        scores = decision_function(self.Xte, self.w, self.b)
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
        window = tk.Toplevel(self.root)
        window.title('Linear SVM — Bangla Lab Guide')
        window.geometry('900x700')
        text = tk.Text(window, wrap='word', font=('Segoe UI', 11), padx=15, pady=15)
        scroll = ttk.Scrollbar(window, command=text.yview)
        text.config(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y'); text.pack(fill='both', expand=True)
        text.insert('1.0', (BASE / 'LAB_GUIDE_BN.txt').read_text(encoding='utf-8'))
        text.config(state='disabled')


if __name__ == '__main__':
    App = tk.Tk()
    LinearSVMApp(App)
    App.mainloop()
