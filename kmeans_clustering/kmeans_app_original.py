# =====================================================================================
#  K-MEANS CLUSTERING (k = 2)  --  built from scratch (no sklearn)
#  Only matplotlib is used for plotting; the clustering algorithm itself
#  (distance calculation, cluster assignment, centroid update, convergence
#  check, and the elbow method for choosing k) is implemented manually in
#  plain Python so every step / equation can be shown and explained.
#
#  THEORY / EQUATIONS USED (also reproduced inside the GUI panels)
#  -------------------------------------------------------------------------------
#  Euclidean distance between a point p=(px,py) and a centroid c=(cx,cy):
#       d(p, c) = sqrt( (px - cx)^2 + (py - cy)^2 )
#
#  K-MEANS ALGORITHM (Lloyd's algorithm):
#     1. Choose k initial centroids (here: entered by the user, c1 and c2).
#     2. ASSIGNMENT STEP: assign every point to its nearest centroid
#            cluster(p) = argmin_i  d(p, c_i)
#     3. UPDATE STEP: recompute each centroid as the mean of the points
#        currently assigned to it:
#            c_i_new = (1/|S_i|) * sum_{p in S_i} p
#     4. Repeat steps 2-3 until the cluster assignments stop changing
#        (convergence).
#
#  WITHIN-CLUSTER SUM OF SQUARES (WCSS / inertia), used for the elbow method:
#       WCSS(k) = sum_i sum_{p in cluster i} d(p, c_i)^2
#  The ELBOW METHOD runs k-means for several values of k and plots k against
#  WCSS(k); the "elbow" (point where the curve stops dropping sharply) is a
#  good choice of k. This is our hyperparameter search for k.
# =====================================================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv, os, math, itertools
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DEFAULT_CSV = 'kmeans_data.csv'
CLUSTER_COLORS = ['#2e6bd6', '#e0473c', '#1f9d55', '#f2a400']


# --------------------------------------------------------------------------------
# 1) DATA LOADING  --  CSV with columns: label,x,y
# --------------------------------------------------------------------------------
def load_points_csv(path):
    labels, coords = [], []
    with open(path, encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row.get('label', f'P{len(labels)+1}'))
            coords.append((float(row['x']), float(row['y'])))
    return labels, coords


# --------------------------------------------------------------------------------
# 2) EUCLIDEAN DISTANCE  --  d(p,c) = sqrt((px-cx)^2 + (py-cy)^2)
# --------------------------------------------------------------------------------
def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


# --------------------------------------------------------------------------------
# 3) ASSIGNMENT STEP  --  cluster(p) = argmin_i d(p, c_i)
# --------------------------------------------------------------------------------
def assign_clusters(coords, centroids):
    assignments = []
    dist_table = []
    for p in coords:
        dists = [euclidean(p, c) for c in centroids]
        dist_table.append(dists)
        assignments.append(dists.index(min(dists)))
    return assignments, dist_table


# --------------------------------------------------------------------------------
# 4) UPDATE STEP  --  c_i_new = mean of points assigned to cluster i
# --------------------------------------------------------------------------------
def update_centroids(coords, assignments, k, old_centroids):
    new_centroids = []
    for ci in range(k):
        members = [p for p, a in zip(coords, assignments) if a == ci]
        if members:
            mx = sum(p[0] for p in members) / len(members)
            my = sum(p[1] for p in members) / len(members)
            new_centroids.append((mx, my))
        else:
            new_centroids.append(old_centroids[ci])   # keep old centroid if cluster became empty
    return new_centroids


# --------------------------------------------------------------------------------
# 5) FULL K-MEANS LOOP  --  repeat assign + update until assignments stop changing
# --------------------------------------------------------------------------------
def kmeans_full(coords, init_centroids, max_iter=50):
    """ Returns (final_centroids, final_assignments, history) where history is a
        list of dicts, one per iteration, recording every intermediate value
        needed to reproduce the manual calculation. """
    centroids = list(init_centroids)
    history = []
    prev_assign = None
    it = 0
    while it < max_iter:
        it += 1
        assignments, dist_table = assign_clusters(coords, centroids)
        if assignments == prev_assign:
            break                                    # converged: no point changed cluster
        new_centroids = update_centroids(coords, assignments, len(centroids), centroids)
        history.append({'iter': it, 'centroids_before': list(centroids), 'dist_table': dist_table,
                         'assignments': list(assignments), 'centroids_after': list(new_centroids)})
        prev_assign = assignments
        centroids = new_centroids
    return centroids, prev_assign, history


def wcss(coords, centroids, assignments):
    """ Within-Cluster Sum of Squares (inertia): sum of squared distances of
        each point to the centroid of its assigned cluster. """
    return sum(euclidean(p, centroids[a])**2 for p, a in zip(coords, assignments))


# --------------------------------------------------------------------------------
# 6) ELBOW METHOD  --  brute-force search over which points make the best initial
#    centroids for each k (small dataset, so trying every combination is cheap),
#    then plot k vs. best WCSS(k) to justify the chosen k.
# --------------------------------------------------------------------------------
def elbow_method(coords, k_max=5):
    n = len(coords)
    k_max = min(k_max, n)
    results = {}
    for k in range(1, k_max + 1):
        best_w = None
        for combo in itertools.combinations(range(n), k):
            init = [coords[i] for i in combo]
            centroids, assignments, _ = kmeans_full(coords, init)
            w = wcss(coords, centroids, assignments)
            if best_w is None or w < best_w:
                best_w = w
        results[k] = best_w
    return results


# =====================================================================================
#  GUI APPLICATION
# =====================================================================================
class KMeansApp:
    def __init__(self, root):
        self.root = root
        root.title('K-Means Clustering (k=2)  \u2014  built from scratch')
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

        tk.Label(self.body, text='K-MEANS CLUSTERING  (k = 2, Euclidean distance)', bg='#0f6b4c', fg='white',
                 font=('Segoe UI', 20, 'bold'), pady=16).pack(fill='x')
        tk.Label(self.body, text='Pure Python implementation  \u2022  manual iteration log  \u2022  elbow method for choosing k',
                 bg='#eaf3ff', fg='#0f6b4c', font=('Segoe UI', 10, 'italic')).pack(fill='x', pady=(0, 6))

        bar = tk.Frame(self.body, bg='white')
        bar.pack(fill='x', padx=15, pady=10)
        tk.Button(bar, text='Load CSV', command=self.load_csv, bg='#e5f7ef').pack(side='left', padx=5, pady=8)
        tk.Button(bar, text='Run Elbow Method (k=1..5)', command=self.run_elbow, bg='#e5f7ef').pack(side='left', padx=5)
        tk.Button(bar, text='Save Report (TXT/CSV)', command=self.save_report, bg='#e5f7ef').pack(side='left', padx=5)

        # ---- initial centroid input -------------------------------------------------
        tk.Label(self.body, text='Initial centroids (enter, then click Run K-Means)', bg='#cdeedd', fg='#0f6b4c',
                 font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        cpanel = tk.Frame(self.body, bg='white')
        cpanel.pack(fill='x', padx=15, pady=10)
        tk.Label(cpanel, text='C1 =  (', bg='white', font=('Segoe UI', 11)).grid(row=0, column=0, padx=(10, 0), pady=14)
        self.c1x = tk.Entry(cpanel, width=7, font=('Segoe UI', 11)); self.c1x.grid(row=0, column=1)
        tk.Label(cpanel, text=',', bg='white').grid(row=0, column=2)
        self.c1y = tk.Entry(cpanel, width=7, font=('Segoe UI', 11)); self.c1y.grid(row=0, column=3)
        tk.Label(cpanel, text=')', bg='white').grid(row=0, column=4)
        tk.Label(cpanel, text='   C2 =  (', bg='white', font=('Segoe UI', 11)).grid(row=0, column=5, padx=(20, 0))
        self.c2x = tk.Entry(cpanel, width=7, font=('Segoe UI', 11)); self.c2x.grid(row=0, column=6)
        tk.Label(cpanel, text=',', bg='white').grid(row=0, column=7)
        self.c2y = tk.Entry(cpanel, width=7, font=('Segoe UI', 11)); self.c2y.grid(row=0, column=8)
        tk.Label(cpanel, text=')', bg='white').grid(row=0, column=9)
        tk.Button(cpanel, text='Run K-Means (k=2)', command=self.run_kmeans, bg='#0f6b4c', fg='white',
                  font=('Segoe UI', 10, 'bold')).grid(row=0, column=10, padx=20)
        self.c1x.insert(0, '1'); self.c1y.insert(0, '1')
        self.c2x.insert(0, '5'); self.c2y.insert(0, '7')

        # ---- diagrams -----------------------------------------------------------------
        tk.Label(self.body, text='Cluster diagram + elbow curve', bg='#cdeedd', fg='#0f6b4c',
                 font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        plots = tk.Frame(self.body, bg='#eaf3ff')
        plots.pack(fill='both', padx=15, pady=5)

        self.fig_main = Figure(figsize=(7.6, 5.6), dpi=100)
        self.ax_main = self.fig_main.add_subplot(111)
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=plots)
        self.canvas_main.get_tk_widget().pack(side='left', fill='both', expand=True)

        self.fig_elbow = Figure(figsize=(4.4, 5.6), dpi=100)
        self.ax_elbow = self.fig_elbow.add_subplot(111)
        self.canvas_elbow = FigureCanvasTkAgg(self.fig_elbow, master=plots)
        self.canvas_elbow.get_tk_widget().pack(side='left', fill='both', expand=True, padx=(10, 0))

        # ---- iteration log --------------------------------------------------------------
        tk.Label(self.body, text='Iteration-by-iteration manual calculation log', bg='#cdeedd', fg='#0f6b4c',
                 font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        self.log = tk.Text(self.body, height=20, bg='white', fg='black', font=('Consolas', 10))
        self.log.pack(fill='x', padx=15, pady=5)

        # ---- final results ---------------------------------------------------------------
        tk.Label(self.body, text='Final result', bg='#cdeedd', fg='#0f6b4c',
                 font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        self.result = tk.Text(self.body, height=9, bg='white', fg='black', font=('Consolas', 10))
        self.result.pack(fill='x', padx=15, pady=5)

        # ---- classify new point ------------------------------------------------------------
        tk.Label(self.body, text='Classify a new data point (x, y)', bg='#cdeedd', fg='#0f6b4c',
                 font=('Segoe UI', 12, 'bold'), anchor='w', padx=10, pady=7).pack(fill='x', padx=15, pady=(10, 3))
        p = tk.Frame(self.body, bg='white')
        p.pack(fill='x', padx=15, pady=10)
        tk.Label(p, text='x =', bg='white', font=('Segoe UI', 11)).grid(row=0, column=0, padx=8, pady=14)
        self.ex = tk.Entry(p, width=10, font=('Segoe UI', 11)); self.ex.grid(row=0, column=1)
        tk.Label(p, text='y =', bg='white', font=('Segoe UI', 11)).grid(row=0, column=2, padx=8)
        self.ey = tk.Entry(p, width=10, font=('Segoe UI', 11)); self.ey.grid(row=0, column=3)
        tk.Button(p, text='Classify Point', command=self.classify_point, bg='#0f6b4c', fg='white',
                  font=('Segoe UI', 10, 'bold')).grid(row=0, column=4, padx=15)
        self.out = tk.Label(p, text='Cluster: \u2014', bg='white', fg='#087f5b', font=('Segoe UI', 13, 'bold'))
        self.out.grid(row=0, column=5, padx=15)

        # state
        self.labels = self.coords = None
        self.final_centroids = self.final_assign = None
        self.elbow_results = None
        self.report = ''
        self.last_classification = ''

        default_path = DEFAULT_CSV
        if os.path.exists(default_path):
            self.load_csv(default_path)
        else:
            messagebox.showwarning('CSV not found', f'{default_path} not found next to this script.')

    # ------------------------------------------------------------------
    def load_csv(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])
            if not path:
                return
        try:
            self.labels, self.coords = load_points_csv(path)
        except Exception as e:
            messagebox.showerror('Error loading CSV', str(e))
            return
        self.final_centroids = self.final_assign = None
        self.elbow_results = None
        self.log.delete('1.0', 'end')
        self.result.delete('1.0', 'end')
        self.out.config(text='Cluster: \u2014')
        pts_str = ', '.join(f'{l}=({x:g},{y:g})' for l, (x, y) in zip(self.labels, self.coords))
        self.log.insert('end', f'Loaded {len(self.coords)} points:\n  {pts_str}\n\n'
                                'Enter initial centroids C1, C2 above and click "Run K-Means (k=2)".\n'
                                'Or click "Run Elbow Method" to see the WCSS-vs-k curve first.\n')
        self.plot_points_only()

    # ------------------------------------------------------------------
    def plot_points_only(self):
        self.ax_main.clear()
        xs = [p[0] for p in self.coords]; ys = [p[1] for p in self.coords]
        self.ax_main.scatter(xs, ys, c='#555', s=70, zorder=3)
        for l, (x, y) in zip(self.labels, self.coords):
            self.ax_main.annotate(l, (x, y), textcoords='offset points', xytext=(6, 6), fontsize=9)
        self.ax_main.set_xlabel('x'); self.ax_main.set_ylabel('y')
        self.ax_main.set_title('Data points (not yet clustered)')
        self.ax_main.grid(alpha=0.25)
        self.fig_main.tight_layout()
        self.canvas_main.draw()
        self.ax_elbow.clear()
        self.ax_elbow.set_title('Elbow curve (k vs WCSS)')
        self.ax_elbow.set_xlabel('k (number of clusters)'); self.ax_elbow.set_ylabel('WCSS (inertia)')
        self.fig_elbow.tight_layout()
        self.canvas_elbow.draw()

    # ------------------------------------------------------------------
    def run_elbow(self):
        if not self.coords:
            return
        self.elbow_results = elbow_method(self.coords, k_max=min(5, len(self.coords)))
        ks = sorted(self.elbow_results)
        ws = [self.elbow_results[k] for k in ks]

        self.ax_elbow.clear()
        self.ax_elbow.plot(ks, ws, 'o-', color='#0f6b4c', linewidth=2)
        self.ax_elbow.set_xlabel('k (number of clusters)'); self.ax_elbow.set_ylabel('WCSS (inertia)')
        self.ax_elbow.set_title('Elbow method')
        self.ax_elbow.set_xticks(ks)
        self.ax_elbow.grid(alpha=0.3)
        self.fig_elbow.tight_layout()
        self.canvas_elbow.draw()

        lines = '\n'.join(f'   k={k} -> best WCSS = {w:.4f}' for k, w in self.elbow_results.items())
        self.log.insert('end',
            "\n=== ELBOW METHOD (hyperparameter search for k) ===\n"
            "WCSS(k) = sum over clusters of sum of squared distances of each point to its centroid.\n"
            "For each k, every combination of k points was tried as the initial centroids and the\n"
            "run with the lowest WCSS was kept (brute force, feasible because the dataset is small).\n"
            f"{lines}\n"
            "The 'elbow' (where WCSS stops dropping sharply) indicates a good choice of k -> here it\n"
            "points to k = 2, which matches the two natural groups visible in the data.\n"
        )
        self.log.see('end')

    # ------------------------------------------------------------------
    def run_kmeans(self):
        if not self.coords:
            return
        try:
            c1 = (float(self.c1x.get()), float(self.c1y.get()))
            c2 = (float(self.c2x.get()), float(self.c2y.get()))
        except ValueError:
            messagebox.showerror('Invalid input', 'Please enter numeric values for C1 and C2.')
            return

        final_centroids, final_assign, history = kmeans_full(self.coords, [c1, c2])
        self.final_centroids = final_centroids
        self.final_assign = final_assign

        self.log.delete('1.0', 'end')
        self.log.insert('end', f'Initial centroids:  C1 = {c1},  C2 = {c2}\n')
        for h in history:
            it = h['iter']
            cb = h['centroids_before']
            block = [f"\n--- ITERATION {it} ---",
                     f"Centroids used this iteration: C1={self._fmt(cb[0])}, C2={self._fmt(cb[1])}",
                     f"{'Point':8s} {'d to C1':>10s} {'d to C2':>10s}  Assigned"]
            for lbl, p, dists, a in zip(self.labels, self.coords, h['dist_table'], h['assignments']):
                block.append(f"{lbl:8s} {dists[0]:10.4f} {dists[1]:10.4f}  -> Cluster {a+1}")
            c1_mem = [l for l, a in zip(self.labels, h['assignments']) if a == 0]
            c2_mem = [l for l, a in zip(self.labels, h['assignments']) if a == 1]
            ca = h['centroids_after']
            block.append(f"\nCluster 1 members: {{{', '.join(c1_mem) or '-'}}}  ->  new C1 = mean = {self._fmt(ca[0])}")
            block.append(f"Cluster 2 members: {{{', '.join(c2_mem) or '-'}}}  ->  new C2 = mean = {self._fmt(ca[1])}")
            self.log.insert('end', '\n'.join(block) + '\n')
        self.log.insert('end', f"\nAssignments unchanged from the previous iteration -> CONVERGED after {len(history)} iteration(s).\n")

        w = wcss(self.coords, final_centroids, final_assign)
        self.result.delete('1.0', 'end')
        c1_mem = [l for l, a in zip(self.labels, final_assign) if a == 0]
        c2_mem = [l for l, a in zip(self.labels, final_assign) if a == 1]
        self.result.insert('end',
            f"Final C1 = {self._fmt(final_centroids[0])}   members: {{{', '.join(c1_mem) or '-'}}}\n"
            f"Final C2 = {self._fmt(final_centroids[1])}   members: {{{', '.join(c2_mem) or '-'}}}\n"
            f"Total WCSS (inertia) = {w:.4f}\n"
            f"Converged in {len(history)} iteration(s)\n"
        )

        self.plot_clusters()
        self.report = ('=== K-MEANS CLUSTERING REPORT (k=2) ===\n\n'
                        + self.log.get('1.0', 'end') + '\n' + self.result.get('1.0', 'end'))

    @staticmethod
    def _fmt(pt):
        return f'({pt[0]:.4f}, {pt[1]:.4f})'

    # ------------------------------------------------------------------
    def plot_clusters(self):
        self.ax_main.clear()
        for ci in range(2):
            xs = [p[0] for p, a in zip(self.coords, self.final_assign) if a == ci]
            ys = [p[1] for p, a in zip(self.coords, self.final_assign) if a == ci]
            self.ax_main.scatter(xs, ys, c=CLUSTER_COLORS[ci], s=90, label=f'Cluster {ci+1}',
                                  edgecolor='white', zorder=3)
        for l, (x, y) in zip(self.labels, self.coords):
            self.ax_main.annotate(l, (x, y), textcoords='offset points', xytext=(6, 6), fontsize=9)
        for ci, c in enumerate(self.final_centroids):
            self.ax_main.scatter([c[0]], [c[1]], marker='X', s=220, c=CLUSTER_COLORS[ci],
                                  edgecolor='black', linewidths=1.5, zorder=4)
            self.ax_main.annotate(f'C{ci+1}', c, textcoords='offset points', xytext=(8, -12),
                                   fontsize=10, fontweight='bold')
        if hasattr(self, 'last_point') and self.last_point is not None:
            self.ax_main.scatter([self.last_point[0]], [self.last_point[1]], marker='*', s=320,
                                  c='#ffb703', edgecolor='black', linewidths=1.2, label='New point', zorder=5)
        self.ax_main.set_xlabel('x'); self.ax_main.set_ylabel('y')
        self.ax_main.set_title('K-Means result (k=2)')
        self.ax_main.legend(loc='best', fontsize=9)
        self.ax_main.grid(alpha=0.25)
        self.fig_main.tight_layout()
        self.canvas_main.draw()

    # ------------------------------------------------------------------
    def classify_point(self):
        if self.final_centroids is None:
            messagebox.showinfo('Run K-Means first', 'Please click "Run K-Means (k=2)" before classifying a new point.')
            return
        try:
            x = float(self.ex.get()); y = float(self.ey.get())
        except ValueError:
            messagebox.showerror('Invalid input', 'Please enter numeric values for x and y.')
            return
        pt = (x, y)
        dists = [euclidean(pt, c) for c in self.final_centroids]
        cluster = dists.index(min(dists))
        self.last_point = pt
        self.out.config(text=f'Cluster: {cluster+1}   (d(P,C1)={dists[0]:.4f}, d(P,C2)={dists[1]:.4f})')
        self.last_classification = (f"\nNEW POINT CLASSIFICATION\n"
                                     f"  point = ({x:g}, {y:g})\n"
                                     f"  d(P, C1) = sqrt(({x:g}-{self.final_centroids[0][0]:.4f})^2 + "
                                     f"({y:g}-{self.final_centroids[0][1]:.4f})^2) = {dists[0]:.4f}\n"
                                     f"  d(P, C2) = sqrt(({x:g}-{self.final_centroids[1][0]:.4f})^2 + "
                                     f"({y:g}-{self.final_centroids[1][1]:.4f})^2) = {dists[1]:.4f}\n"
                                     f"  nearest centroid -> Cluster {cluster+1}\n")
        self.report = (self.report + self.last_classification) if self.report else self.last_classification
        self.plot_clusters()

    # ------------------------------------------------------------------
    def save_report(self):
        if not self.report:
            messagebox.showinfo('Nothing to save', 'Please run K-Means first (click "Run K-Means (k=2)").')
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


if __name__ == '__main__':
    App = tk.Tk()
    KMeansApp(App)
    App.mainloop()
