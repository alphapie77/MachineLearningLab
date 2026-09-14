"""K-means teaching GUI: Tkinter interface with Matplotlib graphs."""
import csv
import math
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from kmeans_core import load_points_csv, kmeans_full, initial_centroids, elbow_method, euclidean, wcss

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from ui_theme import setup_theme, polish
COLORS = ['#2563eb', '#dc2626', '#059669', '#9333ea', '#d97706', '#0891b2', '#db2777']


class KMeansApp:
    def __init__(self, root):
        self.root = root
        setup_theme(root)
        root.title('K-Means Clustering')
        root.geometry('1180x820')
        root.minsize(920, 650)
        self.labels, self.coords, self.seed_vars = [], [], []
        self.model, self.last_point = None, None
        self.elbow_results = {}
        self.training_text = self.prediction = ''
        self.k = tk.StringVar(value='2')
        self.status = tk.StringVar()
        ttk.Label(root, text='K-MEANS CLUSTERING LAB', style='Title.TLabel').pack(anchor='w', padx=20, pady=(18,8))
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill='both', expand=True, padx=15)
        frames = [ttk.Frame(self.tabs, padding=12) for _ in range(5)]
        for f, name in zip(frames, ['1 · Dataset', '2 · Train & graph', '3 · Calculations', '4 · New input', '5 · Guide']):
            self.tabs.add(f, text=name)
        self.data_tab, self.train_tab, self.steps_tab, self.predict_tab, self.help_tab = frames
        ttk.Label(root, textvariable=self.status, wraplength=1100).pack(anchor='w', padx=20, pady=10)
        self.build_data()
        self.build_training()
        root.after_idle(lambda: polish(root))
        self.steps = self.textbox(self.steps_tab)
        ttk.Label(self.predict_tab, text='Input: numerical x, y · Output: nearest learned cluster').pack(anchor='w', pady=10)
        bar = ttk.Frame(self.predict_tab)
        bar.pack(fill='x', pady=10)
        self.px, self.py = tk.StringVar(value='2'), tk.StringVar(value='2')
        for name, var in [('x:', self.px), ('y:', self.py)]:
            ttk.Label(bar, text=name).pack(side='left', padx=5)
            ttk.Entry(bar, textvariable=var, width=12).pack(side='left')
        self.button(bar, 'Predict cluster', self.predict)
        self.pred_box = self.textbox(self.predict_tab)
        guide = self.textbox(self.help_tab)
        guide.insert('1.0', (BASE / 'LAB_GUIDE_BN.txt').read_text(encoding='utf-8'))
        guide.config(state='disabled')
        self.load_csv(BASE / 'kmeans_data.csv')
        self.k.trace_add('write', lambda *_: self.invalidate())

    def button(self, parent, title, command):
        primary = title.startswith(('Run', 'Predict', 'Prepare'))
        ttk.Button(parent, text=title, command=command, style='Primary.TButton' if primary else 'TButton').pack(side='left', padx=5)

    def textbox(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True)
        box = tk.Text(frame, wrap='word', font=('Consolas', 11), padx=10, pady=10)
        sb = ttk.Scrollbar(frame, command=box.yview)
        box.config(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        box.pack(fill='both', expand=True)
        return box

    def build_data(self):
        bar = ttk.Frame(self.data_tab)
        bar.pack(fill='x')
        self.button(bar, 'Load CSV', self.load_csv)
        self.button(bar, 'Restore example', lambda: self.load_csv(BASE / 'kmeans_data.csv'))
        self.button(bar, 'Save dataset CSV', self.save_data)
        ttk.Label(self.data_tab, text='Features: x, y · ID is not a target · Raw 2D coordinates (no scaling)').pack(anchor='w', pady=12)
        frame = ttk.Frame(self.data_tab)
        frame.pack(fill='both', expand=True)
        self.table = ttk.Treeview(frame, columns=('id', 'x', 'y'), show='headings')
        for col, title in [('id', 'Point ID'), ('x', 'Feature x'), ('y', 'Feature y')]:
            self.table.heading(col, text=title)
            self.table.column(col, width=180, anchor='center')
        sb = ttk.Scrollbar(frame, command=self.table.yview)
        self.table.config(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.table.pack(fill='both', expand=True)
        self.table.bind('<<TreeviewSelect>>', self.select_row)
        edit = ttk.Frame(self.data_tab)
        edit.pack(fill='x', pady=15)
        self.entries = []
        for title in ('ID', 'x', 'y'):
            ttk.Label(edit, text=title).pack(side='left', padx=5)
            entry = ttk.Entry(edit, width=10)
            entry.pack(side='left')
            self.entries.append(entry)
        self.button(edit, 'Add point', lambda: self.edit_point(False))
        self.button(edit, 'Update selected', lambda: self.edit_point(True))
        self.button(edit, 'Delete selected', self.delete_points)

    def build_training(self):
        bar = ttk.Frame(self.train_tab)
        bar.pack(fill='x')
        ttk.Label(bar, text='Clusters k:').pack(side='left')
        self.k_input = ttk.Spinbox(bar, from_=1, to=1, textvariable=self.k, width=5)
        self.k_input.pack(side='left', padx=5)
        self.button(bar, 'Prepare centroids', self.prepare)
        self.button(bar, 'Run K-Means', self.train)
        self.elbow_button = ttk.Button(bar, text='Elbow (load data first)', command=self.elbow)
        self.elbow_button.pack(side='left', padx=5)
        self.button(bar, 'Save report', self.save_report)
        ttk.Label(self.train_tab, text='k → initial centroids → Run K-Means').pack(anchor='w', pady=10)
        self.seed_canvas = tk.Canvas(self.train_tab, height=72, highlightthickness=0)
        self.seed_canvas.pack(fill='x')
        sb = ttk.Scrollbar(self.train_tab, orient='horizontal', command=self.seed_canvas.xview)
        sb.pack(fill='x')
        self.seed_canvas.config(xscrollcommand=sb.set)
        self.seeds = ttk.Frame(self.seed_canvas)
        self.seed_canvas.create_window(0, 0, window=self.seeds, anchor='nw')
        self.seeds.bind('<Configure>', lambda _: self.seed_canvas.configure(scrollregion=self.seed_canvas.bbox('all')))
        self.result = tk.StringVar(value='No trained model.')
        ttk.Label(self.train_tab, textvariable=self.result, wraplength=1050).pack(anchor='w', pady=10)
        plots = ttk.Frame(self.train_tab)
        plots.pack(fill='both', expand=True)
        self.figure = Figure(figsize=(10, 4), dpi=100, layout='constrained')
        self.ax, self.ax_elbow = self.figure.subplots(1, 2)
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=plots)
        self.plot_canvas.get_tk_widget().pack(fill='both', expand=True)

    def invalidate(self):
        self.model = self.last_point = None
        self.training_text = self.prediction = ''
        self.result.set('Train with the current dataset and settings to see results.')
        self.steps.delete('1.0', 'end')
        self.pred_box.delete('1.0', 'end')
        self.status.set('Data/settings changed. Prepare centroids if k changed, then train again.')
        self.draw()

    def refresh_data(self):
        self.table.delete(*self.table.get_children())
        for i, (label, point) in enumerate(zip(self.labels, self.coords)):
            self.table.insert('', 'end', iid=str(i), values=(label, *point))
        self.elbow_results = {}
        sample_count = len(self.coords)
        unique_count = len(set(self.coords))
        self.k_input.configure(to=max(1, unique_count))
        self.elbow_button.configure(text=f'Elbow (k=1..{unique_count})' if sample_count else 'Elbow (load data first)')
        self.invalidate()
        self.draw_elbow()
        if self.coords:
            try:
                k = int(self.k.get())
            except ValueError:
                k = 2
            self.k.set(str(max(1, min(k, unique_count))))
            self.prepare()
        self.status.set(f'{sample_count} samples · {unique_count} unique points · 2 features · no target labels. Use Save dataset CSV to keep edits.')

    def load_csv(self, path=None):
        path = path or filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])
        if not path:
            return
        try:
            labels, coords = load_points_csv(path)
            if len(coords) > 100 or any(abs(v) > 1e100 for p in coords for v in p):
                raise ValueError('Teaching app limit: 100 samples; coordinate magnitude <= 1e100.')
            self.labels, self.coords = labels, coords
            self.refresh_data()
        except (ValueError, OSError) as e:
            messagebox.showerror('Dataset error', str(e))

    def select_row(self, _=None):
        selected = self.table.selection()
        if len(selected) == 1:
            i = int(selected[0])
            for entry, value in zip(self.entries, (self.labels[i], *self.coords[i])):
                entry.delete(0, 'end')
                entry.insert(0, str(value))

    @staticmethod
    def number(value):
        value = float(value)
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError('Use finite numbers with magnitude <= 1e100.')
        return value

    def edit_point(self, update):
        try:
            selected = self.table.selection()
            if update and len(selected) != 1:
                raise ValueError('Select exactly one row to update.')
            if not update and len(self.coords) >= 100:
                raise ValueError('Maximum 100 samples for the step-by-step lab.')
            index = int(selected[0]) if update else None
            label = self.entries[0].get().strip()
            if not label or any(l == label and i != index for i, l in enumerate(self.labels)):
                raise ValueError('Enter a nonempty, unique point ID.')
            point = tuple(self.number(e.get()) for e in self.entries[1:])
            if update:
                self.labels[index], self.coords[index] = label, point
            else:
                self.labels.append(label)
                self.coords.append(point)
            self.refresh_data()
        except ValueError as e:
            messagebox.showerror('Invalid point', str(e))

    def delete_points(self):
        for i in sorted(map(int, self.table.selection()), reverse=True):
            del self.labels[i]
            del self.coords[i]
        self.refresh_data()

    def save_data(self):
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['label', 'x', 'y'])
                    writer.writerows((l, *p) for l, p in zip(self.labels, self.coords))
                self.status.set(f'Dataset saved: {path}')
            except OSError as e:
                messagebox.showerror('Save failed', str(e))

    def get_k(self):
        k = int(self.k.get())
        if not 1 <= k <= len(set(self.coords)):
            raise ValueError('k must be an integer from 1 to the number of unique data points.')
        return k

    def prepare(self):
        try:
            k = self.get_k()
        except ValueError as e:
            messagebox.showerror('Invalid k', str(e))
            return
        self.invalidate()
        for child in self.seeds.winfo_children():
            child.destroy()
        self.seed_vars = []
        for i, point in enumerate(initial_centroids(self.coords, k)):
            frame = ttk.LabelFrame(self.seeds, text=f'C{i+1}: x, y', padding=5)
            frame.pack(side='left', padx=4)
            pair = []
            for value in point:
                var = tk.StringVar(value=f'{value:g}')
                ttk.Entry(frame, textvariable=var, width=10).pack(side='left', padx=2)
                var.trace_add('write', lambda *_: self.invalidate())
                pair.append(var)
            self.seed_vars.append(pair)

    def train(self):
        try:
            k = self.get_k()
            if len(self.seed_vars) != k:
                raise ValueError('Click Prepare centroids after changing k.')
            seeds = [tuple(self.number(v.get()) for v in pair) for pair in self.seed_vars]
            c, a, history = kmeans_full(self.coords, seeds)
            self.invalidate()
            self.model = (c, a, history)
            lines = ['K-MEANS LAB REPORT', f'Samples={len(self.coords)}; features=x,y; k={k}',
                     'Raw values, no scaling. ID is not a target.',
                     'd(P,C)=sqrt((Px-Cx)^2+(Py-Cy)^2). Assign to nearest centroid.',
                     'Ties: lowest cluster number. Empty cluster: reseed at a farthest data point.', '\nDATASET']
            lines += [f'{l}: {p}' for l, p in zip(self.labels, self.coords)]
            for h in history:
                lines += [f'\nITERATION {h["iter"]}', f'Centroids before: {h["centroids_before"]}']
                for label, point, ds, cluster in zip(self.labels, self.coords, h['dist_table'], h['assignments']):
                    lines += [f'd({label},C{i+1})=sqrt(({point[0]:g}-({old[0]:.4f}))^2 + ({point[1]:g}-({old[1]:.4f}))^2) = {d:.4f}' for i, (old, d) in enumerate(zip(h['centroids_before'], ds))]
                    lines.append(f'  => {label}: Cluster {cluster+1}')
                for i, centroid in enumerate(h['centroids_after']):
                    members = [j for j, cluster in enumerate(h['assignments']) if cluster == i]
                    lines.append(f'C{i+1} members: ' + ', '.join(self.labels[j] for j in members))
                    if members:
                        for axis, name in enumerate(('x', 'y')):
                            expression = ' + '.join(f'({self.coords[j][axis]:g})' for j in members)
                            lines.append(f'  mean {name} = ({expression}) / {len(members)} = {centroid[axis]:.4f}')
                    else:
                        lines.append(f'  Empty cluster: reseed at farthest data point {centroid}.')
                if h['converged']:
                    lines.append('Assignments unchanged: convergence confirmed. Stop.')
            outcome = 'Converged' if history[-1]['converged'] else 'Iteration limit reached; NOT confirmed converged'
            summary = [f'{outcome}: {len(history)} assignment passes, including final stability check.']
            summary += [f'C{i+1}=({p[0]:.4f}, {p[1]:.4f}): ' + ', '.join(l for l, cluster in zip(self.labels, a) if cluster == i) for i, p in enumerate(c)]
            score = f'WCSS (sum of squared distances) = {wcss(self.coords, c, a):.4f}'
            summary.append(score)
            if len(set(a)) < k:
                summary.append('Empty clusters remain: try different initial centroids or a smaller k.')
            self.training_text = '\n'.join(lines + ['\nFINAL RESULT'] + summary)
            self.steps.insert('1.0', self.training_text)
            self.result.set('\n'.join(summary) if k <= 7 else summary[0] + '\n' + score + '\nSee Calculations for all centroids and members.')
            self.status.set('Training complete. Inspect Calculations or predict a point under New input.')
            self.draw()
        except ValueError as e:
            messagebox.showerror('Training input error', str(e))

    def predict(self):
        if self.model is None:
            messagebox.showinfo('Training required', 'Run K-Means with the current data and settings first.')
            return
        try:
            point = (self.number(self.px.get()), self.number(self.py.get()))
            c = self.model[0]
            ds = [euclidean(point, center) for center in c]
            cluster = ds.index(min(ds))
            lines = [f'New point: {point}', f'OUTPUT: Cluster {cluster+1}', '']
            lines += [f'd(P,C{i+1})=sqrt(({point[0]:g}-({center[0]:.4f}))^2 + ({point[1]:g}-({center[1]:.4f}))^2) = {d:.4f}' for i, (center, d) in enumerate(zip(c, ds))]
            lines += ['', 'Nearest centroid determines the cluster. Ties use the lowest cluster number.', 'No retraining. See the black diamond in the Train & graph tab.']
            self.prediction = '\n'.join(lines)
            self.last_point = point
            self.pred_box.delete('1.0', 'end')
            self.pred_box.insert('1.0', self.prediction)
            self.draw()
        except ValueError as e:
            messagebox.showerror('Invalid input', str(e))

    def elbow(self):
        if not self.coords:
            messagebox.showinfo('Dataset required', 'Add or load data first.')
            return
        self.elbow_results = elbow_method(self.coords, k_max=len(set(self.coords)))
        self.draw_elbow()
        self.status.set('Elbow: look for where improvement slows. k is NOT selected automatically. Values are included in reports.')

    def report_text(self):
        text = self.training_text
        if self.elbow_results:
            text += '\n\nELBOW RESULTS\n' + '\n'.join(f'k={k}: WCSS={w:.6f}' for k, w in self.elbow_results.items())
            text += '\nAll seed combinations if <=100; otherwise 20 deterministic restarts (seed 42). Best observed WCSS; no automatic k selection.'
        return text + ('\n\nNEW POINT\n' + self.prediction if self.prediction else '')

    def save_report(self):
        if self.model is None:
            messagebox.showinfo('Training required', 'Run K-Means first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text', '*.txt'), ('CSV report', '*.csv')])
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    if path.lower().endswith('.csv'):
                        writer = csv.writer(f)
                        writer.writerow(['report_line'])
                        writer.writerows([line] for line in self.report_text().splitlines())
                    else:
                        f.write(self.report_text())
                self.status.set(f'Report saved: {path}')
            except OSError as e:
                messagebox.showerror('Save failed', str(e))

    def draw(self):
        ax = self.ax
        ax.clear()
        if self.coords:
            assignments = self.model[1] if self.model else [0] * len(self.coords)
            for i in sorted(set(assignments)):
                points = [p for p, a in zip(self.coords, assignments) if a == i]
                ax.scatter(*zip(*points), color=COLORS[i % len(COLORS)],
                           label=f'Cluster {i+1}' if self.model else 'Data', s=55)
            for label, point in zip(self.labels, self.coords):
                ax.annotate(label, point, xytext=(5, 5), textcoords='offset points', fontsize=8)
            if self.model:
                for i, center in enumerate(self.model[0]):
                    ax.scatter(*center, marker='X', s=140, color=COLORS[i % len(COLORS)], edgecolor='black')
                    ax.annotate(f'C{i+1}', center, xytext=(6, -12), textcoords='offset points')
            if self.last_point:
                ax.scatter(*self.last_point, marker='D', color='black', s=65, label='New point')
            ax.legend(fontsize=8)
        ax.set(title='Clusters and centroids (X)', xlabel='x', ylabel='y')
        ax.set_aspect('equal', adjustable='datalim')
        ax.margins(.15)
        ax.grid(alpha=.25)
        self.plot_canvas.draw_idle()

    def draw_elbow(self):
        ax = self.ax_elbow
        ax.clear()
        if self.elbow_results:
            ks, values = list(self.elbow_results), list(self.elbow_results.values())
            ax.plot(ks, values, 'o-', color='#059669')
            ax.set_xticks(ks)
            for k, value in self.elbow_results.items():
                ax.annotate(f'{value:.2f}', (k, value), xytext=(0, 7),
                            textcoords='offset points', ha='center', fontsize=8)
            ax.margins(.15)
        else:
            ax.text(.5, .5, 'Click Elbow to compare k', ha='center', transform=ax.transAxes)
        ax.set(title='Elbow: best observed WCSS', xlabel='k (clusters)', ylabel='WCSS')
        ax.grid(alpha=.25)
        self.plot_canvas.draw_idle()


if __name__ == '__main__':
    root = tk.Tk()
    KMeansApp(root)
    root.mainloop()
