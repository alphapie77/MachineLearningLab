"""Exam-friendly ID3 Decision Tree GUI."""
import csv
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from decision_tree_core import (load_csv, entropy, information_gain, stratified_split,
    select_depth, build_tree, evaluate, predict, positive_probability, tree_stats, tree_text)

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from ui_theme import setup_theme, polish
TARGET = 'Play'


class App:
    def __init__(self, root):
        self.root = root
        setup_theme(root)
        root.title('ID3 Decision Tree Classifier')
        root.geometry('1200x820'); root.minsize(900, 650)
        self.rows, self.features, self.tree = [], [], None
        self.report = ''; self.status = tk.StringVar(value='Load data to begin.')
        ttk.Label(root, text='ID3 DECISION TREE CLASSIFIER', style='Title.TLabel').pack(anchor='w', padx=18, pady=(18,8))
        self.tabs = ttk.Notebook(root); self.tabs.pack(fill='both', expand=True, padx=14)
        frames = [ttk.Frame(self.tabs, padding=12) for _ in range(5)]
        for frame, title in zip(frames, ['1 · Dataset', '2 · Train & evaluate', '3 · Tree diagram', '4 · New input', '5 · Guide']): self.tabs.add(frame, text=title)
        self.data_tab, self.train_tab, self.tree_tab, self.predict_tab, guide_tab = frames
        ttk.Label(root, textvariable=self.status, wraplength=1150).pack(anchor='w', padx=18, pady=9)
        self.build_data(); self.build_train(); self.build_tree_tab(); self.build_predict()
        root.after_idle(lambda: polish(root))
        guide = self.textbox(guide_tab); guide.insert('1.0', (BASE/'LAB_GUIDE_BN.txt').read_text(encoding='utf-8')); guide.config(state='disabled')
        self.read(BASE/'decision_tree_data_1000.csv')

    @staticmethod
    def textbox(parent):
        frame = ttk.Frame(parent); frame.pack(fill='both', expand=True)
        box = tk.Text(frame, wrap='word', font=('Consolas', 10), padx=10, pady=10)
        sb = ttk.Scrollbar(frame, command=box.yview); box.config(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y'); box.pack(fill='both', expand=True); return box

    def build_data(self):
        bar = ttk.Frame(self.data_tab); bar.pack(fill='x')
        ttk.Button(bar, text='Load CSV', command=self.choose_csv).pack(side='left', padx=4)
        ttk.Button(bar, text='Restore example', command=lambda: self.read(BASE/'decision_tree_data_1000.csv')).pack(side='left', padx=4)
        self.data_info = ttk.Label(self.data_tab, wraplength=1100); self.data_info.pack(anchor='w', pady=10)
        frame = ttk.Frame(self.data_tab); frame.pack(fill='both', expand=True)
        self.table = ttk.Treeview(frame, show='headings')
        sy = ttk.Scrollbar(frame, command=self.table.yview); sx = ttk.Scrollbar(frame, orient='horizontal', command=self.table.xview)
        self.table.config(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side='right', fill='y'); sx.pack(side='bottom', fill='x'); self.table.pack(fill='both', expand=True)

    def build_train(self):
        bar = ttk.Frame(self.train_tab); bar.pack(fill='x')
        ttk.Label(bar, text='Candidate max depths: 1, 2, 3, 4, 5').pack(side='left')
        ttk.Button(bar, text='Train / Evaluate', command=self.train, style='Primary.TButton').pack(side='left', padx=12)
        ttk.Button(bar, text='Save report', command=self.save).pack(side='left', padx=4)
        ttk.Label(self.train_tab, text='80% training / 20% untouched test · 5-fold CV on training only · positive class = Yes').pack(anchor='w', pady=10)
        self.results = self.textbox(self.train_tab)

    def build_tree_tab(self):
        frame = ttk.Frame(self.tree_tab); frame.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(frame, bg='white', highlightthickness=1, highlightbackground='#cccccc')
        sx = ttk.Scrollbar(frame, orient='horizontal', command=self.canvas.xview); sy = ttk.Scrollbar(frame, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=sx.set, yscrollcommand=sy.set)
        sx.pack(side='bottom', fill='x'); sy.pack(side='right', fill='y'); self.canvas.pack(fill='both', expand=True)

    def build_predict(self):
        ttk.Label(self.predict_tab, text='Input: all feature values · Output: Play class and root-to-leaf path').pack(anchor='w', pady=10)
        self.input_frame = ttk.Frame(self.predict_tab); self.input_frame.pack(fill='x'); self.inputs = {}
        ttk.Button(self.predict_tab, text='Predict Play', command=self.classify, style='Primary.TButton').pack(anchor='w', pady=12)
        self.prediction_box = self.textbox(self.predict_tab)

    def choose_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])
        if path: self.read(path)

    def read(self, path):
        try:
            rows, features, target = load_csv(path, TARGET); self.rows, self.features = rows, features
            self.tree = None; self.report = ''; self.results.delete('1.0', 'end'); self.prediction_box.delete('1.0', 'end'); self.canvas.delete('all')
            columns = features + [target]; self.table.delete(*self.table.get_children()); self.table.config(columns=columns)
            for column in columns:
                self.table.heading(column, text=column + (' (target)' if column == target else '')); self.table.column(column, width=145, anchor='center')
            for i, row in enumerate(rows): self.table.insert('', 'end', iid=str(i), values=[row[c] for c in columns])
            classes = sorted({r[target] for r in rows})
            self.data_info.config(text=f'{len(rows)} samples · {len(features)} categorical features: {", ".join(features)} · Target: {target} · Classes: {", ".join(classes)}')
            for child in self.input_frame.winfo_children(): child.destroy()
            self.inputs = {}
            for feature in features:
                group = ttk.Frame(self.input_frame); group.pack(side='left', padx=7); ttk.Label(group, text=feature).pack()
                values = sorted({r[feature] for r in rows}); var = tk.StringVar(value=values[0])
                ttk.Combobox(group, textvariable=var, values=values, state='readonly', width=13).pack(); self.inputs[feature] = var
            self.status.set('Dataset loaded. Target labels are used for supervised training; click Train / Evaluate.')
        except (ValueError, OSError) as error: messagebox.showerror('Dataset error', str(error))

    def train(self):
        if not self.rows: return
        try:
            training, test = stratified_split(self.rows, TARGET); depth, cv = select_depth(training, self.features, TARGET)
            tree = build_tree(training, self.features, TARGET, depth); metrics = evaluate(test, tree, TARGET, positive='Yes')
            leaves, actual_depth = tree_stats(tree); root_entropy = entropy([row[TARGET] for row in training])
            cv_text = '\n'.join(f'  max_depth={d}: mean validation accuracy={score:.4f}' for d, score in cv.items())
            gain_text = '\n'.join(f'  Gain({f})={information_gain(training, f, TARGET):.6f}' for f in self.features)
            text = (f'DATA SPLIT (seed 42, stratified)\nTraining: {len(training)} rows (80%)\nTest: {len(test)} rows (20%, untouched until final evaluation)\n\n'
                    f'TRAINING ROOT ENTROPY\nH(Play)={root_entropy:.6f}\n\nROOT INFORMATION GAINS\n{gain_text}\n\n'
                    f'HYPERPARAMETER SELECTION — 5-fold CV on training only\n{cv_text}\nSelected max_depth={depth} (smaller depth wins a tie)\n\n'
                    f'FINAL MODEL\nTrained on all {len(training)} training rows\nLeaves={leaves}; actual depth={actual_depth}\n\n'
                    f'UNTOUCHED TEST EVALUATION\nAccuracy={metrics["accuracy"]:.4f}\nPrecision (Yes)={metrics["precision"]:.4f}\nRecall (Yes)={metrics["recall"]:.4f}\nF1 (Yes)={metrics["f1"]:.4f}\nROC-AUC (leaf Yes probability)={metrics["auc"]:.4f}\n\n'
                    f'Confusion matrix [rows=actual, columns=predicted]\n             Pred Yes  Pred {metrics["negative"]}\nActual Yes   {metrics["tp"]:8d}  {metrics["fn"]:8d}\nActual {metrics["negative"]:<3}   {metrics["fp"]:8d}  {metrics["tn"]:8d}\n\nTREE RULES\n{tree_text(tree)}')
            self.tree, self.report = tree, text; self.results.delete('1.0', 'end'); self.results.insert('1.0', text)
            self.draw_tree(); self.status.set('Training complete. Metrics come from the untouched test set; New input is ready.')
        except ValueError as error: messagebox.showerror('Training error', str(error))

    def classify(self):
        if self.tree is None: messagebox.showinfo('Training required', 'Click Train / Evaluate first.'); return
        row = {feature: var.get() for feature, var in self.inputs.items()}; label, node, path = predict(self.tree, row)
        lines = ['INPUT'] + [f'{f} = {row[f]}' for f in self.features] + ['', 'DECISION PATH']
        lines += [f'{feature} = {value}' for feature, value in path]
        lines += ['', f'OUTPUT: Play = {label}', f'Leaf counts: {node["counts"]}', f'Estimated P(Yes) = {positive_probability(node, "Yes"):.4f}']
        text = '\n'.join(lines); self.prediction_box.delete('1.0', 'end'); self.prediction_box.insert('1.0', text)
        self.report = self.report.split('\n\nLAST PREDICTION')[0] + '\n\nLAST PREDICTION\n' + text

    def save(self):
        if not self.report: messagebox.showinfo('Training required', 'Train the model first.'); return
        path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text', '*.txt'), ('CSV report', '*.csv')])
        if not path: return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as file:
                if path.lower().endswith('.csv'):
                    writer = csv.writer(file); writer.writerow(['report_line']); writer.writerows([line] for line in self.report.splitlines())
                else: file.write(self.report)
            self.status.set(f'Report saved: {path}')
        except OSError as error: messagebox.showerror('Save failed', str(error))

    def draw_tree(self):
        self.canvas.delete('all')
        if self.tree is None: return
        positions, counter = {}, [0]
        def locate(node, depth=0):
            if node['type'] == 'leaf': x = counter[0]; counter[0] += 1
            else:
                xs = [locate(child, depth + 1) for child in node['children'].values()]; x = sum(xs)/len(xs)
            positions[id(node)] = (x, depth); return x
        locate(self.tree); xgap, ygap, margin = 170, 115, 90
        _, depth = tree_stats(self.tree); self.canvas.config(scrollregion=(0, 0, max(900, counter[0]*xgap+180), max(500, (depth+1)*ygap+100)))
        def draw(node):
            x, d = positions[id(node)]; cx, cy = margin+x*xgap, 60+d*ygap
            if node['type'] == 'node':
                for value, child in node['children'].items():
                    x2, d2 = positions[id(child)]; cx2, cy2 = margin+x2*xgap, 60+d2*ygap
                    self.canvas.create_line(cx, cy+25, cx2, cy2-25, arrow=tk.LAST, fill='#64748b')
                    self.canvas.create_text((cx+cx2)/2, (cy+cy2)/2-8, text=value, fill='#7c3aed', font=('Segoe UI', 8, 'bold')); draw(child)
                self.canvas.create_rectangle(cx-62, cy-25, cx+62, cy+25, fill='#2563eb', outline='#1e3a8a')
                self.canvas.create_text(cx, cy, text=f'{node["feature"]}\ngain={node["gain"]:.3f}', fill='white', font=('Segoe UI', 9, 'bold'))
            else:
                color = '#059669' if node['prediction'] == 'Yes' else '#dc2626'
                self.canvas.create_oval(cx-62, cy-25, cx+62, cy+25, fill=color, outline='#333333')
                self.canvas.create_text(cx, cy, text=f'Play={node["prediction"]}\n{node["counts"]}', fill='white', font=('Segoe UI', 9, 'bold'))
        draw(self.tree)


if __name__ == '__main__':
    root = tk.Tk(); App(root); root.mainloop()
