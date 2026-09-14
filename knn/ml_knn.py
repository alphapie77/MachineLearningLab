"""Interactive, exam-friendly KNN shirt-size classification lab."""
import csv, math, sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from knn_core import load_csv, train_evaluate, classify, transform_point

BASE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent))
from ui_theme import setup_theme, polish
DATA_FILE = BASE / "shirt_size_data_1000.csv"

class KNNApp:
    def __init__(self, root):
        self.root=root; setup_theme(root); root.title("KNN Shirt Size Classifier"); root.geometry("1120x800"); root.minsize(880,620)
        self.rows=[]; self.result=None; self.prediction=None; self.source=None
        ttk.Label(root,text="KNN SHIRT SIZE CLASSIFIER",style="Title.TLabel").pack(anchor="w",padx=18,pady=(18,8))
        self.tabs=ttk.Notebook(root); self.tabs.pack(fill="both",expand=True,padx=14)
        frames=[ttk.Frame(self.tabs,padding=12) for _ in range(5)]
        for f,t in zip(frames,["1 · Dataset","2 · Train/Evaluate","3 · Confusion & ROC","4 · New Input","5 · Guide"]): self.tabs.add(f,text=t)
        self.data_tab,self.eval_tab,self.chart_tab,self.input_tab,guide_tab=frames
        self.status=tk.StringVar(value="Load a dataset, then run evaluation."); ttk.Label(root,textvariable=self.status).pack(anchor="w",padx=18,pady=9)
        self.build_dataset(); self.build_evaluation(); self.build_charts(); self.build_input()
        root.after_idle(lambda: polish(root))
        guide=self.textbox(guide_tab); guide.insert("1.0",(BASE/"LAB_GUIDE_BN.txt").read_text(encoding="utf-8")); guide.config(state="disabled")
        self.load_path(DATA_FILE)

    @staticmethod
    def textbox(parent):
        t=tk.Text(parent,wrap="word",font=("Consolas",10),padx=10,pady=10); t.pack(fill="both",expand=True); return t

    def build_dataset(self):
        bar=ttk.Frame(self.data_tab); bar.pack(fill="x"); ttk.Button(bar,text="Load CSV",command=self.choose_csv).pack(side="left")
        self.file_label=ttk.Label(bar); self.file_label.pack(side="left",padx=12); self.data_text=self.textbox(self.data_tab)

    def build_evaluation(self):
        ttk.Button(self.eval_tab,text="Run Leakage-Free Evaluation",command=self.evaluate,style="Primary.TButton").pack(anchor="w",pady=(0,10)); self.eval_text=self.textbox(self.eval_tab)

    def build_charts(self):
        self.figure=Figure(figsize=(9,5),dpi=100,layout="constrained"); self.cm_ax,self.roc_ax=self.figure.subplots(1,2)
        self.canvas=FigureCanvasTkAgg(self.figure,master=self.chart_tab); self.canvas.get_tk_widget().pack(fill="both",expand=True); self.draw_charts()

    def build_input(self):
        form=ttk.Frame(self.input_tab); form.pack(fill="x"); ttk.Label(form,text="Height (cm):").pack(side="left")
        self.hvar=tk.StringVar(); ttk.Entry(form,textvariable=self.hvar,width=12).pack(side="left",padx=6); ttk.Label(form,text="Weight (kg):").pack(side="left")
        self.wvar=tk.StringVar(); ttk.Entry(form,textvariable=self.wvar,width=12).pack(side="left",padx=6)
        ttk.Button(form,text="Classify",command=self.predict,style="Primary.TButton").pack(side="left",padx=8); ttk.Button(form,text="Save report",command=self.save_report).pack(side="left")
        self.pred_label=ttk.Label(self.input_tab,text="Prediction: —",font=("Segoe UI",15,"bold")); self.pred_label.pack(anchor="w",pady=12); self.pred_text=self.textbox(self.input_tab)

    def choose_csv(self):
        path=filedialog.askopenfilename(initialdir=BASE,filetypes=[("CSV files","*.csv")]);
        if path: self.load_path(Path(path))

    def load_path(self,path):
        try: rows=load_csv(path)
        except Exception as e: messagebox.showerror("Dataset error",str(e)); return
        self.rows=rows; self.source=Path(path); self.result=None; self.prediction=None
        self.eval_text.delete("1.0","end"); self.pred_text.delete("1.0","end"); self.pred_label.config(text="Prediction: —")
        labels=sorted(set(r[2] for r in rows)); counts={x:sum(r[2]==x for r in rows) for x in labels}; hs=[r[0] for r in rows]; ws=[r[1] for r in rows]
        self.file_label.config(text=str(self.source)); self.data_text.delete("1.0","end")
        self.data_text.insert("1.0",f"Rows: {len(rows)}\nFeatures: height_cm, weight_kg\nTarget: shirt_size\nClasses: {counts}\nHeight range: {min(hs):.2f}–{max(hs):.2f} cm\nWeight range: {min(ws):.2f}–{max(ws):.2f} kg\n\nPipeline: stratified 80/20 split → training-only scaling → 5-fold CV on training only → best k by macro-F1 → untouched test evaluation.")
        self.status.set("Dataset loaded. Old model/results were cleared."); self.draw_charts()

    def evaluate(self):
        if not self.rows: messagebox.showwarning("Dataset required","Load a valid dataset first."); return
        try: self.result=train_evaluate(self.rows)
        except Exception as e: messagebox.showerror("Evaluation error",str(e)); return
        r=self.result; lines=["LEAKAGE-FREE KNN PIPELINE",f"Training rows: {len(r['train_raw'])}",f"Untouched test rows: {len(r['test_raw'])}","Scaling: training mean/std only","Selection metric: validation macro-F1","","5-FOLD CV (TRAINING SET ONLY)"]
        lines += [f"k={k}: macro-F1={v*100:.2f}%" for k,v in r["cv"].items()]
        lines += ["",f"Selected k: {r['best_k']}","","UNTOUCHED TEST SET",f"Accuracy: {r['accuracy']*100:.2f}%",f"Macro precision: {r['macro_precision']*100:.2f}%",f"Macro recall / balanced accuracy: {r['macro_recall']*100:.2f}%",f"Macro F1: {r['macro_f1']*100:.2f}%"]
        for label in r["labels"]:
            m=r["per_class"][label]; lines.append(f"{label}: precision={m['precision']*100:.2f}%, recall={m['recall']*100:.2f}%, F1={m['f1']*100:.2f}%, support={m['support']}")
        self.eval_text.delete("1.0","end"); self.eval_text.insert("1.0","\n".join(lines)); self.status.set("Evaluation complete. Enter a new height and weight."); self.draw_charts()

    def draw_charts(self):
        self.cm_ax.clear(); self.roc_ax.clear(); self.cm_ax.set_title("Confusion matrix"); self.roc_ax.set_title("ROC curve")
        if not self.result:
            self.cm_ax.text(.5,.5,"Run evaluation",ha="center"); self.roc_ax.text(.5,.5,"Run evaluation",ha="center"); self.canvas.draw_idle(); return
        r=self.result; labels=r["labels"]; values=[[r["matrix"][a][p] for p in labels] for a in labels]; self.cm_ax.imshow(values,cmap="Blues")
        for i,row in enumerate(values):
            for j,value in enumerate(row): self.cm_ax.text(j,i,str(value),ha="center",va="center")
        self.cm_ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,xlabel="Predicted",ylabel="Actual")
        positive=min(labels,key=lambda x:sum(r["matrix"][x].values())); scored=[]
        for row in r["test"]:
            _,_,votes=classify(r["train"],row,r["best_k"],details=True); scored.append((votes.get(positive,0)/r["best_k"],int(row[2]==positive)))
        points=[]
        for t in sorted({1.1,-.1,*(s for s,_ in scored)},reverse=True):
            tp=sum(y and s>=t for s,y in scored); fn=sum(y and s<t for s,y in scored); fp=sum(not y and s>=t for s,y in scored); tn=sum(not y and s<t for s,y in scored)
            points.append((fp/(fp+tn) if fp+tn else 0,tp/(tp+fn) if tp+fn else 0))
        points.sort(); auc=sum((points[i][0]-points[i-1][0])*(points[i][1]+points[i-1][1])/2 for i in range(1,len(points)))
        self.roc_ax.plot([0,1],[0,1],"--",label="Random"); self.roc_ax.plot([x for x,_ in points],[y for _,y in points],"o-",label=f"Positive={positive}, AUC={auc:.3f}")
        self.roc_ax.set(xlabel="False positive rate",ylabel="True positive rate",xlim=(0,1),ylim=(0,1)); self.roc_ax.grid(alpha=.25); self.roc_ax.legend(); r.update(roc_positive=positive,auc=auc); self.canvas.draw_idle()

    def predict(self):
        if not self.result: messagebox.showwarning("Evaluation required","Run evaluation first."); return
        try: h,w=float(self.hvar.get()),float(self.wvar.get())
        except ValueError: messagebox.showerror("Invalid input","Height and weight must be numeric."); return
        if not math.isfinite(h) or not math.isfinite(w) or h<=0 or w<=0: messagebox.showerror("Invalid input","Enter finite positive height and weight."); return
        r=self.result; point=transform_point((h,w),r["means"],r["stds"]); label,near,votes=classify(r["train"],point,r["best_k"],details=True)
        raw=[(d,lab,sh*r["stds"][0]+r["means"][0],sw*r["stds"][1]+r["means"][1]) for d,lab,sh,sw in near]
        hs=[x[0] for x in r["train_raw"]]; ws=[x[1] for x in r["train_raw"]]; warning="\nWarning: outside training range; prediction is extrapolation.\n" if not(min(hs)<=h<=max(hs) and min(ws)<=w<=max(ws)) else ""
        lines=[f"Input: height={h:.2f} cm, weight={w:.2f} kg",f"Scaled input: ({point[0]:.4f}, {point[1]:.4f})",warning,f"NEAREST {r['best_k']} NEIGHBOURS","Rank | Distance | Height | Weight | Class"]
        lines += [f"{i:>4} | {d:>8.4f} | {nh:>6.2f} | {nw:>6.2f} | {lab}" for i,(d,lab,nh,nw) in enumerate(raw,1)]
        lines += ["","Votes: "+", ".join(f"{x}={votes[x]}" for x in sorted(votes)),f"Majority vote → {label}"]
        self.prediction=dict(height=h,weight=w,label=label); self.pred_label.config(text=f"Prediction: {label}"); self.pred_text.delete("1.0","end"); self.pred_text.insert("1.0","\n".join(lines)); self.status.set("Prediction complete.")

    def report(self):
        parts=["KNN SHIRT SIZE LAB REPORT",f"Dataset: {self.source}"]
        if self.result: parts += [self.eval_text.get("1.0","end").strip(),f"ROC positive={self.result.get('roc_positive')} AUC={self.result.get('auc',0):.4f}"]
        if self.prediction: parts.append(self.pred_text.get("1.0","end").strip())
        return "\n\n".join(parts)

    def save_report(self):
        if not self.result: messagebox.showwarning("Evaluation required","Run evaluation first."); return
        path=filedialog.asksaveasfilename(initialdir=BASE,defaultextension=".txt",filetypes=[("Text","*.txt"),("CSV","*.csv")])
        if not path:return
        with open(path,"w",newline="",encoding="utf-8") as f:
            if path.lower().endswith(".csv"):
                w=csv.writer(f); w.writerow(["report_line"]); w.writerows([line] for line in self.report().splitlines())
            else:f.write(self.report())
        self.status.set(f"Report saved: {path}")

if __name__=="__main__":
    root=tk.Tk(); KNNApp(root); root.mainloop()
