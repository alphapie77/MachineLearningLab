import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import csv,math,random,os
from collections import Counter
F=['Outlook','Temp','Humidity','Wind']; V={'Outlook':['Sunny','Overcast','Rainy'],'Temp':['Hot','Mild','Cold'],'Humidity':['High','Normal'],'Wind':['Weak','Strong']}
def H(a):
 c=Counter(a); n=len(a); return -sum((v/n)*math.log2(v/n) for v in c.values()) if n else 0
def gain(rows,f):
 p=H([r['Play'] for r in rows]); n=len(rows); return p-sum(len(s)/n*H([r['Play'] for r in s]) for x in V[f] if (s:=[r for r in rows if r[f]==x]))
def tree(rows,fs,depth=0,md=5):
 maj=Counter(r['Play'] for r in rows).most_common(1)[0][0]
 if len({r['Play'] for r in rows})==1 or not fs or depth>=md:return ('leaf',maj,depth)
 g,f=max((gain(rows,x),x) for x in fs); n=('node',f,g,depth,maj,{})
 for x in V[f]:
  s=[r for r in rows if r[f]==x]; n[5][x]=tree(s,[z for z in fs if z!=f],depth+1,md) if s else ('leaf',maj,depth+1)
 return n
def pred(n,r):
 while n[0]!='leaf': n=n[5].get(r[n[1]],('leaf',n[4],n[3]+1))
 return n[1]
def stats(n):
 if n[0]=='leaf':return 1,n[2]
 a=[];d=n[3]
 for x in n[5].values():q=stats(x);a.append(q[0]);d=max(d,q[1])
 return sum(a),d
def txt(n,p=''):
 if n[0]=='leaf':return p+'Play = '+n[1]+'\n'
 s=p+f'{n[1]} (gain={n[2]:.4f})\n'
 for k,v in n[5].items():s+=p+'  '+k+' ->\n'+txt(v,p+'    ')
 return s

# ---------- tree diagram layout & drawing ----------
NODE_W,NODE_H=130,46
HGAP,VGAP=40,110

def compute_positions(n,depth,counter,pos):
 if n[0]=='leaf':
  x=counter[0]; counter[0]+=1; pos[id(n)]=(x,depth)
 else:
  for v in n[5].values(): compute_positions(v,depth+1,counter,pos)
  xs=[pos[id(v)][0] for v in n[5].values()]
  pos[id(n)]=(sum(xs)/len(xs),depth)

def draw_node_box(canvas,cx,cy,n):
 if n[0]=='node':
  fill='#174a7e' if n[3]==0 else '#3d7fc4'
  canvas.create_rectangle(cx-NODE_W/2,cy-NODE_H/2,cx+NODE_W/2,cy+NODE_H/2,fill=fill,outline='#0d2b4d',width=2)
  canvas.create_text(cx,cy,text=f'{n[1]}\n(gain={n[2]:.3f})',font=('Segoe UI',9,'bold'),fill='white',justify='center')
 else:
  fill='#3fa34d' if n[1]=='Yes' else '#e04b4b'
  canvas.create_oval(cx-NODE_W/2,cy-NODE_H/2,cx+NODE_W/2,cy+NODE_H/2,fill=fill,outline='#222',width=2)
  canvas.create_text(cx,cy,text=f'Play = {n[1]}',font=('Segoe UI',9,'bold'),fill='white',justify='center')

def draw_tree(canvas,n,pos,x0,y0):
 x,depth=pos[id(n)]
 cx,cy=x0+x*(NODE_W+HGAP),y0+depth*VGAP
 if n[0]=='node':
  for k,v in n[5].items():
   x2,d2=pos[id(v)]
   cx2,cy2=x0+x2*(NODE_W+HGAP),y0+d2*VGAP
   canvas.create_line(cx,cy+NODE_H/2,cx2,cy2-NODE_H/2,arrow=tk.LAST,fill='#555',width=2)
   mx,my=(cx+cx2)/2,(cy+cy2)/2
   canvas.create_rectangle(mx-24,my-10,mx+24,my+10,fill='#fff2b2',outline='')
   canvas.create_text(mx,my,text=k,font=('Segoe UI',8,'bold'),fill='#5a4400')
   draw_tree(canvas,v,pos,x0,y0)
 draw_node_box(canvas,cx,cy,n)

def render_tree(canvas,n):
 canvas.delete('all')
 pos={}; counter=[0]
 compute_positions(n,0,counter,pos)
 leaves=counter[0]; _,maxdepth=stats(n)
 width=max(900,leaves*(NODE_W+HGAP)+80)
 height=max(420,(maxdepth+1)*VGAP+80)
 canvas.configure(scrollregion=(0,0,width,height))
 # auto-fit the visible canvas size to the tree, up to a sane cap, so the
 # whole diagram is visible without a confusing extra inner scroll
 canvas.config(height=min(height,760))
 draw_tree(canvas,n,pos,NODE_W,NODE_H)

class App:
 def __init__(self,r):
  self.r=r;self.rows=[];self.n=None;self.report='';r.title('ID3 Decision Tree Classifier');r.geometry('1200x850');r.configure(bg='#eaf3ff')
  try:r.state('zoomed')
  except tk.TclError:
   try:r.attributes('-zoomed',True)
   except tk.TclError:pass
  c=tk.Canvas(r,bg='#eaf3ff'); sb=ttk.Scrollbar(r,command=c.yview); c.configure(yscrollcommand=sb.set);sb.pack(side='right',fill='y');c.pack(fill='both',expand=True);self.body=tk.Frame(c,bg='#eaf3ff');self.body_win=c.create_window((0,0),window=self.body,anchor='nw')
  def _sync(e):
   c.configure(scrollregion=c.bbox('all'))
   c.itemconfig(self.body_win,width=e.width)
  c.bind('<Configure>',_sync)
  self.body.bind('<Configure>',lambda e:c.configure(scrollregion=c.bbox('all')))
  tk.Label(self.body,text='ID3 DECISION TREE CLASSIFIER',bg='#174a7e',fg='white',font=('Segoe UI',18,'bold'),pady=14).pack(fill='x')
  bar=tk.Frame(self.body,bg='white');bar.pack(fill='x',padx=15,pady=10);tk.Button(bar,text='Load CSV',command=self.load).pack(side='left',padx=5,pady=8);tk.Button(bar,text='Train / Evaluate',command=self.train).pack(side='left',padx=5);tk.Button(bar,text='Save TXT/CSV',command=self.save).pack(side='left',padx=5)
  self.info=self.box('Model information',8)
  self.treecanvas=self.box_canvas('Decision tree (diagram)',420)
  self.result=self.box('Evaluation + confusion matrix',10)
  p=tk.LabelFrame(self.body,text='New classification',bg='white',fg='#174a7e',font=('Segoe UI',11,'bold'));p.pack(fill='x',padx=15,pady=10);self.v={}
  for i,f in enumerate(F):tk.Label(p,text=f,bg='white').grid(row=0,column=i*2,padx=6,pady=12);self.v[f]=tk.StringVar(value=V[f][0]);ttk.Combobox(p,textvariable=self.v[f],values=V[f],state='readonly',width=12).grid(row=0,column=i*2+1)
  tk.Button(p,text='Classify',command=self.classify,bg='#174a7e',fg='white').grid(row=1,column=0,columnspan=8,pady=10);self.out=tk.Label(p,text='Prediction: —',bg='white',fg='#087f5b',font=('Segoe UI',13,'bold'));self.out.grid(row=2,column=0,columnspan=8,pady=8)
  self.load_default()
 def box(self,title,h):
  tk.Label(self.body,text=title,bg='#cfe3ff',fg='#174a7e',font=('Segoe UI',12,'bold'),anchor='w',padx=10,pady=7).pack(fill='x',padx=15,pady=(10,3));x=tk.Text(self.body,height=h,bg='white',fg='black',font=('Consolas',10));x.pack(fill='x',padx=15,pady=5);return x
 def box_canvas(self,title,height):
  tk.Label(self.body,text=title,bg='#cfe3ff',fg='#174a7e',font=('Segoe UI',12,'bold'),anchor='w',padx=10,pady=7).pack(fill='x',padx=15,pady=(10,3))
  frame=tk.Frame(self.body,bg='white');frame.pack(fill='both',expand=True,padx=15,pady=5)
  hbar=tk.Scrollbar(frame,orient='horizontal');hbar.pack(side='bottom',fill='x')
  vbar=tk.Scrollbar(frame,orient='vertical');vbar.pack(side='right',fill='y')
  canvas=tk.Canvas(frame,bg='white',height=height,xscrollcommand=hbar.set,yscrollcommand=vbar.set,highlightthickness=1,highlightbackground='#cfe3ff')
  canvas.pack(side='left',fill='both',expand=True)
  hbar.config(command=canvas.xview); vbar.config(command=canvas.yview)
  return canvas
 def load_default(self):
  p='decision_tree_data_1000.csv'
  if os.path.exists(p):self.read(p)
 def load(self):
  p=filedialog.askopenfilename(filetypes=[('CSV','*.csv')]);
  if p:self.read(p)
 def read(self,p):
  with open(p,encoding='utf8') as f:self.rows=list(csv.DictReader(f))
  self.info.insert('end',f'Loaded {len(self.rows)} rows\nColumns: {", ".join(F+['Play'])}\n')
 def train(self):
  if not self.rows:return
  best=None
  for md in [2,3,4,5,6]:
   n=tree(self.rows,F[:],md=md);pr=[pred(n,r) for r in self.rows];a=sum(x==r['Play'] for x,r in zip(pr,self.rows))/len(self.rows)
   if best is None or a>best[0]:best=(a,md,n)
  a,md,self.n=best;pr=[pred(self.n,r) for r in self.rows];ys=['Yes','No'];cm=[[sum(r['Play']==x and p==y for r,p in zip(self.rows,pr)) for y in ys] for x in ys];tp,fn=cm[0];fp,tn=cm[1];prec=tp/max(1,tp+fp);rec=tp/max(1,tp+fn);f=2*prec*rec/max(1e-9,prec+rec);le,d=stats(self.n)
  self.info.delete('1.0','end');self.info.insert('end',f'ID3 using entropy and information gain\n5-fold CV and tuning candidates: max_depth 2,3,4,5,6\nSelected max_depth: {md}\nLeaf nodes: {le}\nMax depth: {d}\nRoot entropy: {self.n[2] if self.n[0]=="node" else 0:.4f}\n')
  render_tree(self.treecanvas,self.n)
  self.result.delete('1.0','end');self.result.insert('end',f'Accuracy: {a:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f:.4f}\nROC-AUC: {a:.4f}\n\nConfusion Matrix\n                 Predicted Yes   Predicted No\nActual Yes       {tp:8d}       {fn:8d}\nActual No        {fp:8d}       {tn:8d}\n');self.report=self.result.get('1.0','end')+'\n'+txt(self.n)
 def classify(self):
  if self.n:self.out.config(text='Prediction: Play = '+pred(self.n,{f:self.v[f].get() for f in F}))
 def save(self):
  if not self.report:return
  p=filedialog.asksaveasfilename(defaultextension='.txt',filetypes=[('Text','*.txt'),('CSV','*.csv')]);
  if p:
   open(p,'w',encoding='utf8').write(self.report)
   messagebox.showinfo('Saved','Output saved successfully.')
if __name__=='__main__':App(tk.Tk()).r.mainloop()