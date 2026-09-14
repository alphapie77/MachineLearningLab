"""Train and demonstrate a real binary CNN on CIFAR-10 cats and dogs."""
import csv, math, os, sys, threading, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from ui_theme import setup_theme, polish
MODEL_PATH = BASE/'cat_dog_cnn.keras'
# Keep the 170 MB dataset cache beside the lab instead of filling the user's
# system drive.  This must be set before TensorFlow/Keras is imported.
KERAS_HOME = BASE/'.keras'
os.environ.setdefault('KERAS_HOME', str(KERAS_HOME))


class CNNApp:
    def __init__(self, root):
        self.root = root; setup_theme(root); root.title('CNN Cat vs Dog Classifier'); root.geometry('1180x820'); root.minsize(900,650)
        self.data = self.model = self.history = self.metrics = None
        self.image = self.photo = self.last_prediction = None
        self.busy = False; self.status = tk.StringVar(value='Step 1: Prepare the dataset.')
        self.epochs = tk.StringVar(value='12'); self.per_class = tk.StringVar(value='5000')
        ttk.Label(root,text='CNN CAT vs DOG CLASSIFIER',style='Title.TLabel').pack(anchor='w',padx=18,pady=(18,8))
        self.tabs=ttk.Notebook(root); self.tabs.pack(fill='both',expand=True,padx=14)
        frames=[ttk.Frame(self.tabs,padding=12) for _ in range(5)]
        for f,t in zip(frames,['1 · Dataset','2 · Train CNN','3 · Evaluation','4 · Image input','5 · Guide']): self.tabs.add(f,text=t)
        self.data_tab,self.train_tab,self.eval_tab,self.input_tab,guide_tab=frames
        ttk.Label(root,textvariable=self.status,wraplength=1140).pack(anchor='w',padx=18,pady=9)
        self.build_data(); self.build_train(); self.eval_text=self.textbox(self.eval_tab); self.build_input()
        root.after_idle(lambda: polish(root))
        guide=self.textbox(guide_tab); guide.insert('1.0',(BASE/'LAB_GUIDE_BN.txt').read_text(encoding='utf-8')); guide.config(state='disabled')
        if MODEL_PATH.exists(): self.load_saved_model()

    @staticmethod
    def textbox(parent):
        f=ttk.Frame(parent); f.pack(fill='both',expand=True); t=tk.Text(f,wrap='word',font=('Consolas',10),padx=10,pady=10)
        s=ttk.Scrollbar(f,command=t.yview); t.config(yscrollcommand=s.set); s.pack(side='right',fill='y'); t.pack(fill='both',expand=True); return t

    def build_data(self):
        bar=ttk.Frame(self.data_tab); bar.pack(fill='x')
        ttk.Label(bar,text='Training images per class:').pack(side='left')
        ttk.Spinbox(bar,from_=100,to=5000,increment=100,textvariable=self.per_class,width=8).pack(side='left',padx=6)
        self.prepare_btn=ttk.Button(bar,text='Prepare CIFAR-10 Cat/Dog Dataset',command=self.prepare_data,style='Primary.TButton'); self.prepare_btn.pack(side='left',padx=8)
        ttk.Label(self.data_tab,text='CIFAR-10 · 32×32 RGB · class 3=Cat, class 5=Dog · official test set untouched').pack(anchor='w',pady=10)
        self.data_info=self.textbox(self.data_tab)

    def build_train(self):
        bar=ttk.Frame(self.train_tab); bar.pack(fill='x'); ttk.Label(bar,text='Epochs (1–20):').pack(side='left')
        ttk.Spinbox(bar,from_=1,to=20,textvariable=self.epochs,width=6).pack(side='left',padx=6)
        self.train_btn=ttk.Button(bar,text='Train CNN',command=self.train,style='Primary.TButton'); self.train_btn.pack(side='left',padx=8)
        ttk.Button(bar,text='Load saved model',command=self.load_saved_model).pack(side='left',padx=5)
        ttk.Button(bar,text='Save report',command=self.save_report).pack(side='left',padx=5)
        ttk.Label(self.train_tab,text='Conv32 → Pool → Conv64 → Pool → Flatten → Dense64 → Sigmoid').pack(anchor='w',pady=10)
        self.train_log=tk.Text(self.train_tab,height=9,font=('Consolas',10)); self.train_log.pack(fill='x',pady=5)
        self.figure=Figure(figsize=(9,3.5),dpi=100,layout='constrained'); self.loss_ax,self.acc_ax=self.figure.subplots(1,2)
        self.chart=FigureCanvasTkAgg(self.figure,master=self.train_tab); self.chart.get_tk_widget().pack(fill='both',expand=True)
        self.draw_history()

    def build_input(self):
        bar=ttk.Frame(self.input_tab); bar.pack(fill='x')
        ttk.Button(bar,text='Choose Cat/Dog Image',command=self.choose_image).pack(side='left',padx=5)
        self.pred_label=ttk.Label(bar,text='Prediction: —',font=('Segoe UI',15,'bold')); self.pred_label.pack(side='left',padx=15)
        body=ttk.Frame(self.input_tab); body.pack(fill='both',expand=True,pady=12)
        self.image_label=ttk.Label(body,text='Choose a JPG/PNG/BMP/WebP image',anchor='center'); self.image_label.pack(side='left',fill='both',expand=True)
        self.pred_text=tk.Text(body,width=55,wrap='word',font=('Consolas',10)); self.pred_text.pack(side='left',fill='both',expand=True,padx=(10,0))

    def valid_int(self,var,low,high,name):
        try: value=int(var.get())
        except ValueError: raise ValueError(f'{name} must be a whole number.')
        if not low<=value<=high: raise ValueError(f'{name} must be between {low} and {high}.')
        return value

    def set_busy(self,value,text=None):
        self.busy=value
        for button in (self.prepare_btn,self.train_btn): button.config(state='disabled' if value else 'normal')
        if text: self.status.set(text)

    def background(self,work,done):
        if self.busy: return
        self.set_busy(True)
        def runner():
            try: result=work(); self.root.after(0,lambda: finish(result,None))
            except Exception: error=traceback.format_exc(); self.root.after(0,lambda: finish(None,error))
        def finish(result,error):
            self.set_busy(False)
            if error: self.status.set('Operation failed.'); messagebox.showerror('CNN error',error)
            else: done(result)
        threading.Thread(target=runner,daemon=True).start()

    def prepare_data(self):
        try: count=self.valid_int(self.per_class,100,5000,'Images per class')
        except ValueError as e: messagebox.showerror('Invalid input',str(e)); return
        self.set_busy(False,'Downloading/loading CIFAR-10 and selecting cats/dogs...')
        def work():
            from tensorflow.keras.datasets import cifar10
            (x_train,y_train),(x_test,y_test)=cifar10.load_data(); y_train=y_train.ravel(); y_test=y_test.ravel()
            rng=np.random.default_rng(42); train_parts=[]; label_parts=[]
            for original,binary in ((3,0),(5,1)):
                idx=np.where(y_train==original)[0]; rng.shuffle(idx); idx=idx[:count]
                train_parts.append(x_train[idx]); label_parts.append(np.full(len(idx),binary,dtype=np.float32))
            X=np.concatenate(train_parts).astype('float32')/255.; y=np.concatenate(label_parts)
            order=rng.permutation(len(y)); X,y=X[order],y[order]
            test_idx=np.where((y_test==3)|(y_test==5))[0]; Xt=x_test[test_idx].astype('float32')/255.; yt=(y_test[test_idx]==5).astype('float32')
            # 20% of selected training rows becomes validation; official test stays untouched.
            split=int(len(y)*.8); return X[:split],y[:split],X[split:],y[split:],Xt,yt
        def done(data):
            self.data=data; self.model=self.history=self.metrics=None; self.last_prediction=None
            X,y,Xv,yv,Xt,yt=data; self.data_info.delete('1.0','end')
            self.data_info.insert('1.0',f'Source: CIFAR-10\nImage shape: 32 × 32 × 3 (RGB)\nClasses: CAT=0, DOG=1\nTraining: {len(y)} images\nValidation: {len(yv)} images\nOfficial untouched test: {len(yt)} images\nPixel preprocessing: uint8 [0,255] → float32 [0,1]\nRandom seed: 42\nNo test image is used for training or validation.')
            self.status.set('Dataset ready. Step 2: choose epochs and Train CNN.')
        self.background(work,done)

    @staticmethod
    def make_model():
        from tensorflow import keras
        return keras.Sequential([keras.layers.Input((32,32,3)),keras.layers.Conv2D(32,3,activation='relu'),keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64,3,activation='relu'),keras.layers.MaxPooling2D(),keras.layers.Flatten(),
            keras.layers.Dense(64,activation='relu'),keras.layers.Dropout(.3),keras.layers.Dense(1,activation='sigmoid')])

    def train(self):
        if self.data is None: messagebox.showinfo('Dataset required','Prepare the dataset first.'); return
        try: epochs=self.valid_int(self.epochs,1,20,'Epochs')
        except ValueError as e: messagebox.showerror('Invalid input',str(e)); return
        X,y,Xv,yv,Xt,yt=self.data; self.set_busy(False,f'Training for {epochs} epochs...')
        def work():
            import tensorflow as tf
            tf.keras.utils.set_random_seed(42); model=self.make_model(); model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
            history=model.fit(X,y,validation_data=(Xv,yv),epochs=epochs,batch_size=64,verbose=0,callbacks=[callback])
            loss,acc=model.evaluate(Xt,yt,verbose=0); probabilities=model.predict(Xt,verbose=0).ravel(); pred=(probabilities>=.5).astype(int)
            tp=int(np.sum((yt==1)&(pred==1))); tn=int(np.sum((yt==0)&(pred==0))); fp=int(np.sum((yt==0)&(pred==1))); fn=int(np.sum((yt==1)&(pred==0)))
            precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.
            model.save(MODEL_PATH); return model,history.history,dict(loss=loss,accuracy=acc,precision=precision,recall=recall,f1=f1,tp=tp,tn=tn,fp=fp,fn=fn,test=len(yt),epochs=epochs,train=len(y),validation=len(yv))
        def done(result):
            self.model,self.history,self.metrics=result; self.show_training(); self.show_evaluation(); self.status.set(f'Training complete; model saved as {MODEL_PATH.name}. New image input is ready.')
        self.background(work,done)

    def show_training(self):
        h=self.history; self.train_log.delete('1.0','end')
        lines=['TRAINING HISTORY']+[f'Epoch {i+1}: loss={h["loss"][i]:.4f}, accuracy={h["accuracy"][i]:.4f}, val_loss={h["val_loss"][i]:.4f}, val_accuracy={h["val_accuracy"][i]:.4f}' for i in range(len(h['loss']))]
        self.train_log.insert('1.0','\n'.join(lines)); self.draw_history()

    def draw_history(self):
        self.loss_ax.clear(); self.acc_ax.clear()
        if self.history:
            e=range(1,len(self.history['loss'])+1); self.loss_ax.plot(e,self.history['loss'],'o-',label='train'); self.loss_ax.plot(e,self.history['val_loss'],'o-',label='validation')
            self.acc_ax.plot(e,self.history['accuracy'],'o-',label='train'); self.acc_ax.plot(e,self.history['val_accuracy'],'o-',label='validation')
            self.loss_ax.legend(); self.acc_ax.legend()
        self.loss_ax.set(title='Binary cross-entropy loss',xlabel='epoch',ylabel='loss'); self.acc_ax.set(title='Accuracy',xlabel='epoch',ylabel='accuracy')
        self.loss_ax.grid(alpha=.25); self.acc_ax.grid(alpha=.25); self.chart.draw_idle()

    def show_evaluation(self):
        if not self.metrics: return
        m=self.metrics; text=(f'UNTOUCHED CIFAR-10 TEST SET (n={m["test"]})\nAccuracy={m["accuracy"]:.4f}\nLoss={m["loss"]:.4f}\nPrecision (Dog)={m["precision"]:.4f}\nRecall (Dog)={m["recall"]:.4f}\nF1 (Dog)={m["f1"]:.4f}\n\nConfusion Matrix [rows=actual, columns=predicted]\n             Pred Cat  Pred Dog\nActual Cat   {m["tn"]:8d}  {m["fp"]:8d}\nActual Dog   {m["fn"]:8d}  {m["tp"]:8d}\n\nThreshold: sigmoid probability >= 0.5 → DOG; otherwise CAT.')
        self.eval_text.delete('1.0','end'); self.eval_text.insert('1.0',text)

    def load_saved_model(self):
        if not MODEL_PATH.exists(): self.status.set('No saved model yet. Prepare data and train first.'); return
        self.set_busy(False,'Loading saved CNN model...')
        self.background(lambda: __import__('tensorflow').keras.models.load_model(MODEL_PATH),lambda model:(setattr(self,'model',model),self.status.set('Saved model loaded. Choose an image.')))

    def choose_image(self):
        if self.model is None: messagebox.showinfo('Model required','Train or load a saved model first.'); return
        path=filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png *.bmp *.webp')])
        if not path:return
        try:
            self.pred_label.config(text='Prediction: analysing new image…')
            self.pred_text.delete('1.0','end'); self.pred_text.insert('1.0',f'Loading: {Path(path).name}')
            self.root.update_idletasks()
            image=Image.open(path); image.load(); image=image.convert('RGB')
            if image.width<8 or image.height<8: raise ValueError('Image is too small; minimum is 8×8 pixels.')
            preview=image.copy(); preview.thumbnail((430,430)); self.photo=ImageTk.PhotoImage(preview); self.image_label.config(image=self.photo,text='')
            array=np.asarray(image.resize((32,32)),dtype=np.float32)/255.; probability=float(self.model.predict(array[None,...],verbose=0)[0,0])
            label='DOG' if probability>=.5 else 'CAT'; confidence=probability if label=='DOG' else 1-probability
            self.last_prediction=dict(path=path,label=label,dog_probability=probability,confidence=confidence,original=image.size)
            self.pred_label.config(text=f'Prediction: {label} ({confidence*100:.2f}%) · {Path(path).name}'); self.pred_text.delete('1.0','end')
            self.pred_text.insert('1.0',f'INPUT\nFile: {path}\nOriginal size: {image.width}×{image.height}\nConverted to RGB\nResized to 32×32\nNormalized: pixel/255\n\nMODEL OUTPUT\nSigmoid P(Dog)={probability:.6f}\nThreshold=0.5\nPrediction={label}\nClass confidence={confidence:.6f}\n\nExternal photos differ from tiny CIFAR-10 images, so confidence is not a guarantee.')
        except Exception as e: messagebox.showerror('Image error',str(e))

    def report(self):
        parts=['CNN CAT/DOG LAB REPORT','Architecture: Conv32-Pool-Conv64-Pool-Flatten-Dense64-Dropout-Sigmoid']
        if self.metrics: parts.append(self.eval_text.get('1.0','end').strip()); parts.append(self.train_log.get('1.0','end').strip())
        if self.last_prediction: parts.append(self.pred_text.get('1.0','end').strip())
        return '\n\n'.join(parts)

    def save_report(self):
        if not self.metrics: messagebox.showinfo('Training required','Train and evaluate first.'); return
        path=filedialog.asksaveasfilename(defaultextension='.txt',filetypes=[('Text','*.txt'),('CSV report','*.csv')])
        if not path:return
        with open(path,'w',newline='',encoding='utf-8') as file:
            if path.lower().endswith('.csv'):
                w=csv.writer(file); w.writerow(['report_line']); w.writerows([line] for line in self.report().splitlines())
            else:file.write(self.report())
        self.status.set(f'Report saved: {path}')


if __name__=='__main__':
    root=tk.Tk(); CNNApp(root); root.mainloop()
