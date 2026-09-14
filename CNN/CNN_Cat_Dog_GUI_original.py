import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import threading

# Required packages:
# pip install tensorflow pillow numpy

class CNNClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Cat vs Dog Image Classifier")
        self.root.geometry("1050x760")
        self.root.minsize(700, 520)

        self.model = None
        self.current_image = None
        self.photo = None

        self.build_ui()

    def build_ui(self):
        self.root.configure(bg="#eaf3ff")

        outer = tk.Frame(self.root, bg="#eaf3ff")
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg="#eaf3ff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#eaf3ff")

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        title = tk.Label(scroll_frame, text="CNN Cat vs Dog Classifier",
                         font=("Segoe UI", 22, "bold"), bg="#eaf3ff", fg="#12355b")
        title.pack(pady=(18, 4))

        subtitle = tk.Label(scroll_frame,
            text="Convolutional Neural Network • Image preprocessing • Prediction • Evaluation",
            font=("Segoe UI", 10), bg="#eaf3ff", fg="#42627a")
        subtitle.pack(pady=(0, 15))

        control = tk.Frame(scroll_frame, bg="white", bd=1, relief="solid")
        control.pack(fill="x", padx=22, pady=8)

        tk.Button(control, text="Load CNN Model", command=self.load_model,
                  bg="#1769aa", fg="white", font=("Segoe UI", 11, "bold"),
                  padx=14, pady=8).pack(side="left", padx=12, pady=12)
        tk.Button(control, text="Choose Cat/Dog Image", command=self.choose_image,
                  bg="#198754", fg="white", font=("Segoe UI", 11, "bold"),
                  padx=14, pady=8).pack(side="left", padx=5, pady=12)
        tk.Button(control, text="Evaluate / Confusion Matrix",
                  command=self.show_evaluation,
                  bg="#6f42c1", fg="white", font=("Segoe UI", 11, "bold"),
                  padx=14, pady=8).pack(side="left", padx=5, pady=12)

        self.status = tk.Label(scroll_frame, text="Status: Model not loaded",
                               font=("Segoe UI", 11), bg="#eaf3ff", fg="#9c2c2c")
        self.status.pack(pady=8)

        image_box = tk.LabelFrame(scroll_frame, text="Input Image",
                                  font=("Segoe UI", 12, "bold"),
                                  bg="white", fg="#12355b", padx=10, pady=10)
        image_box.pack(fill="both", expand=True, padx=22, pady=8)

        self.image_label = tk.Label(image_box, text="Choose a cat or dog image",
                                    bg="#f7fbff", fg="#6c7a89",
                                    font=("Segoe UI", 13), width=55, height=16)
        self.image_label.pack(fill="both", expand=True)

        result_box = tk.LabelFrame(scroll_frame, text="CNN Prediction Output",
                                   font=("Segoe UI", 12, "bold"),
                                   bg="white", fg="#12355b", padx=12, pady=12)
        result_box.pack(fill="x", padx=22, pady=8)

        self.prediction = tk.Label(result_box, text="Prediction: ---",
                                   font=("Segoe UI", 18, "bold"),
                                   bg="white", fg="#12355b")
        self.prediction.pack(pady=5)

        self.confidence = tk.Label(result_box, text="Confidence: ---",
                                   font=("Segoe UI", 12), bg="white", fg="#42627a")
        self.confidence.pack(pady=5)

        self.details = tk.Label(result_box,
            text=("CNN steps: resize image → normalize pixels → convolution → "
                  "activation → pooling → feature extraction → classification"),
            wraplength=900, justify="left", bg="white", fg="#42627a",
            font=("Segoe UI", 10))
        self.details.pack(pady=8)

        eval_box = tk.LabelFrame(scroll_frame, text="Evaluation / Confusion Matrix",
                                font=("Segoe UI", 12, "bold"),
                                bg="white", fg="#12355b", padx=12, pady=12)
        eval_box.pack(fill="x", padx=22, pady=8)

        self.eval_text = tk.Text(eval_box, height=8, width=90, wrap="word",
                                 font=("Consolas", 10), bg="#f8fbff")
        self.eval_text.pack(fill="both", expand=True)
        self.eval_text.insert("1.0",
            "Evaluation is optional. Put labeled images inside dataset/cat and dataset/dog,\n"
            "then click Evaluate / Confusion Matrix.\n")

    def load_model(self):
        def worker():
            try:
                self.status.config(text="Status: Loading MobileNetV2 CNN weights...")
                from tensorflow.keras.applications import MobileNetV2
                self.model = MobileNetV2(weights="imagenet")
                self.status.config(text="Status: CNN model loaded successfully",
                                   fg="#087f23")
            except Exception as e:
                self.status.config(text="Status: Model loading failed", fg="#9c2c2c")
                messagebox.showerror("Model Error",
                    "Install requirements and ensure internet is available for first-time weights download.\n\n"+str(e))
        threading.Thread(target=worker, daemon=True).start()

    def choose_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.current_image = img
            preview = img.copy()
            preview.thumbnail((650, 390))
            self.photo = ImageTk.PhotoImage(preview)
            self.image_label.config(image=self.photo, text="")
            if self.model is None:
                messagebox.showwarning("Model not loaded",
                                       "Click 'Load CNN Model' first.")
                return
            self.predict(img)
        except Exception as e:
            messagebox.showerror("Image Error", str(e))

    def predict(self, img):
        try:
            from tensorflow.keras.applications.mobilenet_v2 import (
                preprocess_input, decode_predictions)
            arr = np.array(img.resize((224, 224)), dtype=np.float32)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            preds = self.model.predict(arr, verbose=0)
            decoded = decode_predictions(preds, top=10)[0]

            cat_words = ("cat", "kitten", "tabby", "tiger_cat", "siamese")
            dog_words = ("dog", "puppy", "retriever", "terrier", "shepherd",
                         "poodle", "spaniel", "husky", "mastiff", "hound")

            cat_score = sum(float(p) for _, label, p in decoded
                            if any(w in label.lower() for w in cat_words))
            dog_score = sum(float(p) for _, label, p in decoded
                            if any(w in label.lower() for w in dog_words))

            if cat_score == 0 and dog_score == 0:
                label, score = "Unknown / Not clearly cat or dog", max(float(p) for _,_,p in decoded)
            elif cat_score >= dog_score:
                label, score = "CAT", cat_score
            else:
                label, score = "DOG", dog_score

            self.prediction.config(text=f"Prediction: {label}")
            self.confidence.config(text=f"Confidence score: {score*100:.2f}%")
            self.details.config(text="Top CNN/ImageNet outputs:\n" +
                                "\n".join(f"{i+1}. {lab}: {p*100:.2f}%"
                                          for i, (_, lab, p) in enumerate(decoded[:5])))
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def show_evaluation(self):
        import os
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        except Exception:
            messagebox.showerror("Missing package",
                "Install scikit-learn: pip install scikit-learn")
            return
        if self.model is None:
            messagebox.showwarning("Model not loaded", "Load the CNN model first.")
            return

        y_true, y_pred = [], []
        for class_name, class_id in [("cat", 0), ("dog", 1)]:
            folder = os.path.join("dataset", class_name)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                    try:
                        img = Image.open(os.path.join(folder, fn)).convert("RGB")
                        arr = preprocess_input(np.expand_dims(
                            np.array(img.resize((224,224)), dtype=np.float32), axis=0))
                        from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
                        out = decode_predictions(self.model.predict(arr, verbose=0), top=10)[0]
                        cats = sum(float(p) for _, lab, p in out if "cat" in lab.lower())
                        dogs = sum(float(p) for _, lab, p in out if any(w in lab.lower()
                                  for w in ("dog","retriever","terrier","shepherd","poodle","hound")))
                        pred = 0 if cats >= dogs else 1
                        y_true.append(class_id); y_pred.append(pred)
                    except Exception:
                        pass

        if not y_true:
            self.eval_text.delete("1.0", "end")
            self.eval_text.insert("1.0", "No evaluation images found. Add images to dataset/cat and dataset/dog.")
            return

        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        text = (f"Confusion Matrix [rows=true, columns=predicted]\n"
                f"             Pred Cat   Pred Dog\n"
                f"True Cat     {cm[0,0]:8d}   {cm[0,1]:8d}\n"
                f"True Dog     {cm[1,0]:8d}   {cm[1,1]:8d}\n\n"
                f"Accuracy : {accuracy_score(y_true,y_pred):.4f}\n"
                f"Precision: {precision_score(y_true,y_pred,zero_division=0):.4f}\n"
                f"Recall   : {recall_score(y_true,y_pred,zero_division=0):.4f}\n"
                f"F1-score : {f1_score(y_true,y_pred,zero_division=0):.4f}\n")
        self.eval_text.delete("1.0", "end")
        self.eval_text.insert("1.0", text)

if __name__ == "__main__":
    root = tk.Tk()
    app = CNNClassifierApp(root)
    root.mainloop()
