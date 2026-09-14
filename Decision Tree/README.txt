ID3 DECISION TREE ML LAB
========================
Double-click Start Decision Tree.vbs. It uses ML/.venv when available and needs
only Python/Tkinter. Alternatively run from ML:
  .venv\Scripts\python.exe "Decision Tree\decision_tree_id3.py"

Workflow: inspect/load CSV -> Train/Evaluate -> inspect tree -> enter new values
-> predict -> save report. CSV needs categorical features and a binary Play target.

Evaluation is a stratified 80/20 holdout. max_depth 1..5 is selected with 5-fold
CV on training rows only. The final model is evaluated once on untouched test rows.
ROC-AUC uses leaf positive-class probabilities; it is not copied from accuracy.

Read LAB_GUIDE_BN.txt (also displayed in the Guide tab) for the experiment and viva.
decision_tree_id3_original.py is the previous version kept as a backup.
