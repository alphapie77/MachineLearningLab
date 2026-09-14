LINEAR SVM CLASSIFIER (built from scratch, no sklearn)
========================================================

Run by double-clicking "Start Linear SVM.vbs". It uses ML/.venv.
Requirements: numpy, matplotlib, tkinter.

Run:
    python linear_svm_app.py

Files:
    linear_svm_app.py   - the GUI application (all SVM math implemented manually)
    svm_linear_data.csv - 500 sample data points (x, y, label) loaded automatically
                          (two well-separated, non-overlapping clusters -
                          one per class - so the hyperplane/margin diagram
                          is easy to read; accuracy is typically ~95-100%)

What it does:
    1. Loads the 500-point 2-class dataset from svm_linear_data.csv (or
       generates a fresh one with "Generate New Data").
    2. Makes a stratified 80/20 split, then fits manual z-score scaling on
       training data only (test data remains unseen).
    3. Runs 5-fold cross validation over C = [0.01, 0.1, 1, 10, 100] to pick
       the best regularisation hyperparameter (manual k-fold split, no
       sklearn.model_selection).
    4. Trains the final model using the PEGASOS algorithm (stochastic
       sub-gradient descent on the primal SVM objective) - see the equations
       in the comments at the top of linear_svm_app.py and in the
       "Model information" panel of the app.
    5. Shows:
         - the maximum-margin hyperplane + margin diagram (x/y axes, both
           classes colour-coded so the separation is clear)
         - an ROC curve
         - a confusion matrix + accuracy / precision / recall / F1 / AUC
         - a step-by-step manual numeric example of how a prediction is
           computed (w.x + b), so you can show your instructor exactly how
           the model decides.
    6. Lets you type in a new (x, y) point and see which class it falls into,
       plotted live on the diagram.
    7. "Save Report (TXT/CSV)" exports the model info + evaluation metrics +
       last classification to a .txt or .csv file you choose.

Click "Load CSV" to use your own data (needs columns: x, y, label with
label being +1 or -1), or "Generate New Data" for a fresh random dataset
(accuracy will vary a bit each time since the data is randomly generated).
Read LAB_GUIDE_BN.txt or click Bangla Lab Guide for the full experiment and viva.
