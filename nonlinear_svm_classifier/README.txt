NON-LINEAR SVM CLASSIFIER - RBF KERNEL (built from scratch, no sklearn)
=========================================================================

Run by double-clicking "Start Nonlinear SVM.vbs". It uses ML/.venv.
Requirements: numpy, matplotlib, tkinter.

Run:
    python nonlinear_svm_app.py

Files:
    nonlinear_svm_app.py - the GUI application (all SVM math implemented manually)
    svm_circle_data.csv  - 500 sample data points (x, y, label) loaded automatically
                            (two noisy, overlapping concentric rings - NOT
                            linearly separable; tuned so the kernel SVM
                            realistically lands around ~95% test accuracy)

What it does:
    1. Loads the 500-point 2-class dataset from svm_circle_data.csv (or
       generates a fresh one with "Generate New Data").
    2. Makes a stratified 80/20 split, then fits manual z-score scaling on
       training data only (test data remains unseen).
    3. Runs 3-fold cross validation over a grid of C x gamma to pick the best
       hyperparameters (manual k-fold split + manual grid search, no
       sklearn.model_selection / GridSearchCV).
    4. Trains the final model using the SIMPLIFIED SMO ALGORITHM (Sequential
       Minimal Optimization) on the DUAL soft-margin SVM problem with an RBF
       (Gaussian) kernel - see the equations in the comments at the top of
       nonlinear_svm_app.py and in the "Model information" panel of the app.
    5. Shows:
         - the non-linear decision boundary + margin diagram (x/y axes, both
           classes colour-coded so the separation is clear)
         - an ROC curve
         - a confusion matrix + accuracy / precision / recall / F1 / AUC
         - a step-by-step manual numeric example of the kernel evaluation and
           decision function f(x) = sum(alpha_i y_i K(x_i,x)) + b, so you can
           show your instructor exactly how the model decides.
    6. Lets you type in a new (x, y) point and see which class it falls into,
       plotted live on the diagram.
    7. "Save Report (TXT/CSV)" exports the model info + evaluation metrics +
       last classification to a .txt or .csv file you choose.

NOTE: training runs a grid search (3x3 combos x 3-fold CV = 9 SMO trainings)
so clicking "Train / Evaluate" can take around 20 seconds - this is normal.

Click "Load CSV" to use your own data (needs columns: x, y, label with
label being +1 or -1), or "Generate New Data" for a fresh random dataset
(accuracy will vary a bit each time since the data is randomly generated).
Read LAB_GUIDE_BN.txt or click Bangla Lab Guide for the full experiment and viva.
