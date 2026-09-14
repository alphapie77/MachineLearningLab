K-MEANS INTERACTIVE ML LAB
========================
Open Start KMeans.vbs with a double-click. Python 3 with Tkinter is required.
Matplotlib draws the graphs; clustering is still implemented from scratch.
The launcher uses ML/.venv automatically when available.
Setup from ML: python -m venv .venv
Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
Alternatively, from ML run .venv\Scripts\python.exe kmeans_clustering\kmeans_app.py.

Workflow:
1. Dataset: inspect the example, load CSV, or add/update/delete points.
2. Train & graph: choose k (1..sample count), Prepare centroids, edit x,y,
   then Run K-Means. Initial seeds use deterministic farthest-point selection.
3. Calculations: inspect every distance, assignment, mean and stop condition.
4. New input: enter x,y and Predict cluster using learned centroids.
5. Guide: read the full Bengali lab instructions and worked example.

Both the k input limit and Elbow range automatically become 1..u, where u is
the number of unique data points. Elbow does not choose k automatically.
Save report exports training, any elbow results, and the latest prediction.
Save dataset CSV exports input data separately. CSV reports are not datasets.
Changing data/settings invalidates the learned model and prediction.

Files:
kmeans_core.py           Plain Python algorithm and CSV validation
kmeans_app.py            Tkinter GUI and graphs
kmeans_data.csv          Original seven-point example
LAB_GUIDE_BN.txt         Bengali tutorial, worked example and viva preparation
Start KMeans.vbs         Windows double-click launcher
launch_kmeans.pyw        Windowed Python entry point with startup error dialog
kmeans_app_original.py  Backup of the previous application
test_kmeans.py           Algorithm and GUI regression checks

Limits: two raw numerical features, up to 100 samples, no feature scaling.
k cannot exceed the number of unique coordinates. Empty clusters are reseeded
at a farthest data point and training continues.
Elbow tries all initial sample combinations when there are <=100, otherwise
20 deterministic restarts. Best observed WCSS is not a global-optimum guarantee.
Default result: C1=(1.25,1.5), C2=(3.9,5.1), WCSS=8.525.
The log includes 2 changing passes and the third, final stability check.
