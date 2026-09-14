import tempfile
import unittest
from pathlib import Path
import numpy as np
import linear_svm_app as m

BASE = Path(__file__).resolve().parent


class CoreTests(unittest.TestCase):
    def setUp(self): self.X, self.y = m.load_csv_data(BASE/'svm_linear_data.csv')

    def test_pipeline_and_metrics(self):
        Xtr, ytr, Xte, yte = m.train_test_split(self.X, self.y)
        self.assertEqual((len(ytr), len(yte)), (400, 100))
        self.assertEqual(set(ytr), {-1, 1}); self.assertEqual(set(yte), {-1, 1})
        best, scores = m.grid_search_C(Xtr, ytr, [.01, .1, 1, 10, 100], epochs=5)
        mu, sigma = m.standardize_fit(Xtr)
        w, b = m.pegasos_train(m.standardize_apply(Xtr, mu, sigma), ytr, best, epochs=30)
        pred = m.predict(m.standardize_apply(Xte, mu, sigma), w, b)
        self.assertTrue(set(pred).issubset({-1, 1}))
        self.assertGreater(np.mean(pred == yte), .9)
        self.assertEqual(len(scores), 5)

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.csv'
            cases = ['a,b\n1,2\n', 'x,y,label\n1,2,0\n'*20,
                     'x,y,label\nnan,2,1\n' + '1,2,-1\n'*19]
            for value in cases:
                path.write_text(value)
                with self.assertRaises(ValueError): m.load_csv_data(path)


class GuiTest(unittest.TestCase):
    def test_train_and_predict(self):
        import tkinter as tk
        root = tk.Tk(); root.withdraw(); self.addCleanup(root.destroy)
        app = m.LinearSVMApp(root); app.train()
        self.assertIsNotNone(app.w); self.assertIn('Accuracy', app.report)
        app.ex.insert(0, '2'); app.ey.insert(0, '2'); app.classify_point()
        self.assertIn('NEW POINT', app.report)
        self.assertTrue(np.isfinite(app.w).all())


if __name__ == '__main__': unittest.main()
