import tempfile
import unittest
from pathlib import Path
from decision_tree_core import *

BASE = Path(__file__).resolve().parent


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.features, self.target = load_csv(BASE/'decision_tree_data_1000.csv')

    def test_honest_pipeline(self):
        train, test = stratified_split(self.rows, self.target)
        self.assertEqual((len(train), len(test)), (800, 200))
        depth, scores = select_depth(train, self.features, self.target)
        self.assertIn(depth, range(1, 6)); self.assertEqual(len(scores), 5)
        tree = build_tree(train, self.features, self.target, depth)
        metrics = evaluate(test, tree, self.target)
        for key in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
            self.assertGreaterEqual(metrics[key], 0); self.assertLessEqual(metrics[key], 1)
        self.assertEqual(sum(metrics[k] for k in ('tp','tn','fp','fn')), len(test))

    def test_entropy_gain_and_unseen(self):
        self.assertAlmostEqual(entropy(['Yes', 'Yes']), 0)
        self.assertAlmostEqual(entropy(['Yes', 'No']), 1)
        tree = build_tree(self.rows, self.features, self.target, 3)
        label, node, path = predict(tree, {f: 'UNSEEN' for f in self.features})
        self.assertIn(label, ('Yes', 'No')); self.assertTrue(path)

    def test_bad_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.csv'
            for text in ('A,Play\nx,Yes\n', 'A,Play\n,Yes\n' + 'x,No\n'*9,
                         'A,Play\n' + 'x,Yes\n'*10):
                path.write_text(text)
                with self.assertRaises(ValueError): load_csv(path)


class GuiTest(unittest.TestCase):
    def test_train_predict_report(self):
        import tkinter as tk
        from unittest.mock import patch
        from decision_tree_id3 import App
        root = tk.Tk(); root.withdraw(); self.addCleanup(root.destroy)
        app = App(root); app.train()
        self.assertIsNotNone(app.tree); self.assertIn('UNTOUCHED TEST', app.report)
        app.classify(); self.assertIn('OUTPUT: Play =', app.report)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/'report.txt')
            with patch('decision_tree_id3.filedialog.asksaveasfilename', return_value=path): app.save()
            self.assertIn('ROC-AUC', Path(path).read_text())


if __name__ == '__main__': unittest.main()
