import tempfile
import unittest
from pathlib import Path
from kmeans_core import *

BASE = Path(__file__).resolve().parent


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.labels, self.points = load_points_csv(BASE / 'kmeans_data.csv')

    def test_known_example_and_prediction(self):
        c, a, h = kmeans_full(self.points, [(1, 1), (5, 7)])
        for actual, expected in zip(c, [(1.25, 1.5), (3.9, 5.1)]):
            for x, y in zip(actual, expected):
                self.assertAlmostEqual(x, y)
        self.assertEqual(a, [0, 0, 1, 1, 1, 1, 1])
        self.assertAlmostEqual(wcss(self.points, c, a), 8.525)
        self.assertEqual(len(h), 3)
        self.assertTrue(h[-1]['converged'])
        self.assertEqual(assign_clusters([(2, 2)], c)[0], [0])

    def test_all_k_fixed_points(self):
        for k in range(1, 8):
            c, a, h = kmeans_full(self.points, initial_centroids(self.points, k))
            self.assertEqual(len(c), k)
            self.assertTrue(h[-1]['converged'])
            self.assertEqual(assign_clusters(self.points, c)[0], a)
            for i, center in enumerate(c):
                members = [p for p, cluster in zip(self.points, a) if cluster == i]
                for axis in (0, 1):
                    self.assertAlmostEqual(center[axis], sum(p[axis] for p in members)/len(members))
        self.assertAlmostEqual(wcss(self.points, c, a), 0)

    def test_tie_empty_cluster_and_limit(self):
        self.assertEqual(assign_clusters([(0, 0)], [(-1, 0), (1, 0)])[0], [0])
        self.assertFalse(kmeans_full(self.points, [(1, 1), (5, 7)], max_iter=1)[2][-1]['converged'])

    def test_arbitrary_centroids_and_rejection(self):
        for seeds in [[(-100, -100), (100, 100)], [(0, 0), (0, 0)],
                      [(-1e100, -1e100), (1e100, 1e100)]]:
            c, a, h = kmeans_full(self.points, seeds)
            self.assertTrue(h[-1]['converged'])
            self.assertEqual(len(set(a)), 2)
        for seeds in [[(float('nan'), 0)], [(float('inf'), 0)], []]:
            with self.assertRaises(ValueError):
                kmeans_full(self.points, seeds)
        with self.assertRaises(ValueError):
            kmeans_full([(1, 1), (1, 1)], [(0, 0), (2, 2)])

    def test_elbow(self):
        result = elbow_method(self.points)
        self.assertAlmostEqual(result[1], 37.0714285714)
        self.assertAlmostEqual(result[2], 8.525)
        self.assertAlmostEqual(result[7], 0)
        self.assertTrue(all(result[k+1] <= result[k] for k in range(1, 7)))

    def test_invalid_csv_and_missing_label(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.csv'
            for text in ['label,x,y\n', 'x,y\nnan,1\n', 'x,y\n1,\n', 'a,b\n1,2\n', 'label,x,y\nP,1,2\nP,3,4\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_points_csv(path)
            path.write_text('x,y\n1,2\n')
            self.assertEqual(load_points_csv(path), (['P1'], [(1., 2.)]))


class GuiTests(unittest.TestCase):
    def test_complete_demo_and_state_reset(self):
        import tkinter as tk
        from unittest.mock import patch
        from kmeans_app import KMeansApp
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = KMeansApp(root)
        self.assertEqual(str(app.k_input.cget('to')), '7')
        self.assertIn('1..7', app.elbow_button.cget('text'))
        errors = []
        with patch('kmeans_app.messagebox.showerror', side_effect=lambda *args: errors.append(args)):
            for k in range(1, 8):
                app.k.set(str(k))
                app.prepare()
                app.train()
                self.assertEqual(len(app.model[0]), k)
                app.predict()
                self.assertIn('OUTPUT: Cluster', app.prediction)
            app.k.set('2')
            self.assertIsNone(app.model)
            app.prepare()
            app.train()
            app.elbow()
            app.predict()
            app.plot_canvas.draw()
            self.assertEqual(len(app.figure.axes), 2)
            self.assertEqual(len(app.ax_elbow.lines[0].get_xdata()), 7)
            self.assertEqual(app.ax.get_xlabel(), 'x')
            text = app.report_text()
            self.assertIn('ELBOW RESULTS', text)
            self.assertIn('OUTPUT: Cluster 1', text)
            with tempfile.TemporaryDirectory() as folder:
                for suffix in ('txt', 'csv'):
                    path = str(Path(folder) / ('report.' + suffix))
                    with patch('kmeans_app.filedialog.asksaveasfilename', return_value=path):
                        app.save_report()
                    self.assertIn('ELBOW RESULTS', Path(path).read_text(encoding='utf-8'))
            for entry, value in zip(app.entries, ('P8', '2', '3')):
                entry.delete(0, 'end')
                entry.insert(0, value)
            app.edit_point(False)
            self.assertEqual(len(app.coords), 8)
            self.assertEqual(str(app.k_input.cget('to')), '8')
            self.assertIn('1..8', app.elbow_button.cget('text'))
            self.assertIsNone(app.model)
            self.assertEqual(app.elbow_results, {})
            self.assertEqual(app.report_text(), '')
            app.table.selection_set('7')
            app.delete_points()
            self.assertEqual(len(app.coords), 7)
            app.train()
            app.seed_vars[0][0].set('2')
            self.assertIsNone(app.model)
            app.k.set('0')
            app.train()
            self.assertEqual(len(errors), 1)
            app.k.set('2')
            app.prepare()
            app.train()
            app.table.selection_set(app.table.get_children())
            app.delete_points()
            self.assertEqual(app.coords, [])
            self.assertIsNone(app.model)
        root.update_idletasks()


if __name__ == '__main__':
    unittest.main()
