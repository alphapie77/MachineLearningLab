import math, tempfile, unittest
from pathlib import Path
from knn_core import load_csv, stratified_split, train_evaluate, classify

BASE=Path(__file__).resolve().parent

class KNNCoreTests(unittest.TestCase):
    def test_pipeline_is_stratified_and_finite(self):
        rows=load_csv(BASE/'shirt_size_data_1000.csv'); result=train_evaluate(rows)
        self.assertEqual(len(result['train_raw']),800); self.assertEqual(len(result['test_raw']),200)
        self.assertEqual({x:sum(r[2]==x for r in result['test_raw']) for x in result['labels']},{'L':33,'M':167})
        self.assertIn(result['best_k'],(3,5,7,9)); self.assertTrue(all(math.isfinite(v) for v in result['cv'].values()))

    def test_neighbour_details_and_tie(self):
        train=[(0,0,'B'),(0,1,'A'),(9,9,'A')]
        label,near,votes=classify(train,(0,.4),2,True)
        self.assertEqual(label,'B'); self.assertEqual(len(near),2); self.assertEqual(votes,{'A':1,'B':1})

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.csv'; p.write_text('height_cm,weight_kg,shirt_size\nnan,60,M\n',encoding='utf-8')
            with self.assertRaises(ValueError): load_csv(p)

if __name__=='__main__': unittest.main()
