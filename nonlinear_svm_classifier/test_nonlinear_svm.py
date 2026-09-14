import tempfile, unittest
from pathlib import Path
import numpy as np
import nonlinear_svm_app as m

BASE=Path(__file__).resolve().parent

class CoreTests(unittest.TestCase):
    def setUp(self): self.X,self.y=m.load_csv_data(BASE/'svm_circle_data.csv')

    def test_pipeline_boundaries_and_finite_output(self):
        Xr,yr,Xt,yt=m.train_test_split(self.X,self.y)
        self.assertEqual((len(yr),len(yt)),(400,100)); self.assertEqual(set(yr),{-1,1})
        mu,sigma=m.standardize_fit(Xr); A=m.standardize_apply(Xr,mu,sigma); B=m.standardize_apply(Xt,mu,sigma)
        alpha,b=m.smo_train(A,yr,C=1,gamma=.5,max_passes=2)
        scores=m.decision_function_kernel(A,yr,alpha,b,B,.5)
        self.assertTrue(np.isfinite(alpha).all()); self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(set(np.where(scores>=0,1,-1)).issubset({-1,1}))

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.csv'
            for value in ('a,b\n1,2\n','x,y,label\n1,2,0\n'*30,'x,y,label\nnan,2,1\n'+'1,2,-1\n'*29):
                path.write_text(value,encoding='utf-8')
                with self.assertRaises(ValueError): m.load_csv_data(path)

    def test_kernel_extreme_queries(self):
        X=np.array([[0.,0.],[1.,1.]]); y=np.array([-1,1]); alpha=np.array([.5,.5])
        scores=m.decision_function_kernel(X,y,alpha,0,np.array([[0,0],[1e100,-1e100]]),.5)
        self.assertTrue(np.isfinite(scores).all())

if __name__=='__main__': unittest.main()
