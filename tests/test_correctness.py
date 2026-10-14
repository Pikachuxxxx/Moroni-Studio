import unittest
import numpy as np
from src import moroni as M


class Correctness(unittest.TestCase):
    def test_parameter_updates_and_harmonics(self):
        f=M.make_cells(); x,y=M.grid(100,60,f.bbox); renderer=M.BinnedRenderer(f)
        for k in range(1,7):
            for beta in (0,.1,1.5):
                p=dict(alpha=.1,beta=beta,gamma=.45,kappa=k,lam=4)
                self.assertTrue(np.array_equal(renderer.render(x,y,**p),M.render_reference(f,x,y,**p)))

    def test_moved_grid_and_exterior(self):
        f=M.make_cells(rows=3,cols=4); r=M.BinnedRenderer(f)
        x,y=M.grid(50,30,f.bbox)
        for offset in (0,-7,20):
            for analytic in (False,True):
                self.assertTrue(np.array_equal(r.render(x+offset,y-offset,analytic=analytic),
                    M.render_reference(f,x+offset,y-offset,analytic=analytic)))

    def test_degenerate_weights_ties_and_single_cell(self):
        for rows,cols in ((1,1),(1,5),(5,1),(3,4)):
            f=M.make_cells(rows=rows,cols=cols);x,y=M.grid(35,21,f.bbox)
            self.assertTrue(np.all(M.render_binned(f,x,y,alpha=0,beta=0,gamma=0)==0))
            self.assertTrue(np.array_equal(M.render_binned(f,x,y,beta=0,gamma=0),M.voronoi_kdtree(f,x,y)))

    def test_lut_inside_bound_and_exact_outside(self):
        f=M.make_cells(rows=2,cols=3,jitter=.6,sides=(3,9),lut_size=36)
        rng=np.random.default_rng(99)
        for c in range(f.n):
            r=f.lut_radius[c]; x=rng.uniform(-3*r,3*r,5000);y=rng.uniform(-3*r,3*r,5000)
            poly=f.polys[c]-f.centers[c]
            true=M._signed_distance(x,y,poly); got=M._sample_lut(x,y,f.luts[c],r,poly)
            outside=(abs(x)>r)|(abs(y)>r)
            self.assertTrue(np.array_equal(got[outside],true[outside]))
            self.assertLessEqual(float(abs(got-true).max()),f.lut_error[c]+1e-12)

    def test_generator_simple_convex_and_contains_center(self):
        for n in range(3,13):
            f=M.make_cells(rows=2,cols=3,sides=n,jitter=10)
            for c,p in zip(f.centers,f.polys):
                edges=np.roll(p,-1,axis=0)-p; nxt=np.roll(edges,-1,axis=0)
                self.assertTrue(np.all(edges[:,0]*nxt[:,1]-edges[:,1]*nxt[:,0]>0))
                self.assertLess(float(M._signed_distance(c[0],c[1],p)),0)

    def test_invalid_parameters(self):
        f=M.make_cells(rows=1,cols=1);x,y=M.grid(3,3,f.bbox)
        for p in (dict(alpha=-1),dict(kappa=1.5),dict(gamma=np.nan),dict(lam=-1)):
            with self.assertRaises(ValueError):M.render_binned(f,x,y,**p)

    def test_empty_cell_not_omitted_and_margin_nested(self):
        f=M.make_cells(rows=2,cols=3);x,y=M.grid(60,40,f.bbox)
        self.assertEqual(M.shape_fidelity(np.full(x.shape,-1),f,x,y),0)
        raw=M.render_binned(f,x,y)
        a=M.apply_margin(raw,f,x,y,-.1);b=M.apply_margin(raw,f,x,y,.1)
        self.assertTrue(np.all((a<0)|(b==a)))

    def test_analytical_bound(self):
        f=M.make_cells(rows=3,cols=3,sides=3,jitter=.6)
        x,y=M.grid(60,40,(-10,10,-10,10))
        for c in range(f.n):
            d=np.hypot(x-f.centers[c,0],y-f.centers[c,1])
            self.assertTrue(np.all(M._sdf(f,c,x,y,True)>=d-f.radii[c]-1e-12))


if __name__=='__main__':unittest.main()
