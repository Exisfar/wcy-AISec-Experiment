"""保护关键解释：概率定义、零支持、维度退化、对称性，不镜像实现。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from metrics import representation,association,fixed_universe,corrected_v

def test_independence_is_not_marginal_balance():
    t=[[45,45],[5,5]]
    g,_,_=association(t)
    assert np.isclose(g['V'],0)
    assert representation([90,10])['ENS']<2

def test_perfect_dependence_rectangular():
    g,_,_=association([[30,30,0],[0,0,60]])
    assert np.isclose(g['V'],1)
    assert g['T']<1 and g['C']<1

def test_missing_joint_cell_has_negative_one_local_association():
    _,nm,z=association([[50,0],[0,50]])
    assert nm[0,1]==-1 and z[0,1]==-1
    assert nm[0,0]==1 and z[0,0]==1

def test_degenerate_metrics_do_not_claim_fairness():
    assert np.isnan(association([[20,80]])[0]['V'])
    assert np.isnan(representation([100])['SEI'])
    assert fixed_universe([100],2)['coverage']==.5

def test_scale_invariance_of_descriptive_metrics():
    t=np.array([[7,9,3],[2,12,15]])
    a=association(t)[0];b=association(t*100)[0]
    for k in a: assert np.isclose(a[k],b[k])

def test_theil_directions_and_transpose():
    t=np.array([[15,5,0],[0,10,30]])
    a=association(t)[0];b=association(t.T)[0]
    assert np.isclose(a['Ux'],b['Uy'])
    assert np.isclose(a['V'],b['V'])
    assert not np.isclose(a['Ux'],a['Uy'])

def test_corrected_v_does_not_inflate_finite_sample_signal():
    t=np.array([[10,9,11],[12,9,9]])
    assert 0<=corrected_v(t)<=association(t)[0]['V']

def test_floating_point_roundoff_in_single_row_is_not_dependence():
    g,_,z=association([[30,30,30,30,30,30,30]])
    assert np.isnan(g['Ux']) and g['Uy']==0
    assert np.all(z==0)
