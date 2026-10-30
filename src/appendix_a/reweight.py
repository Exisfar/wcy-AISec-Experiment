"""保留类边缘的限权重方案；仅在每个群体×类单元有支持时定义。"""
import numpy as np

def clipped_cell_weights(counts,cap=5.):
    n=np.asarray(counts,dtype=float)
    if n.ndim!=2 or (n<=0).any():raise ValueError('Every group x class cell must have positive support')
    if cap<=0:raise ValueError('cap must be positive')
    target=np.broadcast_to(n.sum(0)[None,:]/n.shape[0],n.shape)
    weights=np.minimum(target/n,cap)
    weighted=n*weights
    weighted*=n.sum(0)[None,:]/weighted.sum(0)[None,:]
    weights=weighted/n
    ess=weighted.sum()**2/np.sum(n*weights*weights)
    return weights,float(ess/n.sum())
