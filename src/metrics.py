"""按论文公式独立实现；不依赖作者函数，便于交叉核验。

定义域：只对实际出现的边缘组求统计量；未观察到的预设组须另做覆盖率审计。
退化列联表的关联强度未定义，返回 NaN，不把它解读为公平。
"""
import numpy as np
from scipy.special import xlogy


def entropy(p):
    p = np.asarray(p, dtype=float)
    return float(-np.sum(xlogy(p, p)))


def representation(counts):
    """counts为各组样本数；只对非零组计算指标，单组SEI与NSD返回NaN。"""
    counts = np.asarray(counts, dtype=float)
    p = counts[counts > 0] / counts.sum()
    r, h = len(p), entropy(p)
    d = float(p @ p)
    return {'R': r, 'ENS': np.exp(h), 'D': d, '1/D': 1/d, '1-D': 1-d,
            'H': h, 'SEI': h/np.log(r) if r > 1 else np.nan,
            'NSD': np.std(p)*r/np.sqrt(r-1) if r > 1 else np.nan,
            'IR': p.max()/p.min(), 'BP': p.max()}


def association(table):
    """table的行为人口组、列为标签；返回全局指标字典及同形的NPMI、Z矩阵。"""
    t = np.asarray(table, dtype=float)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    p = t / t.sum()
    px, py = p.sum(axis=1), p.sum(axis=0)
    expected = np.outer(px, py)
    delta = p - expected
    delta[np.abs(delta) < 1e-15] = 0  # 浮点误差不能被近零分母放大为完全关联。
    chi_per_n = float(np.sum(delta**2 / expected))
    hx, hy, hxy = entropy(px), entropy(py), entropy(p)
    mi = max(0., hx+hy-hxy)
    if p.shape[0] == 1: hx = 0.
    if p.shape[1] == 1: hy = 0.
    if min(p.shape) == 1: mi = 0.
    r, c = p.shape
    v = np.sqrt(chi_per_n/min(r-1, c-1)) if min(r, c) > 1 else np.nan
    tt = np.sqrt(chi_per_n/np.sqrt((r-1)*(c-1))) if min(r,c) > 1 else np.nan
    # 论文 Eq.(12) 排版为 1-χ²/n；此处遵循标准定义与作者 SciPy 实现的 1+χ²/n。
    pearson = np.sqrt(chi_per_n/(1+chi_per_n))
    with np.errstate(divide='ignore', invalid='ignore'):
        npmi = np.log(p/expected)/(-np.log(p))
    npmi[p == 0] = -1  # 零联合概率的极限；整行/列零已先剔除。
    den = np.where(delta >= 0, np.minimum(px[:,None],py[None,:])-expected,
                   expected-np.maximum(0,px[:,None]+py[None,:]-1))
    z = np.divide(delta, den, out=np.zeros_like(delta), where=den > 1e-15)
    return {'V': v, 'T': tt, 'C': pearson, 'Ux': mi/hx if hx > 0 else np.nan,
            'Uy': mi/hy if hy > 0 else np.nan, 'NMI': mi/hxy if hxy > 0 else np.nan}, npmi, z


def fixed_universe(counts, k):
    """在预设K个组上补充覆盖指标，区分缺组与已出现组内部的均匀程度。"""
    """补充观测 richness：固定预设组数 K 的覆盖率与熵归一化。"""
    m = representation(counts)
    return {'coverage': m['R']/k, 'H_over_logK': m['H']/np.log(k),
            'ENS_over_K': m['ENS']/k}


def corrected_v(t):
    """计算有限样本修正的Cramér V，用于03中的不确定性探索。"""
    """Bergsma 型有限样本校正，仅用于总体推断扩展，不覆盖论文描述量。"""
    t = np.asarray(t, float)
    t = t[t.sum(1)>0][:, t.sum(0)>0]
    n, (r,c) = t.sum(), t.shape
    if min(r,c) <= 1 or n <= 1: return np.nan
    p=t/n; e=np.outer(p.sum(1),p.sum(0))
    phi2=np.sum((p-e)**2/e)
    phi2=max(0,phi2-(c-1)*(r-1)/(n-1))
    rc=r-(r-1)**2/(n-1); cc=c-(c-1)**2/(n-1)
    return np.sqrt(phi2/min(rc-1,cc-1)) if min(rc,cc)>1 else np.nan
