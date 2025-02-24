"""
M1 and M2 macrophages ODE models for their ACTIVITY (not proliferation),
We'll model activity based on some calibrated baseline (no BCC), i
influence from other cells (BCC or MX),and a natural limit for activity K.


Define activity as the potential to kill/integrate BCC through a secreted factor
Factors M1: TNF-a, IL-12 (high,T-cell activation), IL-1B
Factors M2: IL-10, TGF-β, VEGF, MMP-9
"""

#M1: we'll assume M1 activity to be related to TNF-a concentration, first linearly, later M.M.?
