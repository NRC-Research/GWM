#%%
import sys
from pathlib import Path

import numpy as np
# import gwm.greedy_wavelet_method as wm

#%%
# RspMath09 Example
HERE = Path(__file__).parent
RSPMATCH09_V2_INPUT_DIR = HERE / r'rspmatch09_v2_data\Input Files'
trs = np.loadtxt(RSPMATCH09_V2_INPUT_DIR / 'cms_T0.2_horiz.tgt', skiprows=3, usecols=(0, 3))
seed = []   
with open(RSPMATCH09_V2_INPUT_DIR / 'set3_h1_cmsT0.2.acc', 'r') as fh:
    fh.readline()
    fh.readline()
    for line in fh:
        seed.extend(line.split())
seed = np.array([float(s) for s in seed])
#
#  %%

np.save('cms_T0.2_horiz.npy', trs)
np.save('set3_h1_cmsT0.2.npy', seed)
