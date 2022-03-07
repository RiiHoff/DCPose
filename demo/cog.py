import os
import os.path as osp
from pickle import FALSE
import sys

sys.path.insert(0, osp.abspath('../'))
from utils.utils_angle import hip_cul, csvplt, csv_angleplt, csv_cogplt, output, angleplt, angleplt_smo, angleplt_cog, coordplt, stack_coords, trandition
from utils.utils_cog import cog_cul, cog_plt 


import numpy as np
import glob
import pandas as pd


files = glob.glob('./results_cog_csv/*.csv')
for f in files:
    x = []
    df = pd.read_csv(f, header=None)
    print(df[1])

    for i in df.index:
        x.append(i)

    print(x)

    angleplt_cog(f.split('/')[2], x, df[0], df[1], 60)