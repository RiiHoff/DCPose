import numpy as np
import pandas as pd

import os
import os.path as osp
import sys
sys.path.insert(0, osp.abspath('../'))
from utils.utils_calculation import rounding, round_dp2


def angle_peek(file_name, angle_list, fps):
    save_path = "./peek/" + file_name + "_peek.csv"

    ang_max = np.nanmax(angle_list)
    ang_min = np.nanmin(angle_list)
    idx_max = np.nanargmax(angle_list)
    idx_min = np.nanargmin(angle_list)
    sec_max = round(float(idx_max / fps), 2)
    sec_min = round(float(idx_min / fps), 2)
    ang_dif = round(float(abs(ang_max - ang_min)), 2)
    idx_dif = abs(idx_max - idx_min)
    sec_dif = round(float(abs(sec_max - sec_min)), 2)

    header=['ang_max', 'ang_min', 'ind_max', 'ind_min', 'sec_max', 'sec_min', 'ang_dif', 'ind_dif', 'sec_dif']
    peek_data = [ang_max, ang_min, idx_max, idx_min, sec_max, sec_min, ang_dif, idx_dif, sec_dif]
    

    df = pd.DataFrame([peek_data], columns=header)
    df.to_csv(save_path,index=False)



    # print("---* " + str(file_name) + " *---")
    print("maximum angle : " + str(ang_max) + " (degree)")
    print("minimum angle : " + str(ang_min) + " (degree)")
    print("maximum index : " + str(idx_max) + " (frame number)")
    print("minimum index : " + str(idx_min) + " (frame number)")
    print("maximum second: " + str(sec_max) + " (frame number)")
    print("minimum second: " + str(sec_min) + " (frame number)")
    print("difference(angle) : " + str(ang_dif) + " (degree)")
    print("difference(index) : " + str(idx_dif) + " (frame namber)")
    print("difference(second): " + str(idx_dif) + " (second)")