import numpy as np
import pandas as pd


def angle_peek(file_name, angle_list):

    ang_max = np.nanmax(angle_list)
    ang_min = np.nanmin(angle_list)
    ind_max = np.nanargmax(angle_list)
    ind_min = np.nanargmin(angle_list)

    ang_dif = abs(ang_max - ang_min)
    ind_dif = abs(ind_max - ind_min)

    cul_data = {'header':['ang_max', 'ang_min', 'ind_max', 'ind_min', 'ang_dif', 'ind_dif'],
                'data': [ang_max, ang_min, ind_max, ind_min, ang_dif, ind_dif]
               }

    save_path = "./peek/" + file_name + "_peek.csv"

    df = pd.DataFrame(cul_data)
    df.to_csv(save_path)



    print("---* " + str(file_name) + " *---")
    print("maximum angle: " + str(ang_max) + " (degree)")
    print("minimum angle: " + str(ang_min) + " (degree)")
    print("maximum index: " + str(ind_max) + " (frame number)")
    print("minimum index: " + str(ind_min) + " (frame number)")
    print("difference(angle): " + str(ang_dif) + " (degree)")
    print("difference(index): " + str(ind_dif) + " (frame namber)")