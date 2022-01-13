import cv2
import os
import os.path as osp
import sys
sys.path.insert(0, osp.abspath('../'))
from utils.utils_calculation import rounding, round_dp2

def comp(l, r):
    a = l if r > l else r
    b = l if l > r else r
    return a, b
def cog_cul(keypoint_coords):
    #重心計算のパラメータ (横井,1993,バイオメカニズム学会誌)
    P_HEAD = 0.069
    P_TRUNK = 0.489
    P_U_ARM = 0.027
    P_F_ARM = 0.016
    P_HAND = 0.006
    P_THIGH = 0.11
    P_SHANK = 0.051
    P_FOOT = 0.011

    #座標取得
    nose = keypoint_coords[0]
    leye = keypoint_coords[1]
    reye = keypoint_coords[2]
    lear = keypoint_coords[3]
    rear = keypoint_coords[4]
    lsho = keypoint_coords[5]
    rsho = keypoint_coords[6]
    lelb = keypoint_coords[7]
    relb = keypoint_coords[8]
    lwri = keypoint_coords[9]
    rwri = keypoint_coords[10]
    lhip = keypoint_coords[11]
    rhip = keypoint_coords[12]
    lkne = keypoint_coords[13]
    rkne = keypoint_coords[14]
    lank = keypoint_coords[15]
    rank = keypoint_coords[16]
    #重心計算
    a_sho_x, b_sho_x = comp(lsho[0], rsho[0])
    a_sho_y, b_sho_y = comp(lsho[1], rsho[1])
    a_hip_x, b_hip_x = comp(lhip[0], rhip[0])
    a_hip_y, b_hip_y = comp(lhip[1], rhip[1])
    a_ruarm_x, b_ruarm_x = comp(rsho[0], relb[0])
    a_ruarm_y, b_ruarm_y = comp(rsho[1], relb[1])
    a_luarm_x, b_luarm_x = comp(lsho[0], lelb[0])
    a_luarm_y, b_luarm_y = comp(lsho[1], lelb[1])
    a_rfarm_x, b_rfarm_x = comp(relb[0], rwri[0])
    a_rfarm_y, b_rfarm_y = comp(relb[1], rwri[1])
    a_lfarm_x, b_lfarm_x = comp(lelb[0], lwri[0])
    a_lfarm_y, b_lfarm_y = comp(lelb[1], lwri[1])
    a_rthigh_x, b_rthigh_x = comp(rhip[0], rkne[0])
    a_rthigh_y, b_rthigh_y = comp(rhip[1], rkne[1])
    a_lthigh_x, b_lthigh_x = comp(lhip[0], lkne[0])
    a_lthigh_y, b_lthigh_y = comp(lhip[1], lkne[1])
    a_rshank_x, b_rshank_x = comp(rkne[0], rank[0])
    a_rshank_y, b_rshank_y = comp(rkne[1], rank[1])
    a_lshank_x, b_lshank_x = comp(lkne[0], lank[0])
    a_lshank_y, b_lshank_y = comp(lkne[1], lank[1])

    Ax = (a_sho_x - b_sho_x)*0.5 + b_sho_x
    Ay = (a_sho_y - b_sho_y)*0.5 + b_sho_y
    Bx = (a_hip_x - b_hip_x)*0.5 + b_hip_x
    By = (a_hip_y - b_hip_y)*0.5 + b_hip_y
    trunkx = (Ax - Bx)*0.5 + Bx
    trunky = (Ay - By)*0.5 + By
    ruarmx = (a_ruarm_x - b_ruarm_x)*0.5 + b_ruarm_x
    ruarmy = (a_ruarm_y - b_ruarm_y)*0.5 + b_ruarm_y
    luarmx = (a_luarm_x - b_luarm_x)*0.5 + b_luarm_x
    luarmy = (a_luarm_y - b_luarm_y)*0.5 + b_luarm_y
    rfarmx = (a_rfarm_x - b_rfarm_x)*0.5 + b_rfarm_x
    rfarmy = (a_rfarm_y - b_rfarm_y)*0.5 + b_rfarm_y
    lfarmx = (a_lfarm_x - b_lfarm_x)*0.5 + b_lfarm_x
    lfarmy = (a_lfarm_y - b_lfarm_y)*0.5 + b_lfarm_y
    rthighx = (a_rthigh_x - b_rthigh_x)*0.5 + b_rthigh_x
    rthighy = (a_rthigh_y - b_rthigh_y)*0.5 + b_rthigh_y
    lthighx = (a_lthigh_x - b_lthigh_x)*0.5 + b_lthigh_x
    lthighy = (a_lthigh_y - b_lthigh_y)*0.5 + b_lthigh_y
    rshankx = (a_rshank_x - b_rshank_x)*0.5 + b_rshank_x
    rshanky = (a_rshank_y - b_rshank_y)*0.5 + b_rshank_y
    lshankx = (a_lshank_x - b_lshank_x)*0.5 + b_lshank_x
    lshanky = (a_lshank_y - b_lshank_y)*0.5 + b_lshank_y
    x_cog = (P_TRUNK + P_HEAD)*trunkx + P_U_ARM*ruarmx + P_U_ARM*luarmx + (P_F_ARM + P_HAND)*rfarmx\
            + (P_F_ARM + P_HAND)*lfarmx + P_THIGH*rthighx + P_THIGH*lthighx + (P_SHANK + P_FOOT)*rshankx\
            + (P_SHANK + P_FOOT)*lshankx
    y_cog = (P_TRUNK + P_HEAD)*trunky + P_U_ARM*ruarmy + P_U_ARM*luarmy + (P_F_ARM + P_HAND)*rfarmy\
            + (P_F_ARM + P_HAND)*lfarmy + P_THIGH*rthighy + P_THIGH*lthighy + (P_SHANK + P_FOOT)*rshanky\
            + (P_SHANK + P_FOOT)*lshanky

    cog = [x_cog, y_cog]
    return cog


def cog_plt(base_img, cog):
    cv2.circle(base_img, (int(round(cog[0])), int(round(cog[1]))), 7, (153, 0, 255), thickness=-1)