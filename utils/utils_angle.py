import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

est_list = []
csv_stack = []

def smoothing(y_list):
    window = 5
    w = np.ones(window) / window  # 移動平均を取るための準備
    smo_y = np.convolve(y_list, w, mode="same")
    return smo_y

def inside_hip_cul(x_a, y_a, x_b, y_b, x_c, y_c):

    a_sq = (x_b - x_c)**2 + (y_b - y_c)**2
    a = np.sqrt(a_sq)
    b_sq = (x_a - x_c)**2 + (y_a - y_c)**2
    b = np.sqrt(b_sq)
    c_sq = (x_a - x_b)**2 + (y_a - y_b)**2
    c = np.sqrt(c_sq)
    cos = (b_sq + c_sq - a_sq) / (2 * b * c)
    theta = np.arccos(cos)
    degree = np.rad2deg(theta)

    est_list = [x_a, y_a, x_b, y_b, x_c, y_c, a_sq, b_sq, c_sq, a, b, c, cos, theta, degree]

    # output(est_list)
    csvplt(est_list)

    return est_list


def hip_cul(x_a, y_a, x_b, y_b):

    x_c = x_a 
    y_c = y_b
    a = np.abs(x_a - x_b)
    a_sq = a**2
    b = np.abs(y_b - y_a)
    b_sq = b**2
    c_sq = (x_a - x_b)**2 + (y_a - y_b)**2
    c = np.sqrt(c_sq)
    cos = (a_sq + c_sq - b_sq) / (2 * a * c)
    theta = np.arccos(cos)
    degree = np.rad2deg(theta)
    if x_a <= x_b:
        degree = 180 - degree

    est_list = [x_a, y_a, x_b, y_a, x_c, y_a, a_sq, b_sq, c_sq, a, b, c, cos, theta, degree]
    csv_stack.append(est_list)
    

    return est_list, csv_stack


def csvplt(input_name, est_list):


    est_header = ['sho_x(coord)', 'sho_y(coord)', 'hip_x(coord)', 'hip_y(coord)', 'c_x(coord)', 'c_y(coord)', \
                  'a_sq(dis)', 'b_sq(dis)', 'c_sq(dis)', 'a(dis)', 'b(dis)', 'c(dis)',  \
                  'cos(rad))', 'theta(theta)', 'degree(deg)']

    save_path = './csv/' + input_name + '_plot.csv'

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(est_header)
        writer.writerows(est_list)



def output(list):
    print("x_a: %s, y_a: %s\n"%(list[0], list[1]))
    print("x_b: %s, y_b: %s\n"%(list[2], list[3]))
    print("x_c: %s, y_c: %s\n"%(list[4], list[5]))
    print("a_2: %s, a:%s\n"%(list[6], list[7]))
    print("b_2: %s, b:%s\n"%(list[8], list[9]))
    print("c_2: %s, c:%s\n"%(list[10], list[11]))
    print("cos: %s\n"%(list[12]))
    print("theta: %s\n"%(list[13]))
    print("degree: %s\n"%(list[14]))

def angleplt(input_name, x, y):
    save_path = './graph/' + input_name + '_graph.png'
    plt.figure()
    plt.title("Angle per frame")
    plt.xlabel("frame_num")
    plt.ylabel("angle")
    plt.xlim(0, len(x))
    plt.ylim(0, 180)
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(save_path)

def angleplt_smo(input_name, x, y):
    save_path = './graph/' + input_name + '_graph.png'
    plt.figure()
    plt.title("Angle per frame")
    plt.xlabel("frame_num")
    plt.ylabel("angle")
    plt.xlim(0, len(x))
    plt.ylim(0, 180)
    plt.grid(True)
    plt.plot(x, smoothing(y))
    plt.savefig(save_path)

def coordplt(x, y_1, y_2, plt_path):
    frame_per_sec = 25
    frame_num = len(x)
    sec_list = []

    for n in range(frame_num):
        fps = x[n] / frame_per_sec
        sec_list.append(round((float(fps)), 2))

    plt.title("trandition")
    plt.xlabel("time[s]")
    plt.ylabel("coordinate")
    # plt.xlim(0, 120)
    plt.ylim(0, 1920)
    plt.grid(True)
    plt.plot(sec_list, y_1, label="x_coordinate")
    plt.plot(sec_list, y_2, label="y_coordinate")
    plt.legend(loc="upper right")
    #plt.show()
    plt.savefig(plt_path)
    plt.close()


def trandition(input_name, angle_info, output_list):
    header_list = ['nose_x', 'nose_y', \
                   'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',\
                   'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y',\
                   'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y',\
                   'left_elbow_x', 'left_elbow_y', 'right_elbow_x', 'right_elbow_y',\
                   'left_wrist_x', 'left_wrist_y', 'right_wrist_x', 'right_wrist_y',\
                   'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y'\
                   'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y',\
                   'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y']
    x: float = 0.0

    save_path = input_name + '_plot.csv'

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        tmp_list = []
        for i in range(17):
            for j in range(2):
                x = angle_info[i][j]
                tmp_list.append(round(x, 2))
                j = j + 1
            i = i + 1
        writer.writerow(tmp_list)
        output_list.append(tmp_list)
