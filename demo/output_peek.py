import glob
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

ab_list = ['ab001','ab002', 'ab003', 'ab004', 'ab005', 'ab006']
pa_list = ['pa001','pa002', 'pa003', 'pa004']
peek_list = []
x_axis = []
y_axis = []

y_ab001 = []
y_ab002 = []
y_ab003 = []
y_ab004 = []
y_ab005 = []
y_ab006 = []
y_pa001 = []
y_pa002 = []
y_pa003 = []
y_pa004 = []

for i in ab_list:
    nor_files = './peek/' + i + '_side_nr02_peek.csv'
    fas_files = './peek/' + i + '_side_fr02_peek.csv'
    save_path = './plt/ab_peeks.png'
    tmp = 'y_' + i
    df = pd.read_csv(nor_files, index_col=0)
    ang_ = df.iat[0 ,1]
    print(type(vasd))





#     for file in nor_files:
#         with open(file, 'r', newline='') as f:
#             reader = csv.reader(f)
#             header = next(reader)
#             for row in reader:
#                 x = x + 1
#                 x_axis.append(x)
#                 tmp.append(float(row[2]), 2)

# plt.title('compare')
# plt.xlabel('frame')
# plt.ylabel('angle(deg)')
# plt.xlim([0.0, 1764])
# plt.ylim([0.0, 180.0])
# plt.grid(True)
# plt.plot(x_list, est_deg, label="estimate_value")
# plt.plot(x_list, rea_deg, label="real_value")
# plt.legend(loc="lower right")
# plt.show()
