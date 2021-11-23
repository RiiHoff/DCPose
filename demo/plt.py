import csv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

est_deg = []
rea_deg = []
i = 0

est_file = open("dcpose_drgree_list.csv", "r")
rea_file = open("STS01.csv", "r")

est_f = csv.reader(est_file)
rea_f = csv.reader(rea_file)
rea_h = next(rea_f)

for est_row in est_f:
    if i < 1763:
        i = i + 1
        est_deg.append(round(float(est_row[13]), 2))
    else:
        break

for rea_row in rea_f:
    rea_deg.append(round(float(rea_row[1]), 2))

x_list = []
for i in range(1763):
    x_list.append(i)

# print(len(x_list), len(est_deg), len(rea_deg))

plt.title('compare')
plt.xlabel('frame')
plt.ylabel('angle(deg)')
plt.xlim([0.0, 1764])
plt.ylim([0.0, 180.0])
plt.grid(True)
plt.plot(x_list, est_deg, label="estimate_value")
plt.plot(x_list, rea_deg, label="real_value")
plt.legend(loc="lower right")
plt.show()