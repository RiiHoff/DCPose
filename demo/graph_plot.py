import sys
import matplotlib.pyplot as plt
import csv
import glob

args = sys.argv
x: int = 0
x_axis = []
y_axis = []

data_path = './csv/' + args[1] + '.csv'
save_path = './graph/graph_' + args[1] + '.png'

with open(data_path, 'r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        x = x + 1
        x_axis.append(x)
        y_axis.append(round(float(row[14]), 2))


plt.title("Angle per frame")
plt.xlabel("frame_num")
plt.ylabel("angle")
plt.xlim(0, len(x_axis))
plt.ylim(0,180)
plt.grid(True)
plt.plot(x_axis, y_axis)
plt.savefig(save_path)