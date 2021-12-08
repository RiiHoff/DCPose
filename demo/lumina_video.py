import numpy as np
from PIL import Image
import sys
import glob
import os
import os.path as osp
import sys

sys.path.insert(0, osp.abspath('../'))

from utils.utils_lumina import *
from utils.utils_video import video2images, image2video


person_name = 'side_oka_t_'
data_path = './data/' + person_name + '/'
out_path = './diff_data/'

files = sorted(glob.glob(data_path + '*'))

toff_path = data_path + '00000000.jpg'

print(toff_path)

for file in files:
    # file_name = os.path.splitext(os.path.basename(file))[0]
    # print(file_name)
    lumina_ex(file, toff_path)
    print('saved ' + file)

image2video(out_path, person_name)