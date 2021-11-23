import cv2
import glob

files = glob.glob('./sts_data/movies' + '/*.mp4')

for file_name in files:
    cap = cv2.VideoCapture(file_name)
    print(cap.get(cv2.CAP_PROP_FPS))