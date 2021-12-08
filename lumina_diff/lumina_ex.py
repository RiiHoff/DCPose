import numpy as np
from PIL import Image
import sys

# 画像の読み込み
image1 = Image.open("./diff_sample/ichimura_turnoff.jpg")
image2 = Image.open("./diff_sample/ichimura_turnon.jpg")

# RGB画像に変換
image1 = image1.convert("RGB")
image2 = image2.convert("RGB")

# NumPy配列へ変換
im1_u8 = np.array(image1)
im2_u8 = np.array(image2)

# サイズや色数が違うならエラー
if im1_u8.shape != im2_u8.shape:
    print("サイズが違います")
    sys.exit()

# 負の値も扱えるようにnp.int16に変換
im1_i16 = im1_u8.astype(np.int16)
im2_i16 = im2_u8.astype(np.int16)

# 差分配列作成
diff_i16 = im1_i16 - im2_i16

'''ここから作成する画像によって異なる処理'''

diff_bool = np.abs(diff_i16) > 30
diff_bin_u8 = diff_bool.astype(np.uint8)
diff_u8 = diff_bin_u8 * 255

# PNG 画像として保存
diff_img = Image.fromarray(diff_u8)

'''ここまで作成する画像によって異なる処理'''

# 画像表示
diff_img.save("result.jpg")