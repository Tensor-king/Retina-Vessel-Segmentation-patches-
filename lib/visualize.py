import numpy as np
from PIL import Image


# Prediction result splicing (original img, predicted probability, binary img, groundtruth)
def concat_result(ori_img, pred_res, gt):
    ori_img = np.transpose(ori_img, (1, 2, 0))
    pred_res = np.transpose(pred_res, (1, 2, 0))
    gt = np.transpose(gt, (1, 2, 0))

    pred_res[pred_res >= 0.5] = 1
    pred_res[pred_res < 0.5] = 0

    if ori_img.shape[2] == 3:
        pred_res = np.repeat((pred_res * 255).astype(np.uint8), repeats=3, axis=2)
        gt = np.repeat((gt * 255).astype(np.uint8), repeats=3, axis=2)
    # 把3张图片按列拼起来  分别是原来的图片、二值化后的图片、以及真实分割图片
    total_img = np.concatenate((ori_img, pred_res, pred_res, gt), axis=1)
    return total_img


# visualize image, save as PIL image
def save_img(data, filename):
    # fromarray转化灰度图片不能三个通道
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    img = Image.fromarray(data.astype(np.uint8))  # the image is between 0-1
    img.save(filename)
