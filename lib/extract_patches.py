"""
This part mainly contains functions related to extracting image patches.
The image patches are randomly extracted in the fov(optional) during the training phase, 
and the test phase needs to be spliced after splitting
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lib.pre_processing import my_PreProc


# =================Load imgs from disk with txt files=====================================
# Load data path index file
# 返回三个路径列表 [path1,path2,....]
def load_file_path_txt(file_path):
    img_list = []
    gt_list = []
    fov_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # read a line
            if not lines:
                break
            img, gt, fov = lines.split(' ')
            img_list.append(img)
            gt_list.append(gt)
            fov_list.append(fov)
    return img_list, gt_list, fov_list


# Load the original image, grroundtruth and FOV of the data set in order, and check the dimensions
def load_data(data_path_list_file):
    img_list, gt_list, fov_list = load_file_path_txt(data_path_list_file)
    imgs = None
    groundTruth = None
    FOVs = None
    for i in range(len(img_list)):
        img = Image.open(img_list[i])
        img = np.array(img)
        gt = Image.open(gt_list[i]).convert("L")
        gt = np.array(gt)
        fov = Image.open(fov_list[i]).convert("L")
        fov = np.array(fov)
        # 在batch维度进行拼接 imgs 最后变为 (B,H,W,3)  GT和FOV没有通道C 所以最终变为(B,H,W)
        imgs = np.expand_dims(img, 0) if imgs is None else np.concatenate(
            (imgs, np.expand_dims(img, 0)))
        groundTruth = np.expand_dims(gt, 0) if groundTruth is None else np.concatenate(
            (groundTruth, np.expand_dims(gt, 0)))
        FOVs = np.expand_dims(fov, 0) if FOVs is None else np.concatenate((FOVs, np.expand_dims(fov, 0)))

    # FOV中心为255 其余为0
    assert (np.min(FOVs) == 0 and np.max(FOVs) == 255)
    # CHASE_DB1数据集GT图像为单通道二值（0和1）图，不过前面已经转成灰度图(0和255)
    assert ((np.min(groundTruth) == 0 and (
            np.max(groundTruth) == 255))), f"labels的最大值不为255"

    # Convert the dimension of imgs to [N,C,H,W]
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    groundTruth = np.expand_dims(groundTruth, 1)
    FOVs = np.expand_dims(FOVs, 1)
    return imgs, groundTruth, FOVs


def is_patch_inside_FOV(x, y, fov_img, patch_h, patch_w, mode='center'):
    """
    check if the patch is contained in the FOV,
    The center mode checks whether the center pixel of the patch is within fov, 
    the all mode checks whether all pixels of the patch are within fov.
    """
    if mode == 'center':
        return fov_img[y, x]
    elif mode == 'all':
        fov_patch = fov_img[y - int(patch_h / 2):y + int(patch_h / 2), x - int(patch_w / 2):x + int(patch_w / 2)]
        # 有一个像素为0，则返回False
        return fov_patch.all()
    else:
        raise ValueError("判断像素在FOV内是否存在")


# =============================Load test data==========================================
def get_data_test_overlap(test_data_path_list, patch_height, patch_width, stride_height, stride_width):
    test_imgs_original, test_masks, test_FOVs = load_data(test_data_path_list)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255.
    test_FOVs = test_FOVs // 255

    # 改变imgs的尺寸使得符合预期
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    # N_patches_tot, 1, patch_h, patch_w
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    return patches_imgs_test, test_imgs_original, test_masks, test_FOVs, test_imgs.shape[2], test_imgs.shape[3]


# extend both images and masks they can be divided exactly by the patches dimensions
def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    img_h = full_imgs.shape[2]  # height of the image
    img_w = full_imgs.shape[3]  # width of the image
    leftover_h = (img_h - patch_h) % stride_h  # leftover on the h dim
    leftover_w = (img_w - patch_w) % stride_w  # leftover on the w dim
    full_imgs = torch.from_numpy(full_imgs).float()
    full_imgs = F.pad(full_imgs, (0, stride_w - leftover_w, 0, stride_h - leftover_h)).numpy()
    assert full_imgs.shape[1] == 1, f"pad出错了"
    return full_imgs


# Extract test image patches in order and overlap
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)
    N_patches_img = ((img_h - patch_h) // stride_h + 1) * (
            (img_w - patch_w) // stride_w + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]
    patches = np.zeros((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches  # array with all the full_imgs divided in patches


# recompone the prediction result patches to images
def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    assert (preds.shape[0] % N_patches_img == 0)
    # 查看一共有多少张图片
    N_full_imgs = preds.shape[0] // N_patches_img
    full_prob = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += preds[
                    k]  # Accumulate predicted values
                # 有的区域的像素经过对此叠加，我们需要除以重叠数，得到真正1像素值
                full_sum[i, :, h * stride_h:(h * stride_h) + patch_h,
                w * stride_w:(w * stride_w) + patch_w] += 1  # Accumulate the number of predictions
                k += 1
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)
    final_avg = full_prob / full_sum  # Take the average
    return final_avg


# return only the predicted pixels contained in the FOV, for both images and masks
def pred_only_in_FOV(data_imgs, data_masks, FOVs):
    assert (len(data_imgs.shape) == 4 and len(data_masks.shape) == 4)  # 4D arrays
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  # loop over the all test images
        for x in range(width):
            for y in range(height):
                if pixel_inside_FOV(i, x, y, FOVs):
                    new_pred_imgs.append(data_imgs[i, :, y, x])
                    new_pred_masks.append(data_masks[i, :, y, x])
    new_pred_imgs = np.array(new_pred_imgs)
    new_pred_masks = np.array(new_pred_masks)
    return new_pred_imgs, new_pred_masks


# Set the pixel value outside FOV to 0, only for visualization
def kill_border(data, FOVs):
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):  # loop over the full images
        for x in range(width):
            for y in range(height):
                if not pixel_inside_FOV(i, x, y, FOVs):
                    data[i, :, y, x] = 0.0


# function to judge pixel(x,y) in FOV or not
def pixel_inside_FOV(i, x, y, FOVs):
    assert (len(FOVs.shape) == 4)  # 4D arrays
    assert (FOVs.shape[1] == 1)
    if x >= FOVs.shape[3] or y >= FOVs.shape[2]:  # Pixel position is out of range
        return False
    return FOVs[i, 0, y, x] > 0  # 0==black pixels
