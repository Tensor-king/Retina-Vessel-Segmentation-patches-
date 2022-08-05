import os
from collections import OrderedDict
from os.path import join

import numpy as np
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.common import *
from lib.datasetV2 import data_preprocess, create_patch_idx, TrainDatasetV2
from lib.metrics import Evaluate
from lib.visualize import save_img


def get_dataloader(args):
    """
    该函数加载数据集所有图像到内存，并创建训练样本提取位置的索引，所以占用内存量较少.
    """
    # 将数据进行预处理：灰度化、CLAHE、标准化、Gamma adjust
    # 返回值在0-1范围内的图片 形状为(B,1,H,W)的numpy数组
    imgs_train, masks_train, fovs_train = data_preprocess(data_path_list=args.train_data_path_list)

    # (N-patches,3) 的 numpy数组 [n, x_center , y_center]
    patches_idx = create_patch_idx(fovs_train, args)

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    val_loader = None

    # 分训练集和验证集  np.vsplit按行分
    if args.val:
        train_idx, val_idx = np.vsplit(patches_idx, (int(np.floor((1 - args.val_ratio) * patches_idx.shape[0])),))
        # 验证模式不需要进行图片增强
        val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, val_idx, mode="val", args=args)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=num_workers)

        # Save some samples of feeding to the neural network
        # 处理或后的图片与原图进行对比
        if args.sample_visualization:
            visual_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, val_idx, mode="val", args=args)
            visual_loader = DataLoader(visual_set, batch_size=1, shuffle=True, num_workers=num_workers)
            N_sample = 100
            visual_imgs = torch.zeros((N_sample, 1, args.train_patch_height, args.train_patch_width))
            visual_masks = torch.zeros((N_sample, 1, args.train_patch_height, args.train_patch_width))

            for i, (img, mask) in tqdm(enumerate(visual_loader)):
                visual_imgs[i] = torch.squeeze(img, dim=0)
                visual_masks[i] = mask
                if i >= N_sample:
                    break
            save_img(torchvision.utils.make_grid(visual_imgs * 255, nrow=10, padding=0).numpy().transpose(1, 2, 0),
                     join(args.outf, args.save, "sample_val_img.png"))
            save_img(torchvision.utils.make_grid(visual_masks * 255, nrow=10, padding=0).numpy().transpose(1, 2, 0),
                     join(args.outf, args.save, "sample_val_mask.png"))
    else:
        train_idx = patches_idx

    train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, train_idx, mode="train", args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers)

    return train_loader, val_loader


def train(train_loader, net, optimizer, device, epoch, total_epoch, lr_scheduler):
    net.train()
    train_loss = 0
    train_loader = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        train_loader.set_description(f"training Epoch:{epoch}/{total_epoch}")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_loader)


def val(val_loader, net, device, epoch, total_epoch, path):
    net.eval()
    val_loss = 0
    evaluater = Evaluate(path)
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        for inputs, targets in val_loader:
            val_loader.set_description(f"valuing Epoch:{epoch}/{total_epoch}")
            inputs, targets = inputs.to(device), targets.to(device)
            # (B,1,H,W)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            evaluater.add_batch(targets, outputs)

    log = OrderedDict([('val_loss', val_loss / len(val_loader)),
                       ('val_acc', evaluater.confusion_matrix()[1]),
                       ('val_sp', evaluater.confusion_matrix()[2]),
                       ('val_se', evaluater.confusion_matrix()[3]),
                       ('val_f1', evaluater.f1_score()),
                       ('val_auc_roc', evaluater.auc_roc()),
                       ("epoch", epoch)])
    return dict(log)
