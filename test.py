import os
from collections import OrderedDict
from os.path import join

from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.dataset import TestDataset
from lib.extract_patches import *
from lib.metrics import Evaluate
from lib.visualize import save_img, concat_result


class Test:
    def __init__(self, args_test):
        self.img_path = None
        self.pred_imgs = None
        self.args = args_test
        self.pred_patches = None
        # save path
        self.path_experiment = join(args_test.outf, args_test.save)

        # 注意测试集的N_patches_tot计算规则：[(imgh_new-path_h)//stride_h]+1 的平方 * 测试图片数目
        # (N_patches_tot,1,test_patch_height,test_patch_width)
        # 原始的图片(B,3,H,W)，值在0-255之间的    mask and FOVS 的值在0-1之间的
        self.patches_imgs_test, self.test_imgs, self.test_masks, self.test_FOVs, self.new_height, self.new_width = \
            get_data_test_overlap(
                test_data_path_list=args_test.test_data_path_list,
                patch_height=args_test.test_patch_height,
                patch_width=args_test.test_patch_width,
                stride_height=args_test.stride_height,
                stride_width=args_test.stride_width
            )
        # 未加padding的图片尺寸
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]

        test_set = TestDataset(self.patches_imgs_test)
        num_workers = min([os.cpu_count(), args_test.batch_size if args_test.batch_size > 1 else 0, 8])
        self.test_loader = DataLoader(test_set, batch_size=args_test.batch_size, shuffle=False, num_workers=num_workers)

    # Inference prediction process
    def inference(self, net):
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.cuda()
                outputs = net(inputs)
                # 3D张量, 1背景的概率值组成的  (B,1,H,W)
                outputs = outputs.cpu().numpy()
                preds.append(outputs)

        # (N_patches_tot,1,H,W)  预测的概率值组成的张量
        self.pred_patches = np.concatenate(preds, axis=0)
        assert self.pred_patches.shape[1] == 1, f"inference尺寸不对"

    # Val on the test set at each epoch
    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        # recover to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        # predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks, self.test_FOVs)
        eval = Evaluate(self.path_experiment)
        eval.add_batch(y_true, y_scores)
        confusion, accuracy, specificity, sensitivity, precision = eval.confusion_matrix()
        log = OrderedDict([('test_auc_roc', eval.auc_roc()),
                           ('test_f1', eval.f1_score()),
                           ('test_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity)])
        return dict(log)

    # save segmentation imgs
    def save_segmentation_result(self):
        assert self.pred_imgs, f"先验证，再输出最后的特征图"
        img_path_list, _, _ = load_file_path_txt(self.args.test_data_path_list)
        kill_border(self.pred_imgs, self.test_FOVs)  # only for visualization
        self.img_path = os.path.join(self.path_experiment, "Results")
        if not self.img_path:
            os.makedirs(self.img_path)
        for i in range(self.test_imgs.shape[0]):
            total_img = concat_result(self.test_imgs[i], self.pred_imgs[i], self.test_masks[i])
            save_img(total_img, join(self.img_path, "Result_" + f"{i}" + '.png'))
