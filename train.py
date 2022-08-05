import os
from os.path import join

import torch.optim.lr_scheduler
import wandb
from torch.optim import lr_scheduler

from config import parse_args
from function import train, val, get_dataloader
from lib.common import *
from models.DU_Net import DU_Net
from test import Test


def main():
    args = parse_args()

    save_path = join(args.outf, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DU_Net(in_channels=1).to(device)

    print("Total number of parameters in this net: " + str(count_parameters(net)))

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.001)

    # # The training speed of this task is fast, so pre_training is not recommended
    # if args.pre_trained:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     checkpoint = torch.load(args.pre_trained)
    #     net.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     args.start_epoch = checkpoint['epoch'] + 1
    # print(optimizer.param_groups)
    # assert args.start_epoch > 20, f"载入预训练权重出错"

    # create a list of learning rate with epochs
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0.0001)
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.N_epochs, warmup=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15, verbose=True, min_lr=0)

    # 将patches打包
    train_loader, val_loader = get_dataloader(args)  # create dataloader

    # Initialize the best epoch and performance(AUC of ROC)
    best = {'epoch': 0, 'AUC_roc': 0.5}
    trigger = 0  # Early stop Counter

    experiment = wandb.init(project='DU_Net')

    # 创建传输数据和文件的端口
    model_artifact = wandb.Artifact(
        'DU_Net', type="model",
        description="hy-parameters and weights",
        metadata=vars(args))

    test_ = Test(args)

    for epoch in range(args.start_epoch, args.N_epochs + 1):

        # train stage
        train_loss = train(train_loader, net, optimizer, device, epoch, args.N_epochs,
                           scheduler)

        experiment.log({"train_loss": train_loss, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch})

        trigger += 1

        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 "lr": optimizer.param_groups[0]['lr']}

        if args.val:
            val_log = val(val_loader, net, device, epoch=epoch, total_epoch=args.N_epochs, path=save_path)
            experiment.log(val_log)
            scheduler.step(val_log['val_auc_roc'])
            if val_log['val_auc_roc'] > best['AUC_roc']:
                torch.save(state, 'best_model.pth')
                best['epoch'] = epoch
                best['AUC_roc'] = val_log['val_auc_roc']
                trigger = 0
        else:
            test_.inference(net)
            test_log = test_.val()
            scheduler.step(test_log['test_auc_roc'])
            experiment.log(test_log)
            if test_log['test_auc_roc'] > best['AUC_roc']:
                torch.save(state, 'best_model.pth')
                best['epoch'] = epoch
                best['AUC_roc'] = test_log['test_auc_roc']
                trigger = 0

        # 结束训练，权重上传云端
        if trigger >= args.early_stop or epoch == args.N_epochs:
            model_artifact.add_file('best_model.pth')
            experiment.log_artifact(model_artifact)
            test_.save_segmentation_result()
            break


if __name__ == '__main__':
    main()
