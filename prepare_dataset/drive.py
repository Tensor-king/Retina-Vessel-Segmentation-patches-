import os
from os.path import join


def get_path_list(img_path, label_path, fov_path):
    tmp_list = [img_path, label_path, fov_path]
    res = []
    for i in tmp_list:
        filename_list = os.listdir(i)
        res.append([join(i, filename) for filename in filename_list])
    return res


def write_path_list(name_list, save_path_new, file_name):
    f = open(join(save_path_new, file_name), 'w')
    for i in range(len(name_list[0])):
        f.write(str(name_list[0][i]) + " " + str(name_list[1][i]) + " " + str(name_list[2][i]) + '\n')
    f.close()


if __name__ == "__main__":
    # 注意给images,mask,fov路径的分割符是空格
    # ------------Path of the dataset -------------------------
    # train
    img_train = "../input/driveorigin/DRIVE/training/images/"
    gt_train = "../input/driveorigin/DRIVE/training/labels/"
    fov_train = "../input/driveorigin/DRIVE/training/mask/"
    # test
    img_test = "../input/driveorigin/DRIVE/test/images/"
    gt_test = "../input/driveorigin/DRIVE/test/labels/"
    fov_test = "../input/driveorigin/DRIVE/test/mask/"
    # ----------------------------------------------------------
    save_path = "../input/driveorigin/DRIVE"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # [[img1,...],[label1,...],[mask1,...]]
    train_list = get_path_list(img_train, gt_train, fov_train)
    write_path_list(train_list, save_path, 'train.txt')

    # [[img1,...],[1st_manual,...],[label,...]]
    test_list = get_path_list(img_test, gt_test, fov_test)
    write_path_list(test_list, save_path, 'test.txt')
    print("finished!")
