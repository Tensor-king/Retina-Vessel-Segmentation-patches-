import os
from os.path import join


def get_path_list(img_path, label_path, fov_path):
    tmp_list = [img_path, label_path, fov_path]
    res = []
    for i in tmp_list:
        filename_list = os.listdir(i)
        res.append([join(i, j) for j in filename_list])
    return res


def write_path_list(name_list, save_path_new2, file_name):
    f = open(join(save_path_new2, file_name), 'w')
    for i in range(len(name_list[0])):
        f.write(str(name_list[0][i]) + " " + str(name_list[1][i]) + " " + str(name_list[2][i]) + '\n')
    f.close()


if __name__ == "__main__":
    # ------------Path of the dataset --------------------------------
    data_root_path = ''
    # train
    img = "CHASEDB1/images"
    gt = "CHASEDB1/1st_label"
    fov = "CHASEDB1/mask"
    # ---------------save path-----------------------------------------
    save_path = "./data_path_list/CHASEDB1"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # -----------------------------------------------------------------
    data_list = get_path_list(img, gt, fov)
    # 数据集划分为7:3
    test_range = (0, 7)
    train_list = [data_list[i][test_range[1]:] for i in range(len(data_list))]
    test_list = [data_list[i][test_range[0]:test_range[1]] for i in range(len(data_list))]

    write_path_list(train_list, save_path, 'train.txt')

    write_path_list(test_list, save_path, 'test.txt')

    print("Finish!")
