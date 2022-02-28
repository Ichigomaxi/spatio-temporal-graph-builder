# from RANSAC import ground_remove
import numpy as np
from numpy.core.fromnumeric import shape
from open3d_vis import Visualizer
import os

def cast_points_to_kitti(points):
    """
    cast points label to kitti format
    points: [-1, n]
    """
    points_xyz = points[:, :3]
    # cast points_xyz to kitti format
    # cast
    points_xyz = points_xyz[:, [0, 2, 1]] # lhw
    points_xyz[:, 1] = -points_xyz[:, 1]
    points[:, :3] = points_xyz
    return points

if __name__ == "__main__":
    # all_path = "/home/zhang/Documents/cvpr/baseline/data"
    # files = os.listdir(all_path)
    # for file in files:
    #     str_filenumber = file.split('.')[0]

    str_filenumber = "007350"
    # read point cloud
    velo_dir = r"C:\Users\maxil\Documents\projects\master_thesis\kitti_dataset\training\velodyne"
    velo_filename = str_filenumber + ".bin" #"000002.bin"
    velo_path = os.path.join(velo_dir, velo_filename)
    points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    # points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    # points = cast_points_to_kitti(points)
    # points[:, 3] /= 255
    # points[:, 4] = 0
    # print(points)
    labels = []

    # read labels
    # baseline labels
    label_dir = r"C:\Users\maxil\Documents\projects\master_thesis\kitti_dataset\cvpr_qualitative\baseline\data"
    # our labels (flow + objdet3D )
    #label_dir = "/home/zhang/Documents/cvpr/ours/data"

    label_filename = str_filenumber + ".txt"#"000002.txt"
    label_path = os.path.join(label_dir, label_filename)
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            if line != []:
                line = line[8:15]
                line = [float(x) for x in line]
                labels.append(line)

    # # calibration(from camera to velodyne)
    # print(labels)
    labels = np.array(labels)
    for i in range(shape(labels)[0]):
        # labels[i][2] = labels[i][2] - 0.8
        x_c = labels[i][3]
        y_c = labels[i][4]
        z_c = labels[i][5]
        x_v = z_c + 0.27
        y_v = -x_c
        z_v = -y_c - 0.08
        labels[i][3] = x_v
        labels[i][4] = y_v
        labels[i][5] = z_v
        labels[i][0], labels[i][1], labels[i][2], labels[i][3], labels[i][4], labels[i][5] = \
            labels[i][3], labels[i][4], labels[i][5], labels[i][1], labels[i][2], labels[i][0]

    # for label in labels:
    #     for i in range(shape(label)[0]):
    #         label[i][2] = label[i][2] - 0.8
    #         x_c = label[i][3]
    #         y_c = label[i][4]
    #         z_c = label[i][5]
    #         x_v = z_c + 0.27
    #         y_v = -x_c
    #         z_v = -y_c - 0.08
    #         label[i][3] = x_v
    #         label[i][4] = y_v
    #         label[i][5] = z_v
    #         label[i][0], label[i][1], label[i][2], label[i][3], label[i][4], label[i][5] = \
    #             label[i][3], label[i][4], label[i][5], label[i][1], label[i][2], label[i][0]

    # for i in range(np.shape(labels)[0]):
    #     labels[i][2] = labels[i][2] - labels[i][5] / 2

    # # use RANSAC
    # indices, model = ground_remove(points[:, :3], distance_threshold=0.3, )
    # print(indices)
    # print(model)
    # points_after = np.delete(points, indices, axis=0)
    #
    # # visualization
    results = Visualizer(points, center_mode='lidar_bottom',points_size=3)

    results.add_bboxes(labels, points_in_box_color=[0.1, 0.2, 0.9])

    labels = []
    label_dir = r"C:\Users\maxil\Documents\projects\master_thesis\kitti_dataset\cvpr_qualitative\ours\data"
    # our labels (flow + objdet3D )
    # label_dir = "/home/zhang/Documents/cvpr/ours/data"

    label_filename = str_filenumber + ".txt"  # "000002.txt"
    label_path = os.path.join(label_dir, label_filename)
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            if line != []:
                line = line[8:15]
                line = [float(x) for x in line]
                labels.append(line)

    # # calibration(from camera to velodyne)
    # print(labels)
    labels = np.array(labels)
    for i in range(shape(labels)[0]):
        # labels[i][2] = labels[i][2] - 0.8
        x_c = labels[i][3]
        y_c = labels[i][4]
        z_c = labels[i][5]
        x_v = z_c + 0.27
        y_v = -x_c
        z_v = -y_c - 0.08
        labels[i][3] = x_v
        labels[i][4] = y_v
        labels[i][5] = z_v
        labels[i][0], labels[i][1], labels[i][2], labels[i][3], labels[i][4], labels[i][5] = \
            labels[i][3], labels[i][4], labels[i][5], labels[i][1], labels[i][2], labels[i][0]

    results.add_bboxes(labels, bbox_color=[0.1, 0.2, 0.9])

    labels = []
    label_dir = r"C:\Users\maxil\Documents\projects\master_thesis\kitti_dataset\training\labels_2"
    # our labels (flow + objdet3D )
    # label_dir = "/home/zhang/Documents/cvpr/ours/data"

    label_filename = str_filenumber + ".txt"  # "000002.txt"
    label_path = os.path.join(label_dir, label_filename)
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            if line != []:
                if line[0] == 'Car':

                    line = line[8:15]
                    line = [float(x) for x in line]
                    labels.append(line)

    # # calibration(from camera to velodyne)
    # print(labels)
    labels = np.array(labels)
    for i in range(shape(labels)[0]):
        # labels[i][2] = labels[i][2] - 0.8
        x_c = labels[i][3]
        y_c = labels[i][4]
        z_c = labels[i][5]
        x_v = z_c + 0.27
        y_v = -x_c
        z_v = -y_c - 0.08
        labels[i][3] = x_v
        labels[i][4] = y_v
        labels[i][5] = z_v
        labels[i][0], labels[i][1], labels[i][2], labels[i][3], labels[i][4], labels[i][5] = \
            labels[i][3], labels[i][4], labels[i][5], labels[i][1], labels[i][2], labels[i][0]
    results.add_bboxes(labels, bbox_color=[0.9, 0.2, 0])
    save_path = r"C:\Users\maxil\Documents\projects\master_thesis\kitti_dataset\cvpr_qualitative\results"
    results.show(save_path=save_path)
    # # after = Visualizer(points_after)
    # # after.add_bboxes(labels)source
    # # after.show()
