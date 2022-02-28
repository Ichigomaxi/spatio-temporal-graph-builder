# from RANSAC import ground_remove
from cProfile import label
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

    # Inputs ####################################################################
    # specify file to visualize
    str_filenumber = "n008-2018-09-18-15-26-58-0400__LIDAR_TOP__1537299185400330"
    # str_filenumber = "n008-2018-08-06-15-06-32-0400__LIDAR_TOP__1533583199699153"
    # str_filenumber = "n008-2018-08-21-11-53-44-0400__LIDAR_TOP__1534867352951893"
    # Lidar pointcloud dir
    velo_dir = "/home/max/projects/masterthesis/nuscenes_validation/LIDAR_TOP"
    # Prediction dir
    predictions_dir = "/home/max/projects/masterthesis/nuscenes_validation/inference_val_v1/data"
    # Groundtruth/ label dir
    # label_dir = r"C:\Users\maxil\Documents\projects\master_thesis\kitti_dataset\training\labels_2"
    label_dir = "/home/max/projects/masterthesis/nuscenes_validation/label"
    
    # Process Point Cloud #######################################################
    # read point cloud
    velo_filename = str_filenumber + ".pcd.bin" #"000002.bin"
    velo_path = os.path.join(velo_dir, velo_filename)
    points = np.fromfile(velo_path, dtype=np.float32)

    # Reshape either in KITTI or Nuscenes format
    
    # Uncomment for KITTI lidar data
    # points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    # points = cast_points_to_kitti(points)

    # Uncomment for Nuscenes lidar data
    points = points.reshape(-1, 5)
    # points = cast_points_to_kitti(points)
    points[:, 3] /= 255
    # points[:, 4] = 0 # Only necessary for nuscenes transformation into KITTI
    
    # Read Predictions ###################################################################
    # pointgnn nuscenes predictions/labels
    labels = []
    label_filename = str_filenumber + ".txt" #"n015-2018-07-27-11-36-48+0800__LIDAR_TOP__1532662846449656.txt"
    label_path = os.path.join(predictions_dir, label_filename)
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            if line != []:
                line = line[8:15]
                line = [float(x) for x in line]
                labels.append(line)

    # Label [1:2] = 3d box dimensions, Label[3:5] = location, label[6] = yaw
    labels = np.array(labels)

    # TODO
    # Calibrate labels from label frame to lidar frame ...(SE3-Transformation)
    for i in range(shape(labels)[0]):
        #switch 3d box dimensions(length,width,heigth) with location values(x,y,z)
        labels[i][0], labels[i][1], labels[i][2], labels[i][3], labels[i][4], labels[i][5] = \
            labels[i][3], labels[i][4], labels[i][5], labels[i][0], labels[i][1], labels[i][2]

    # Calibrate labels from label frame to lidar frame ... (SE3-Transformation)
    # lower/shift Bounding Boxes by half the boxes heigth 
    for i in range(np.shape(labels)[0]):
        labels[i][2] = labels[i][2] - labels[i][5] / 2

    # # visualization
    results = Visualizer(points, center_mode='lidar_bottom',points_size=3)

    # results.add_bboxes(labels, points_in_box_color=[0.1, 0.2, 0.9])
    results.add_bboxes(labels, bbox_color=[0, 0, 1], points_in_box_color=[0, 0, 1])

    
    # Read Groundtruth/Labels ###################################################################
    labels = []
    label_filename = str_filenumber + ".txt"  # "n015-2018-07-27-11-36-48+0800__LIDAR_TOP__1532662846449656.txt"
    label_path = os.path.join(label_dir, label_filename)
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            if line != []:
                if line[0] == 'Car':

                    line = line[1:] # Transformed Labels only have 8 values
                    line = [float(x) for x in line]
                    labels.append(line)

    # Label [1:2] = location, Label[3:5] = 3d box dimensions, label[6] = yaw
    labels = np.array(labels)
    labels_from_file = labels
    
    # Calibrate labels from label frame to lidar frame ... (SE3-Transformation)
    # lower/shift Bounding Boxes by half the boxes heigth 
    for i in range(np.shape(labels)[0]):
        z_coordinate = labels[i][2]
        z_height = labels[i][5]
        labels[i][2] = z_coordinate - z_height / 2

    results.add_bboxes(labels, bbox_color=[0.9, 0.2, 0])

    #### TEST ROTATIONS############
    # x axis Rotation
    # labels_x = labels_from_file
    # labels_x[:,0], labels_x[:,1], labels_x[:,2] = \
    #     labels_x[:,0], -labels_x[:,2], labels_x[:,1]
    # results.add_bboxes(labels_x, bbox_color=[1, 0, 0])
    # # y axis Rotation
    # labels_y = labels_from_file
    # labels_y[:,0], labels_y[:,1], labels_y[:,2] = \
    #     labels_y[:,2], labels_y[:,1], -labels_y[:,0]
    # results.add_bboxes(labels_y, bbox_color=[0, 1, 0])
    # # z axis Rotation
    # labels_z = labels_from_file
    # labels_z[:,0], labels_z[:,1], labels_z[:,2] = \
    #     -labels_z[:,1], labels_z[:,0], labels_z[:,2]
    # results.add_bboxes(labels_z, bbox_color=[0, 0, 1])
    #####################################3


    # Show Results/ Open Visualization ###################################################################

    # save_path = r"C:\Users\maxil\Documents\projects\master_thesis\kitti_dataset\cvpr_qualitative\results"
    save_path = "/home/max/projects/masterthesis/nuscenes_validation/results"
    results.show(save_path=save_path)
    # # after = Visualizer(points_after)
    # # after.add_bboxes(labels)source
    # # after.show()
