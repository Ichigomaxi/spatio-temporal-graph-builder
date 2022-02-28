from RANSAC import ground_remove
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d_vis import Visualizer

if __name__ == "__main__":

    path = r'000000.bin'
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    indices, model = ground_remove(points[:, :3], distance_threshold=0.3, )
    print(indices)
    print(model)

    # point_range = range(0, points.shape[0], skip) # skip points to prevent crash
    # x_before = points[:, 0]
    # y_before = points[:, 1]
    # z_before = points[:, 2]

    points_after = np.delete(points, indices, axis=0)
    # x_after = points_after[:, 0]
    # y_after = points_after[:, 1]
    # z_after = points_after[:, 2]

    before = Visualizer(points)
    before.show()
    after = Visualizer(points_after)
    after.show()

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x_before, y_before, z_before)
    # ax.scatter(x_after, y_after, z_after)
    # plt.show()
