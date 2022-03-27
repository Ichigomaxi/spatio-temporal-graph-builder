import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
import tensorflow as tf
# ... geometry creation code as above ...
logdir = "demo_logs/tf/small_scale"
writer = tf.summary.create_file_writer(logdir)
with writer.as_default():
    for step in range(3):
        cube.paint_uniform_color(colors[step])
        summary.add_3d('cube', to_dict_batch([cube]), step=step, logdir=logdir)
        cylinder.paint_uniform_color(colors[step])
        summary.add_3d('cylinder', to_dict_batch([cylinder]), step=step,
                       logdir=logdir)