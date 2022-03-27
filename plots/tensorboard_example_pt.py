import open3d as o3d
# Monkey-patch torch.utils.tensorboard.SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard import SummaryWriter

cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
cube.compute_vertex_normals()
cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0,
                                                     height=2.0,
                                                     resolution=20,
                                                     split=4)
cylinder.compute_vertex_normals()
colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

logdir = "demo_logs/pytorch/small_scale"
writer = SummaryWriter(logdir)
for step in range(3):
    cube.paint_uniform_color(colors[step])
    writer.add_3d('cube', to_dict_batch([cube]), step=step)
    cylinder.paint_uniform_color(colors[step])
    writer.add_3d('cylinder', to_dict_batch([cylinder]), step=step)