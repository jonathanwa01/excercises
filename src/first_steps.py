import ddg
import numpy as np

ddg.blender.scene.clear(deep=True)

Q = ddg.geometry.Quadric(np.diag((1, 2, 1, -1)))
ddg.blender.convert(Q, "quadric")

cam = ddg.blender.camera.camera(location=(5, 5, 8))
ddg.blender.camera.look_at_point(cam, (0, 0, 0))

ddg.blender.light.light(location=(0, 1, 8))
