import ddg

ddg.blender.scene.clear(deep=True)

# szene setup
cam = ddg.blender.camera.camera(location=(10, 10, 10))
ddg.blender.camera.look_at_point(cam, (2.5, 2.5, 0))
ddg.blender.light.light(location=(0, 1, 8))


def lines_vert(n: int):
    for i in range(n):
        line = ddg.geometry.subspace_from_affine_points([i / 4, 0, 1],
                                                        [i / 4, 1, 1])
        ddg.blender.convert(line, f"Line{i}_vert")


def lines_hor(n: int):
    for i in range(n):
        line = ddg.geometry.subspace_from_affine_points([0, i / 4, 1],
                                                        [1, i / 4, 1])
        ddg.blender.convert(line, f"Line{i}_hor")


lines_vert(20)
lines_hor(20)


ddg.blender.render.render_frame(0, full_path="./grid.png", camera=cam)
