import numpy as np


def get_occupancy_grid(pts, resolution=8, map_size=64, centered=False, marker=False):
    """
    convert the points from cartesian coordinates to an occupancy grid

    args:
        pts       : LiDAR points in cartesian coordinates
        resolution: resolution of the occupancy grid, larger means lesser points in the map
        map_size  : used to construct an array of size (map_size, map_size)
        centered  : set the position of the ego vehicle in the center of the map, otherwise, at position (3/4, 1/2).
        marker    : mark the position of the ego vehicle in the occupancy grid

    returns:

        grid: occupancy grid of size (map_size, map_size)
    """

    grid = np.zeros((map_size, map_size))
    pts = (pts * resolution).astype(int)

    u = map_size // 2 - pts[:, 0] if centered else map_size * 3 // 4 - pts[:, 0]
    v = map_size // 2 - pts[:, 1]
    uv = np.stack((u, v), axis=0).T

    uv = uv[(u > 0) * (v > 0) * (u < map_size) * (v < map_size)]
    grid[uv[:, 0], uv[:, 1]] = 1

    if marker:
        grid[map_size // 2 if centered else map_size * 3 // 4, map_size // 2] = 1

    return grid.astype(int)
