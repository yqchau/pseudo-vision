import numpy as np


def radial_to_xy(
    ranges,
    angle_min=-3.141592741,
    angle_max=3.141592741,
    angle_inc=0.005823155,
    pps=1080,
):
    """
    convert the lidar points from radial into cartesian coordinates

    args:
        ranges   : LiDAR points from the /scan topic
        angle_min: minimum angle of the LiDAR point
        angle_max: maximum angle of the LiDAR point
        angle_inc: increment in angle across neighbouring points
        pps      : point per scan

    returns:
        xy: the points in cartesian coordinates
    """

    pt_angles = np.array([angle_min + i * angle_inc for i in range(pps)])
    assert pt_angles[-1] - angle_max < 0.0001

    pt_radial = np.array(ranges)
    x = pt_radial * np.cos(pt_angles)
    y = pt_radial * np.sin(pt_angles)

    xy = np.stack((x, y)).T
    return xy
