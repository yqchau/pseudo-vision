#!/usr/bin/env python3

import rospy
from pseudo_vision.msg import Image
from sensor_msgs.msg import LaserScan
from utils import get_occupancy_grid, radial_to_xy


class ImagePublisher:
    def __init__(self):
        self.lidar_sub = rospy.Subscriber(
            "/scan", LaserScan, self.scan_callback, queue_size=1
        )
        self.img_pub = rospy.Publisher("/occupancy_grid", Image, queue_size=1)

        self.occ_grid_config = rospy.get_param("occ_grid")
        self.scan_to_xy_config = rospy.get_param("scan_to_xy")

    def scan_callback(self, data):
        """
        get LiDAR points from /scan, converts it into occupancy grid and publishes to /occupancy_grid
        """

        # get occpancy grid
        pts = radial_to_xy(data.ranges, **self.scan_to_xy_config)
        img = get_occupancy_grid(pts, **self.occ_grid_config)

        # publish to /occupancy_grid
        data = Image()
        data.image = img.flatten().tolist()
        self.img_pub.publish(data)
        rospy.loginfo("Image Published!")


def main():
    rospy.init_node("image_publisher")
    ImagePublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
