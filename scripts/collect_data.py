#!/usr/bin/env python3

import message_filters
import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan

from pseudo_vision.msg import TrainingData
from utils import radial_to_xy

class DataCollector:
    def __init__(self):
        self.config = rospy.get_param("data_collection")
        self.scan_to_xy_config = rospy.get_param('scan_to_xy')
        self.scan_sub = message_filters.Subscriber("/scan", LaserScan)
        self.drive_sub = message_filters.Subscriber(self.config["sub_topic"], AckermannDriveStamped)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.scan_sub, self.drive_sub], 10, 0.1
        )  # time synchronizer
        self.ts.registerCallback(self.ts_callback)
        self.train_pub = rospy.Publisher(self.config["pub_topic"], TrainingData, queue_size=1)

    def scan_callback(self, data):
        pts = radial_to_xy(data.ranges, **self.scan_to_xy_config)
        return pts

    def drive_callback(self, data):
        steering_angle = data.drive.steering_angle
        speed = data.drive.speed
        return steering_angle, speed

    def ts_callback(self, scan_data, drive_data):
        pts = self.scan_callback(scan_data)
        steering_angle, speed = self.drive_callback(drive_data)
        # rospy.loginfo(f"Steering Angle: {steering_angle}")

        x = pts[:, 0]
        y = pts[:, 1]
        data = TrainingData()
        data.x = x.tolist()
        data.y = y.tolist()
        data.steering_angle = steering_angle

        if speed < -0.1:
            self.train_pub.publish(data)
            rospy.loginfo("Training Data Published!")


def main():
    rospy.init_node("echo_topic")
    DataCollector()
    rospy.spin()


if __name__ == "__main__":
    main()