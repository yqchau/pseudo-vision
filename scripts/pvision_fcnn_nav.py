#!/usr/bin/env python
import numpy as np
import rospy
import torch
import torch.nn as nn
from ackermann_msgs.msg import AckermannDriveStamped
from ml_nav.msg import Image
from models import FCNN
from utils import lightning_to_torch


class PseudoVisionFcnnNavigation:
    def __init__(self):

        # setup
        rospy.loginfo("Seting up model..")
        self.config = rospy.get_param("pseudo_vision_fcnn")
        self.model = FCNN(
            self.config["input_size"] ** 2,
            self.config["classes"],
            self.config["n_layers"],
        )
        self.model.load_state_dict(lightning_to_torch(self.config["weights"]))
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)

        # subscribers and publishers
        rospy.loginfo("Setting up subscribers & publishers")
        self.image_sub = rospy.Subscriber(
            "/occupancy_grid", Image, self.occ_callback, queue_size=1
        )
        self.drive_pub = rospy.Publisher("/nav", AckermannDriveStamped, queue_size=1)

        rospy.loginfo("done..")

    def occ_callback(self, data):
        grid = np.array(data.image).astype(float)
        inputs = torch.FloatTensor(grid)[None, :]
        outputs = self.softmax(self.model(inputs).detach())
        decision = torch.argmax(outputs).item()
        conf = torch.max(outputs)
        self.steer(decision, conf)

    def steer(self, direction, conf):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.config["speed"]

        if direction == 0:
            drive_msg.drive.steering_angle = 0.3
            rospy.loginfo(f"Steering left @ {conf}")
        elif direction == 1:
            drive_msg.drive.steering_angle = -0.3
            rospy.loginfo(f"Steering right @ {conf}")
        elif direction == 2:
            drive_msg.drive.steering_angle = 0
            rospy.loginfo(f"Going Straight @ {conf}")

        self.drive_pub.publish(drive_msg)


def main():
    rospy.init_node("pseudo_vision_fcnn_nav")
    PseudoVisionFcnnNavigation()
    rospy.spin()


if __name__ == "__main__":
    main()
