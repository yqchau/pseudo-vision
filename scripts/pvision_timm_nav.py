#!/usr/bin/env python3
import numpy as np
import rospy
import torch
import torch.nn as nn
from ackermann_msgs.msg import AckermannDriveStamped
from pseudo_vision.msg import Image
from models import TIMM
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from utils import lightning_to_torch, visualize_return_layers


class PseudoVisionTimmNavigation:
    def __init__(self):

        # setup
        rospy.loginfo("Seting up model..")
        self.config = rospy.get_param("pseudo_vision_timm")
        self.input_size = (self.config["input_size"], self.config["input_size"])
        self.model = TIMM(**self.config["timm"]).cuda()
        self.model.load_state_dict(lightning_to_torch(self.config["weights"]))
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)

        self.return_layers = self.config["return_layers"]
        self.visualize_return_layers = self.config["visualize_return_layers"]
        self.mid_getter = MidGetter(
            self.model, return_layers=self.return_layers, keep_output=True
        )

        # subscribers and publishers
        rospy.loginfo("Setting up subscribers & publishers")
        self.image_sub = rospy.Subscriber(
            "/occupancy_grid", Image, self.occ_callback, queue_size=1
        )
        self.drive_pub = rospy.Publisher(self.config["topic"], AckermannDriveStamped, queue_size=1)

        rospy.loginfo("done..")

    def occ_callback(self, data):
        grid = np.array(data.image).reshape(*self.input_size).astype(float)
        inputs = torch.FloatTensor(grid)[None, :][None, :].cuda()
        mid_outputs, model_output = self.mid_getter(inputs)

        if self.visualize_return_layers and len(mid_outputs) > 0:
            for key in mid_outputs.keys():
                mid_outputs[key] = mid_outputs[key].cpu()
            visualize_return_layers(inputs.cpu(), mid_outputs)
        
        outputs = self.softmax(model_output.cpu().detach())
        decision = torch.argmax(outputs).item()
        conf = torch.max(outputs)
        self.steer(decision, conf)

    def steer(self, direction, conf):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.config["speed"]

        if direction == 0:
            drive_msg.drive.steering_angle = self.config["steering_angle"]
            rospy.loginfo(f"Steering left @ {conf}")
        elif direction == 1:
            drive_msg.drive.steering_angle = -1 * self.config["steering_angle"]
            rospy.loginfo(f"Steering right @ {conf}")
        elif direction == 2:
            drive_msg.drive.steering_angle = 0
            rospy.loginfo(f"Going Straight @ {conf}")

        self.drive_pub.publish(drive_msg)


def main():
    rospy.init_node("pseudo_vision_timm_nav")
    PseudoVisionTimmNavigation()
    rospy.spin()


if __name__ == "__main__":
    main()
