import math
import random

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from turtlesim.msg import Pose


class TargetControllerNode(Node):
    def __init__(self):
        super().__init__("target_controller")
        self.get_logger().info("target_controller")
        self._target_pose = None
        self._cmd_vel_publisher = self.create_publisher(Twist, "turtle1/cmd_vel", 10)
        self.create_subscription(Pose, "turtle1/pose", self.subscribe_target_pose, 10)
        self.create_timer(1.0, self.control_loop)

    def subscribe_target_pose(self, msg):
        self._target_pose = msg

    def control_loop(self):
        if self._target_pose is None:
            return

        target_x = random.uniform(0.0, 10.0)
        target_y = random.uniform(0.0, 10.0)

        dist_x = target_x - self._target_pose.x
        dist_y = target_y - self._target_pose.y
        distance = math.sqrt(dist_x**2 + dist_y**2)

        msg = Twist()

        # position
        msg.linear.x = 1.0 * distance

        # orientation
        goal_theta = math.atan2(dist_y, dist_x)
        diff = goal_theta - self._target_pose.theta
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
        msg.angular.z = 2 * diff

        self._cmd_vel_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
