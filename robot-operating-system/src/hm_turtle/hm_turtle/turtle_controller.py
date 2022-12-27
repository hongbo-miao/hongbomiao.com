import math
import random

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from turtlesim.msg import Pose


class TurtleControllerNode(Node):
    def __init__(self):
        super().__init__("turtle_controller")
        self.target_x = random.uniform(0.0, 10.0)
        self.target_y = random.uniform(0.0, 10.0)

        self._pose = None
        self._cmd_vel_publisher = self.create_publisher(Twist, "turtle1/cmd_vel", 10)
        self.create_subscription(Pose, "turtle1/pose", self.subscribe_turtle_pose, 10)
        self.create_timer(0.01, self.control_loop)

    def subscribe_turtle_pose(self, msg):
        self._pose = msg

    def control_loop(self):
        if self._pose is None:
            return

        dist_x = self.target_x - self._pose.x
        dist_y = self.target_y - self._pose.y
        distance = math.sqrt(dist_x**2 + dist_y**2)

        msg = Twist()

        if distance > 0.5:
            # position
            msg.linear.x = 2 * distance

            # orientation
            goal_theta = math.atan2(dist_y, dist_x)
            diff = goal_theta - self._pose.theta
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi

            msg.angular.z = 6 * diff
        else:
            # target reached!
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self._cmd_vel_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TurtleControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
