import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from turtlesim.msg import Pose


class TurtleControllerNode(Node):
    def __init__(self):
        super().__init__("turtle_controller")
        self.get_logger().info("turtle_controller")
        self._robot_pose = None
        self._target_pose = None
        self._cmd_vel_publisher = self.create_publisher(
            Twist, "robot_turtle/cmd_vel", 10
        )
        self.create_subscription(
            Pose, "robot_turtle/pose", self.subscribe_robot_turtle_pose, 10
        )
        self.create_subscription(Pose, "turtle1/pose", self.subscribe_target_pose, 10)
        self.create_timer(0.01, self.control_loop)

    def subscribe_robot_turtle_pose(self, msg):
        self._robot_pose = msg

    def subscribe_target_pose(self, msg):
        self._target_pose = msg

    def control_loop(self):
        if self._robot_pose is None:
            return

        dist_x = self._target_pose.x - self._robot_pose.x
        dist_y = self._target_pose.y - self._robot_pose.y
        distance = math.sqrt(dist_x**2 + dist_y**2)

        msg = Twist()
        if distance > 1.0:
            # position
            msg.linear.x = 0.8 * distance

            # orientation
            goal_theta = math.atan2(dist_y, dist_x)
            diff = goal_theta - self._robot_pose.theta
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            msg.angular.z = 6 * diff
        else:
            # target reached
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
