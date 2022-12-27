import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from turtlesim.msg import Pose


class TurtleRobotControllerNode(Node):
    def __init__(self):
        super().__init__("turtle_robot_controller")
        self.get_logger().info("turtle_robot_controller")
        self._turtle_robot_pose = None
        self._target_pose = None
        self._cmd_vel_publisher = self.create_publisher(
            Twist, "turtle_robot/cmd_vel", 10
        )
        self.create_subscription(
            Pose, "turtle_robot/pose", self.subscribe_turtle_robot_pose, 10
        )
        self.create_subscription(Pose, "turtle1/pose", self.subscribe_target_pose, 10)
        self.create_timer(0.01, self.control_loop)

    def subscribe_turtle_robot_pose(self, msg):
        self._turtle_robot_pose = msg

    def subscribe_target_pose(self, msg):
        self._target_pose = msg

    def control_loop(self):
        if self._turtle_robot_pose is None:
            return

        dist_x = self._target_pose.x - self._turtle_robot_pose.x
        dist_y = self._target_pose.y - self._turtle_robot_pose.y
        distance = math.sqrt(dist_x**2 + dist_y**2)

        msg = Twist()
        if distance > 1.0:
            # position
            msg.linear.x = 2.0 * distance

            # orientation
            goal_theta = math.atan2(dist_y, dist_x)
            diff = goal_theta - self._turtle_robot_pose.theta
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
    node = TurtleRobotControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
