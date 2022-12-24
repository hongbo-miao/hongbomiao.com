import rclpy
from rclpy.node import Node


class MyNode(Node):
    def __init__(self):
        super().__init__("py_test")
        self._count = 0
        self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        self._count += 1
        self.get_logger().info(f"Hello {self._count}")


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
