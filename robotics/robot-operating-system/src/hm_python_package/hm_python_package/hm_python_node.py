import rclpy
from rclpy.node import Node


class HMNode(Node):
    def __init__(self) -> None:
        super().__init__("hm_python_node")
        self._count = 0
        self.create_timer(0.5, self.timer_callback)

    def timer_callback(self) -> None:
        self._count += 1
        self.get_logger().info(f"Hello {self._count}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
