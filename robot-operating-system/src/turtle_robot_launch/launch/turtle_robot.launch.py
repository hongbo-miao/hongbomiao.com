from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import FindExecutable
from launch_ros.actions import Node


def generate_launch_description():
    hm_namespace = "hm"
    turtlesim_node = Node(
        package="turtlesim",
        namespace=hm_namespace,
        executable="turtlesim_node",
        name="sim",
    )
    spawn_turtle = ExecuteProcess(
        cmd=[
            [
                FindExecutable(name="ros2"),
                " service call ",
                hm_namespace,
                "/spawn ",
                "turtlesim/srv/Spawn ",
                "\"{x: 2.0, y: 2.0, theta: 0.0, name: 'turtle_robot'}\"",
            ]
        ],
        shell=True,
    )
    return LaunchDescription(
        [
            turtlesim_node,
            RegisterEventHandler(
                OnProcessStart(
                    target_action=turtlesim_node,
                    on_start=[
                        LogInfo(msg="Turtlesim started, spawning turtle"),
                        spawn_turtle,
                    ],
                )
            ),
            Node(
                package="turtle_robot",
                namespace=hm_namespace,
                executable="target_controller",
                name="target_controller",
            ),
            Node(
                package="turtle_robot",
                namespace=hm_namespace,
                executable="turtle_robot_controller",
                name="turtle_robot_controller",
            ),
        ]
    )
