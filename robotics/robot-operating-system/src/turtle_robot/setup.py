from setuptools import setup

package_name = "turtle_robot"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Hongbo Miao",
    maintainer_email="Hongbo.Miao@outlook.com",
    description="Turtle Robot",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "target_control_node = turtle_robot.target_control_node:main",
            "turtle_robot_control_node = turtle_robot.turtle_robot_control_node:main",
        ],
    },
)
