from setuptools import setup

package_name = "hm_turtle"

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
    description="HM Turtle",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["turtle_controller = hm_turtle.turtle_controller:main"],
    },
)
