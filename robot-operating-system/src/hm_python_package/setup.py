from setuptools import setup

package_name = "hm_python_package"

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
    description="HM Python Package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["hm_python_node = hm_python_package.hm_python_node:main"],
    },
)
