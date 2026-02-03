from setuptools import setup

package_name = 'zmq_bridge_py'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='dev@example.com',
    description='Python ZMQ Bridge for Gymnasium integration',
    license='MIT',
    entry_points={
        'console_scripts': [
            'zmq_bridge = zmq_bridge_py.zmq_bridge_node:main',
        ],
    },
)
