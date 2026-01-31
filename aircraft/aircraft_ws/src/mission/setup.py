from setuptools import find_packages, setup

package_name = 'mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JacopoPan',
    maintainer_email='jacopo.pan@gmail.com',
    description='Mission node to trigger the action interfaces exposed by autopilot_interface',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission = mission.mission_node:main',
            'gym_control = mission.gym_control_node:main',
        ],
    },
)
