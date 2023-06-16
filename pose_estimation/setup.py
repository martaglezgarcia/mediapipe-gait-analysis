from setuptools import setup

package_name = 'pose_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marta',
    maintainer_email='glezgarmarta@gmail.com',
    description='Pose estimation pub/sub package',
    license='TODO: License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'talker = pose_estimation.publisher_member_function:main',
                'listener = pose_estimation.subscriber_member_function:main',
        ],
    },
)
