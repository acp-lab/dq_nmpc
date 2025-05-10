from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'dq_nmpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config', 'mujoco', 'default'),
            glob('config/mujoco/default/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fer',
    maintainer_email='fernandorecalde@uti.edu.ec',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "dq_nmpc = dq_nmpc.main_dq_nmpc:main",
            "planner = dq_nmpc.main_planner:main"
        ],
    },
)
