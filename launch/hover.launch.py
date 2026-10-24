
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Where the drone should hold, world frame [x, y, z] in metres
HOVER_POSITION = [1.0, 0.5, 1.0]

# Seconds to wait before starting the controller
CONTROLLER_DELAY = 10.0


def generate_launch_description():
    name = 'quadrotor'

    default_model = PathJoinSubstitution([
        get_package_share_directory('quadrotor_simulator_mujoco'),
        'model', 'mujoco', 'drone.xml'
    ])

    args = [
        DeclareLaunchArgument('model_path', default_value=default_model),
        DeclareLaunchArgument('flag_build', default_value='True'),
    ]

    hover_config = os.path.join(
        get_package_share_directory('dq_nmpc'),
        'config', 'mujoco', 'default', 'hover_config.yaml'
    )

    sim = Node(
        package='quadrotor_simulator_mujoco',
        executable='quadrotor_simulator',
        name='quadrrotor_simulator',
        namespace=name,
        output='screen',
        arguments=[LaunchConfiguration('model_path')],
        parameters=[{
            'rate_odom': 200.0,
            'rate_imu': 500.0,
            'world_frame_id': 'world',
            'body_frame_id': name,
        }],
    )

    hover = Node(
        package='dq_nmpc',
        executable='hover_nmpc',
        name='hover_nmpc',
        namespace=name,
        output='screen',
        parameters=[{
            'config_path': hover_config,
            'flag_build': LaunchConfiguration('flag_build'),
            'hover_position': HOVER_POSITION,
        }],
    )

    delayed_hover = TimerAction(
        period=CONTROLLER_DELAY,
        actions=[hover],
    )

    return LaunchDescription(args + [sim, delayed_hover])
