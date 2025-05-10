import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    launch_args = [
        DeclareLaunchArgument('name', default_value='quadrotor'),
        DeclareLaunchArgument('platform_type', default_value='mujoco'),
        DeclareLaunchArgument('world_frame_id', default_value='world'),
        DeclareLaunchArgument('rate_odom', default_value='200.0'),
        DeclareLaunchArgument('rate_imu', default_value='500.0'),
        DeclareLaunchArgument('flag_build', default_value='True'),
    ]

    # Get values from arguments
    name = LaunchConfiguration('name')
    platform_type = LaunchConfiguration('platform_type')
    world_frame_id = LaunchConfiguration('world_frame_id')
    rate_odom = LaunchConfiguration('rate_odom')
    rate_imu = LaunchConfiguration('rate_imu')
    flag_build = LaunchConfiguration('flag_build')

    # Construct the path to the XML model using substitutions
    model_path = PathJoinSubstitution([
        get_package_share_directory('quadrotor_simulator_mujoco'),
        'model',
        platform_type,
        'drone.xml'
    ])

    # Add default for model_path
    launch_args.append(
        DeclareLaunchArgument('model_path', default_value=model_path)
    )
    
    # Path config file
    control_config =  PathJoinSubstitution([
        get_package_share_directory('dq_nmpc'), TextSubstitution(text='config'), 
        platform_type, TextSubstitution(text='default'), TextSubstitution(text='dq_control.yaml')]
    )

    # Create the simulator node
    quadrotor_simulator_mujoco_node = Node(
        package='quadrotor_simulator_mujoco',
        executable='quadrotor_simulator',
        name='quadrrotor_simulator',
        namespace=name,
        output='screen',
        arguments=[LaunchConfiguration('model_path')],
        parameters=[{
        'rate_odom': rate_odom,
        'rate_imu': rate_imu,
         'world_frame_id': world_frame_id,
         'body_frame_id': name
    }]
    )

    # Dq_controller
    nmpc_controller_node = Node(
        package='dq_nmpc',
        executable='dq_nmpc',
        name='dq_controller',
        namespace=name,
        output='screen',
        parameters=[
        control_config, 
        {
            'world_frame_id': world_frame_id,
            'body_frame_id': name,
            'flag_build': flag_build
        }
    ]
    )

    # Planner
    planner_node = Node(
        package='dq_nmpc',
        executable='planner',
        name='planner',
        namespace=name,
        output='screen',
        parameters=[
        control_config, 
        {
            'world_frame_id': world_frame_id,
            'body_frame_id': name
        }
    ]
    )

     # Optional debug print to screen
    debug_print = LogInfo(msg=['[INFO] Using model path: ', LaunchConfiguration('model_path')])

    # Build launch description
    ld = LaunchDescription(launch_args)
    ld.add_action(debug_print)
    ld.add_action(
        GroupAction(actions=[
            quadrotor_simulator_mujoco_node, nmpc_controller_node, planner_node
        ])
    )
    return ld