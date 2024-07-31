from launch import LaunchDescription

import launch.actions
import launch_ros.actions

bag_start_offset = 200.0
#bag_path = '/workspaces/bags/20240222_Günselsdorf_Eve_IMU/tegra_master_2024-02-22-09-34-02'
#bag_path = '/bags/openrealm/20240229_Günselsdorf_Eve_IMU/tegra_master_2024-02-29-09-54-11'
bag_path = '/bags/openrealm/2024_07_25-Guenselsdorf/flight1/bags/all/rosbag2_2024_07_25-09_27_00'

image_topic = '/AIT_Fighter5/down/image'

def generate_launch_description():
    return LaunchDescription([
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'run', 'mono-inertial', 'mono_inertial_node', '/workspaces/ORB_SLAM3_V1.0/Vocabulary/ORBvoc.txt', 'false'],
            output='screen'
        ),
        launch_ros.actions.Node(
            package='image_transport',
            executable='republish',
            name='im_transport',
            output='log',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed',f'{image_topic}/compressed'),
                ('out',f'{image_topic}')
            ]
        ),
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'play', f'{bag_path}', '--start-offset',f'{bag_start_offset}', '--topics', f'{image_topic}/compressed', '/bmi088/imu',
],
            output='log'
        )
])