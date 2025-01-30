from launch import LaunchDescription

import launch.actions
import launch_ros.actions

bag_start_offset = 200.0
bag_path = '/bags/rosbag2_2024_09_25-11_04_09_uncompressed_fighter_4/rosbag2_2024_09_26-06_55_33'

image_topic = '/AIT_Fighter4/down/image'

def generate_launch_description():
    return LaunchDescription([
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'run', 'mono-inertial', 'mono_inertial_node', '/workspaces/ORB_SLAM3_V1.0/Vocabulary/ORBvoc.txt', 'false'],
            output='screen'
        ),
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'play', f'{bag_path}', '--start-offset',f'{bag_start_offset}', '--topics', f'{image_topic}', '/bmi088_F4/imu'],
            output='log'
        )
])