from launch import LaunchDescription

import launch.actions
import launch_ros.actions

bag_start_offset = 180.0
#bag_path = '/bags/240925_guenselsdorf/fighter4/2024_09_25-guenselsdorf/seq0/bags/all/rosbag2_2024_09_25-11_04_09'
bag_path = '/bags/2025_03_04-Steinalpl-rec/seq0/bags/all/rosbag2_2025_03_04-11_31_57/'

air_id = 5
rate = 1.0
image_topic = f'/AIT_Fighter{air_id}/down/image'

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
            cmd=['ros2', 'bag', 'play', f'{bag_path}', 
            '--start-offset',f'{bag_start_offset}', 
            '--topics',f'{image_topic}', f'{image_topic}/compressed', f'/bmi088_F{air_id}/imu',
            '--rate', f'{rate}'],
            output='log'
        )
])