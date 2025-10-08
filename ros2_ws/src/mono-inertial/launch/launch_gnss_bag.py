from launch import LaunchDescription

import launch.actions
import launch_ros.actions

bag_start_offset = 0.0
#bag_path = '/bags/240925_guenselsdorf/fighter4/2024_09_25-guenselsdorf/seq0/bags/all/rosbag2_2024_09_25-11_04_09'
#bag_path = '/bags/2025_03_04-Steinalpl-rec/seq0/bags/all/rosbag2_2025_03_04-11_31_57/'
#bag_path = '/bags/2025_05_15-Calvarina-rec/seq0/bags/all/rosbag2_2025_05_15-08_25_08_circle_110m'
#bag_path = '/bags/2025_05_15-Calvarina-rec/seq0/bags/all/rosbag2_2025_05_15-09_14_04_grid_80m'
#bag_path = '/bags/2025_08_13-Streitdorf-rec/seq0/bags/all/rosbag2_2025_08_13-09_38_53_25fps'
#bag_path = '/bags/2025_08_14-Streitdorf-rec/seq0/bags/all/rosbag2_2025_08_14-08_30_32_15fps'
bag_path = '/bags/2025_09_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_09_22-09_19_35_10hz'

air_id = 6
rate = 0.1
image_topic = f'/AIT_Fighter{air_id}/down/image'
gnss_topic = f'/AIT_Fighter{air_id}/mavros/global_position/global'
read_ahead_queue_size = 5000

def generate_launch_description():
    return LaunchDescription([
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'run', 'mono-inertial', 'mono_inertial_gnss_node', '/workspaces/ORB_SLAM3_V1.0/Vocabulary/ORBvoc.txt', 'false'],
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
            '--topics', f'{image_topic}/compressed', f'/bmi088_F{air_id}/imu', gnss_topic,
            '--read-ahead-queue-size',
            f'{read_ahead_queue_size}',
            '--rate', f'{rate}'],
            output='log'
        )
])