from launch import LaunchDescription

import launch.actions
import launch_ros.actions

bag_start_offset = 150.0

#bag_path = '/bags/2025_03_04-Steinalpl-rec/seq0/bags/all/rosbag2_2025_03_04-11_31_57/'
#bag_path = '/bags/2025_05_15-Calvarina-rec/seq0/bags/all/rosbag2_2025_05_15-08_25_08_circle_110m'
#bag_path = '/bags/2025_05_15-Calvarina-rec/seq0/bags/all/rosbag2_2025_05_15-09_14_04_grid_80m'
#bag_path = '/bags/2025_08_13-Streitdorf-rec/seq0/bags/all/rosbag2_2025_08_13-09_38_53_25fps'
#bag_path = '/bags/2025_08_14-Streitdorf-rec/seq0/bags/all/rosbag2_2025_08_14-08_30_32_15fps'
#bag_path = '/bags/2025_09_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_09_22-09_19_35_10hz'

#bag_path = '/bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-09_49_48_eight' # 150
#bag_path = '/bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-10_05_51_eight_long'
#bag_path = '/bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-10_53_50_ellipse_lang_gegen'
#bag_path = '/bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-11_39_27_ellipse_lang_im'
#bag_path = '/bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-12_06_43_large_circle'
#bag_path = '/bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-12_17_58_ellpise_gegen'
#bag_path = '/bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-12_25_31_ellpise_im'
#bag_path = '/bags/2025_10_15-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_15-12_35_55_eight_heigh_with_ratio'

#bag_path = '/bags/2025_10_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_22-09_20_38_eight_straight' # 100
#bag_path = '/bags/2025_10_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_22-09_36_18_straight_eight' # 100
#bag_path = '/bags/2025_10_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_22-10_12_22_eight_circle' # 100
#bag_path = '/bags/2025_10_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_22-10_21_21_circle_eight' # 100
#bag_path = '/bags/2025_10_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_22-10_34_26_eight_eight_high' # 100
#bag_path = '/bags/2025_10_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_22-10_51_54_eight_high_eight' # 100
#bag_path = '/bags/2025_10_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_22-10_59_56_eight_eight_long' # 100
#bag_path = '/bags/2025_10_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_22-11_16_33_eight_long_eight' # 100
#bag_path = '/bags/2025_10_22-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_10_22-11_33_48_straight_straight' # 100


# F4
#bag_path = '/bags/2025_06_06-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_06_06-09_32_00_12fps'
#bag_path = '/bags/2025_06_06-Guenselsdorf-rec/seq0/bags/all/rosbag2_2025_06_06-09_32_00_max_ts'

air_id = 6
rate = 1.0
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