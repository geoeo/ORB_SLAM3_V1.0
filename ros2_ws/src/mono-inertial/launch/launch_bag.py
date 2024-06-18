from launch import LaunchDescription

import launch.actions
import launch_ros.actions

bag_start_offset = 80.0
#bag_path = '/workspaces/bags/20240222_Günselsdorf_Eve_IMU/tegra_master_2024-02-22-09-34-02'
bag_path = '/bags/openrealm/20240229_Günselsdorf_Eve_IMU/tegra_master_2024-02-29-09-54-11'

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
                ('in/compressed','/down/genicam_0/image/compressed'),
                ('out','/down/genicam_0/image')
            ]
        ),
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'play','--start-offset',f'{bag_start_offset}', f'{bag_path}'],
            output='log'
        )
])