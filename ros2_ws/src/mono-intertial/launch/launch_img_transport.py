from launch import LaunchDescription

import launch.actions
import launch_ros.actions


def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            package='image_transport',
            executable='republish',
            name=f'im_transport',
            output='log',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed','/down/genicam_0/image/compressed'),
                ('out','/down/genicam_0/image')
            ]
        )
])