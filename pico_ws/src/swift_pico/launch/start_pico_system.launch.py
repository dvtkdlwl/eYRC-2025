#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Waypoint Service Node
        Node(
            package='swift_pico',
            executable='waypoint_service.py',
            name='waypoint_service',
            output='screen'
        ),

        # Pico Server (LQR + Action Server)
        Node(
            package='swift_pico',
            executable='pico_server.py',
            name='pico_server',
            output='screen'
        ),

        # Delay client by 5 seconds to let others start up
        TimerAction(
            period=3.5,
            actions=[
                Node(
                    package='swift_pico',
                    executable='pico_client.py',
                    name='pico_client',
                    output='screen'
                )
            ]
        )
    ])

