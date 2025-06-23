from launch import LaunchDescription
from launch_ros.actions import Node
def generate_launch_description():
    return LaunchDescription([
        Node(
            package='analyze_dashboard_pkg',
            executable='image_analyzer_node',
            name='image_analyzer_node',
            output='screen'
        )
    ])