import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue, ParameterFile
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    pkg_share = FindPackageShare("ur5_moveit_config")

    # Build robot description using the shared URDF
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([pkg_share, "config", "ur5_robot.urdf.xacro"]),
            " use_pid:=false",
            " headless:=",
            LaunchConfiguration("headless"),
            " use_mjcf_from_topic:=false",
            " use_transmissions:=false",
        ]
    )

    robot_description_str = robot_description_content.perform(context)
    robot_description = {"robot_description": ParameterValue(value=robot_description_str, value_type=str)}

    parameters_file = PathJoinSubstitution([pkg_share, "config", "ros2_controllers.yaml"])
    mujoco_plugins_file = PathJoinSubstitution([pkg_share, "config", "mujoco_ros2_control_plugins.yaml"])

    nodes = []

    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description, {"use_sim_time": True}],
        )
    )

    nodes.append(
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            emulate_tty=True,
            output="both",
            parameters=[
                {"use_sim_time": True},
                ParameterFile(parameters_file),
                ParameterFile(mujoco_plugins_file),
            ],
            remappings=(
                [("~/robot_description", "/robot_description")] if os.environ.get("ROS_DISTRO") == "humble" else []
            ),
            on_exit=Shutdown(),
        )
    )

    # Controller spawners
    controllers_to_spawn = ["joint_state_broadcaster", "ur5_arm_controller", "hand_controller"]
    for controller in controllers_to_spawn:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller, "--param-file", parameters_file],
                output="both",
            )
        )

    return nodes


def generate_launch_description():
    headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run simulation without visualization window",
    )

    return LaunchDescription(
        [
            headless,
            OpaqueFunction(function=launch_setup),
        ]
    )



