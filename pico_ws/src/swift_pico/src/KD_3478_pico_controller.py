#!/usr/bin/env python3
'''
# Team ID:          3478
# Theme:            KrishiDrone
# Author List:      Rohan Arun Shenoy, Parth Aggarwal, Prathiksha Rao, Vaibhav Krishna Bansal
# Filename:         KD_3478_pico_controller.py
# Functions:        __init__, arm, disarm, whycon_callback, 
#                   linear_map, force_to_throttle_linear, controller, main
# Global variables: MIN_ROLL, MAX_ROLL, MIN_PITCH, MAX_PITCH, 
#                   MIN_THROTTLE, MAX_THROTTLE, BASE_ROLL, BASE_PITCH, BASE_THROTTLE
'''

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from swift_msgs.msg import SwiftMsgs
from error_msg.msg import Error
from control.matlab import c2d, ss
from control import dlqr

# Global Constants
MIN_ROLL = 1000
MAX_ROLL = 2000
MIN_PITCH = 1000
MAX_PITCH = 2000
MIN_THROTTLE = 1250
MAX_THROTTLE = 2000
BASE_ROLL = BASE_PITCH = BASE_THROTTLE = 1500


class Swift_Pico(Node):
    '''
    Purpose:
    ---
    ROS 2 node that implements LQR-based control for the Swift Pico drone.
    It subscribes to pose estimates from WhyCon, computes control inputs, and
    publishes RC commands to the drone.
    '''

    def __init__(self):
        '''
        Purpose:
        ---
        Initialize the controller node, set system parameters,
        design LQR controller, and configure publishers/subscribers.
        '''
        super().__init__('pico_controller')

        # Drone Parameters
        self.m = 0.152      # Drone mass (kg)
        self.g = 9.81       # Gravitational acceleration (m/s^2)
        self.sample_time = 0.01666  # Controller frequency (60 Hz)

        # Desired and current states
        self.desired_state = [-0.7, 0.0, -2.0, 0.0, 0.0, 0.0]  # Desired [x, y, z, ẋ, ẏ, ż]
        self.current_state = [0.0] * 6
        self.prev_state = [0.0] * 6
        self.first_cb = True

        # System Matrices
        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=float)

        self.B = np.array([
            [0,   0,   0],
            [0,   0,   0],
            [0,   0,   0],
            [0,  self.g,   0],    # ẍ ← -g * pitch
            [self.g, 0,   0],     # ÿ ← -g * roll
            [0,   0,  1/self.m],  # z̈ ← +T/m
        ], dtype=float)

        # LQR Weights
        self.Q = np.diag([40.0, 29.0, 66.0, 30.0, 2.0, 30.0])   # State cost
        self.R = np.diag([78.0, 40.0, 28.0])                    # Input cost

        # Discretize system and compute LQR gain
        sysc = ss(self.A, self.B, np.eye(6), np.zeros((6, 3)))
        sysd = c2d(sysc, self.sample_time, method='zoh')
        Ad, Bd = sysd.A, sysd.B
        self.K, _, _ = dlqr(Ad, Bd, self.Q, self.R)
        self.get_logger().info(f"LQR Gain Matrix K:\n{self.K}")

        # Command Message
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = BASE_ROLL
        self.cmd.rc_pitch = BASE_PITCH
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = BASE_THROTTLE

        # ROS Interfaces
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)

        # Arm and start control loop
        self.arm()
        self.create_timer(self.sample_time, self.controller)

    def arm(self):
        '''
        Purpose:
        ---
        Arm the drone by setting the AUX4 RC channel to 2000.
        '''
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    def disarm(self):
        '''
        Purpose:
        ---
        Disarm the drone by setting the AUX4 RC channel to 1000.
        '''
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)

    def whycon_callback(self, msg):
        '''
        Purpose:
        ---
        Callback for WhyCon pose updates.
        Updates position and velocity states with low-pass filtering.

        Input Arguments:
        ---
        `msg` : [ geometry_msgs.msg.PoseArray ]
            Incoming pose data from WhyCon system.

        Returns:
        ---
        None
        '''
        self.prev_state = self.current_state.copy()

        # Initialize filter state for Z if not present
        if not hasattr(self, "z_filtered"):
            self.z_filtered = 0.0  

        # Raw Z (invert and scale)
        z_measured = -(msg.poses[0].position.z / 10.0)

        # Low-pass filter for Z position
        alpha_z = 0.6  
        self.z_filtered = (1 - alpha_z) * self.z_filtered + alpha_z * z_measured

        # Update positions
        self.current_state[0] = msg.poses[0].position.x / 10.0
        self.current_state[1] = msg.poses[0].position.y / 10.0
        self.current_state[2] = self.z_filtered

        if self.first_cb:
            # First callback → Initialize velocities
            self.current_state[3:6] = [0.0, 0.0, 0.0]
            self.vel = [0.0, 0.0, 0.0]
            self.first_cb = False
        else:
            # Velocity estimation from finite difference
            measured_v = [
                (self.current_state[0] - self.prev_state[0]) / self.sample_time,
                (self.current_state[1] - self.prev_state[1]) / self.sample_time,
                (self.current_state[2] - self.prev_state[2]) / self.sample_time,
            ]
            # Low-pass filter velocities
            alpha_v = 0.2
            self.vel = [
                (1 - alpha_v) * v_prev + alpha_v * v_new
                for v_prev, v_new in zip(self.vel, measured_v)
            ]
            self.current_state[3:6] = self.vel

    def linear_map(self, x):
        '''
        Purpose:
        ---
        Map small roll/pitch angles to RC values in [1000, 2000].

        Input Arguments:
        ---
        `x` : [ float ]
            Roll or pitch angle (rad).

        Returns:
        ---
        `mapped_val` : [ float ]
            RC channel value corresponding to angle.
        '''
        max_angle = math.pi / 12
        slope = 200 / max_angle
        return slope * x + 1500

    def force_to_throttle_linear(self, thrust_force):
        '''
        Purpose:
        ---
        Convert desired thrust force into RC throttle value.

        Input Arguments:
        ---
        `thrust_force` : [ float ]
            Thrust force deviation from hover (N).

        Returns:
        ---
        `throttle` : [ int ]
            RC throttle value clipped between MIN_THROTTLE and MAX_THROTTLE.
        '''
        hover_thrust = self.m * self.g
        total_thrust = hover_thrust + thrust_force
        max_thrust = 2.0 * hover_thrust

        slope = (MAX_THROTTLE - BASE_THROTTLE) / (max_thrust - hover_thrust)
        throttle = BASE_THROTTLE + slope * (total_thrust - hover_thrust)

        # Bias adjustment
        throttle += 34

        return int(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))

    def controller(self):
        '''
        Purpose:
        ---
        Main control loop.
        Computes control inputs using LQR, maps them to RC commands,
        and publishes the drone commands and position errors.
        '''
        error = np.array(self.current_state) - np.array(self.desired_state)
        u = -self.K @ error  # Control input: [roll φ, pitch θ, thrust]

        roll  = np.clip(u[0], -math.pi/12, math.pi/12)
        pitch = np.clip(u[1], -math.pi/12, math.pi/12)
        thrust = u[2]

        # Convert control inputs to RC values
        self.cmd.rc_roll = int(self.linear_map(roll))
        self.cmd.rc_pitch = int(self.linear_map(pitch))
        self.cmd.rc_throttle = self.force_to_throttle_linear(thrust)

        # Print log if near-perfect
        margin = 0.04
        if (abs(error[0]) < margin and abs(error[1]) < margin and abs(error[2]) < margin):
            self.get_logger().info("perfect")
        else:
            self.get_logger().info(
                f"x_err={error[0]:+.2f}, y_err={error[1]:+.2f}, z_err={error[2]:+.2f}, "
                f"rc_roll={self.cmd.rc_roll}, rc_pitch={self.cmd.rc_pitch}, rc_throttle={self.cmd.rc_throttle}"
            )

        # Publish commands
        self.command_pub.publish(self.cmd)

        # Publish errors
        e = Error()
        e.roll_error = error[1]
        e.pitch_error = error[0]
        e.throttle_error = error[2]
        self.error_pub.publish(e)


def main(args=None):
    '''
    Purpose:
    ---
    Entry point for the ROS 2 node.
    Initializes ROS 2, starts the Swift_Pico node, and handles shutdown.

    Input Arguments:
    ---
    `args` : [ list ]
        Command-line arguments for ROS 2.

    Returns:
    ---
    None
    '''
    rclpy.init(args=args)
    node = Swift_Pico()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
