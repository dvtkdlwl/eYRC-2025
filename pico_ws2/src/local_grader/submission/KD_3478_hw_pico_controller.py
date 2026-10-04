#!/usr/bin/env python3

import time
import math
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from rc_msgs.msg import RCMessage
from rc_msgs.srv import CommandBool
from error_msg.msg import Error


BATTERY_VOLTAGE = 4.0

# =========================
# SAFE HARDWARE LIMITS
# =========================
MIN_ROLL = 1400
BASE_ROLL = 1500
MAX_ROLL = 1600

MIN_PITCH = 1400
BASE_PITCH = 1450
MAX_PITCH = 1600

MIN_THROTTLE = 1300
BASE_THROTTLE = 1450 # your measured hover
MAX_THROTTLE = 1550


X_TOL = 0.08 
Y_TOL = 0.08 
Z_TOL = 0.08   # ±8 cm tolerance

class Swift_Pico(Node):

    def __init__(self):
        super().__init__('pico_pid_controller')

        # =========================
        # Timing
        # =========================
        self.dt = 0.033   # ~30 Hz

        # =========================
        # States
        # =========================
        self.current_state = [0.0, 0.0, 0.0]   # x, y, z
        self.desired_state = [0.0, -0.55, 1.9]

        # =========================
        # PID gains
        # X → pitch, Y → roll, Z → throttle
        # =========================
        self.Kp = [15, 27, 35]
        self.Ki = [0.2, 10, 8]   # small I on x/y
        self.Kd = [1, 5, 5]

        self.integral = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]
        self.last_time = 0

        # =========================
        # Hover throttle bias
        # =========================
        self.hover_pwm = BASE_THROTTLE

        # =========================
        # RC command
        # =========================
        self.cmd = RCMessage()
        self.cmd.rc_roll = BASE_ROLL
        self.cmd.rc_pitch = BASE_PITCH
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = BASE_THROTTLE

        # =========================
        # ROS interfaces
        # =========================
        self.command_pub = self.create_publisher(
            RCMessage, '/drone/rc_command', 10)

        self.error_pub = self.create_publisher(
            Error, '/pos_error', 10)

        self.create_subscription(
            PoseArray, '/whycon/poses',
            self.whycon_callback, 1)

        # =========================
        # Arming service
        # =========================
        self.arm_client = self.create_client(
            CommandBool, '/drone/cmd/arming')

        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')

        self.arm_drone(True)

        # =========================
        # Control loop
        # =========================
        self.create_timer(self.dt, self.pid_controller)

        self.get_logger().info('PID controller started')

    # =========================
    # Arm / Disarm
    # =========================
    def arm_drone(self, arm: bool):
        req = CommandBool.Request()
        req.value = arm
        self.arm_client.call_async(req)

    # =========================
    # WhyCon callback
    # =========================
    def whycon_callback(self, msg):
        self.current_state[0] = msg.poses[0].position.x / 10.0
        self.current_state[1] = msg.poses[0].position.y / 10.0
        self.current_state[2] = msg.poses[0].position.z / 10.0

    # =========================
    # PID controller
    # =========================
    def pid_controller(self):

        error = [
            self.current_state[i] - self.desired_state[i]
            for i in range(3)
        ]


        derivative = [
            (error[i] - self.prev_error[i])
            for i in range(3)
        ]

        # Integrate slowly (anti-windup)
        for i in range(3):
            self.integral[i] += error[i] * self.dt

        self.integral[0] = np.clip(self.integral[0], -5, 5)
        self.integral[1] = np.clip(self.integral[1], -3, 3)
        self.integral[2] = np.clip(self.integral[2], -4, 4)

        # PID outputs
        pitch_cmd = (
            self.Kp[0] * error[0] +
            self.Ki[0] * self.integral[0] +
            self.Kd[0] * derivative[0]
        )

        roll_cmd = (
            self.Kp[1] * error[1] +
            self.Ki[1] * self.integral[1] +
            self.Kd[1] * derivative[1]
        )

        thrust_cmd = (
            self.Kp[2] * error[2] +
            self.Ki[2] * self.integral[2] +
            self.Kd[2] * derivative[2]
        )

        # Hover bias learning (slow)
        #self.hover_pwm += 1.0 * error[2] * self.dt
        #self.hover_pwm = np.clip(self.hover_pwm, 1400, 1520)

        # Map to RC
        self.cmd.rc_roll = int(np.clip(
            BASE_ROLL - roll_cmd,
            MIN_ROLL, MAX_ROLL))

        self.cmd.rc_pitch = int(np.clip(
            BASE_PITCH - pitch_cmd,
            MIN_PITCH, MAX_PITCH))

        self.cmd.rc_throttle = int(np.clip(
            self.hover_pwm + thrust_cmd,
            MIN_THROTTLE, MAX_THROTTLE))

        self.command_pub.publish(self.cmd)

        # =========================
        # Debug print (single line)
        # =========================
        def err_symbol(e, tol=0.08):
            if abs(e) < tol:
                return '✅'
            return '+' if e > 0 else '-'


        print(
            f"{err_symbol(error[0])}"
            f"{err_symbol(error[1])}"
            f"{err_symbol(error[2])} "
            f"P={self.cmd.rc_pitch} "
            f"R={self.cmd.rc_roll} "
            f"T={self.cmd.rc_throttle}",
            flush=True
        )


        # Publish error
        err = Error()
        err.roll_error = error[1]
        err.pitch_error = error[0]
        err.throttle_error = error[2]
        self.error_pub.publish(err)

        self.prev_error = error.copy()


def main(args=None):
    rclpy.init(args=args)
    node = Swift_Pico()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.arm_drone(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

