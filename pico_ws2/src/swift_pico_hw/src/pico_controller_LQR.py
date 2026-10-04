#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from rc_msgs.msg import RCMessage
from rc_msgs.srv import CommandBool
from error_msg.msg import Error
from sensor_msgs.msg import BatteryState


# =========================
# SAFE HARDWARE LIMITS
# =========================
MIN_ROLL = 1400
BASE_ROLL = 1450
MAX_ROLL = 1600

MIN_PITCH = 1400
BASE_PITCH = 1470
MAX_PITCH = 1600

MIN_THROTTLE = 1300
BASE_THROTTLE = 1450
MAX_THROTTLE = 1550

X_TOL = 0.08
Y_TOL = 0.08
Z_TOL = 0.08


class Swift_Pico(Node):

    def __init__(self):
        super().__init__('pico_pid_controller')

        # =========================
        # Timing
        # =========================
        self.dt = 0.033  # ~30 Hz

        # =========================
        # States
        # =========================
        self.current_state = [0.0, 0.0, 0.0]
        self.desired_state = [0.0, -0.55, 1.9]

        # =========================
        # PID gains
        # =========================
        self.Kp = [5, 18, 40.0]
        self.Ki = [2.0, 5.0, 1.0]
        self.Kd = [2.0, 5.0, 22.0]

        self.integral = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]

        # =========================
        # Voltage-based hover scaling (NO INTEGRATOR)
        # =========================
        self.voltage_table = np.array([4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7])
        self.hover_table   = np.array([1450, 1450,1460,1470,1480,1490,1500])

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
            RCMessage, '/drone/rc_command', 1)

        self.error_pub = self.create_publisher(
            Error, '/pos_error', 10)

        self.create_subscription(
            PoseArray, '/whycon/poses',
            self.whycon_callback, 1)

        self.create_subscription(
            BatteryState,
            '/drone/battery_info',
            self.battery_callback,
            10)

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
    # Battery callback (feedforward ONLY)
    # =========================
    def battery_callback(self, msg):
        voltage = msg.voltage
        if voltage <= 0.0:
            return

        voltage = np.clip(
            voltage,
            self.voltage_table.min(),
            self.voltage_table.max()
        )

        self.hover_pwm = float(
            np.interp(voltage, self.voltage_table, self.hover_table)
        )

        self.hover_pwm = np.clip(self.hover_pwm, 1400, 1520)

    # =========================
    # PID controller
    # =========================
    def pid_controller(self):

        error = [
            self.current_state[i] - self.desired_state[i]
            for i in range(3)
        ]

        # Debug emojis
        for e in error:
            print("✅" if abs(e) < 0.08 else "❌", end='')
        print()

        derivative = [
            (error[i] - self.prev_error[i]) / self.dt
            for i in range(3)
        ]

        for i in range(3):
            self.integral[i] += error[i] * self.dt

        self.integral[0] = np.clip(self.integral[0], -10, 10)
        self.integral[1] = np.clip(self.integral[1], -10, 10)
        self.integral[2] = np.clip(self.integral[2], -0.5, 0.5)

        pitch_cmd = (
            self.Kp[0]*error[0] +
            self.Ki[0]*self.integral[0] +
            self.Kd[0]*derivative[0]
        )

        roll_cmd = (
            self.Kp[1]*error[1] +
            self.Ki[1]*self.integral[1] +
            self.Kd[1]*derivative[1]
        )

        thrust_cmd = (
            self.Kp[2]*error[2] +
            self.Ki[2]*self.integral[2] +
            self.Kd[2]*derivative[2]
        )

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
