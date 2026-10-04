#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Point
from swift_msgs.msg import SwiftMsgs
from error_msg.msg import Error
from control.matlab import c2d, ss
from control import dlqr

MIN_ROLL, MAX_ROLL = 1000, 2000
MIN_PITCH, MAX_PITCH = 1000, 2000
MIN_THROTTLE, MAX_THROTTLE = 1250, 2000
BASE_ROLL = BASE_PITCH = BASE_THROTTLE = 1500

class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')
        self.m, self.g = 0.152, 9.81
        self.sample_time = 0.01666

        self.desired_state = [-0.7, 0.0, -2.0, 0.0, 0.0, 0.0]
        self.current_state = [0.0]*6
        self.prev_state = [0.0]*6
        self.first_cb = True

        A = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],
                      [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],float)
        
        B = np.array([[0,0,0],[0,0,0],[0,0,0],
                      [0,self.g,0],[self.g,0,0],[0,0,1/self.m]],float)
        
        Q = np.diag([40.0,29.0,66.0,30.0,2.0,30.0])

        R = np.diag([78.0,40.0,28.0])

        sysd = c2d(ss(A,B,np.eye(6),np.zeros((6,3))), self.sample_time, method='zoh')
        self.K, _, _ = dlqr(sysd.A, sysd.B, Q, R)
        self.get_logger().info(f"LQR Gain Matrix K:\n{self.K}")

        self.cmd = SwiftMsgs()
        self.cmd.rc_roll, self.cmd.rc_pitch = BASE_ROLL, BASE_PITCH
        self.cmd.rc_yaw, self.cmd.rc_throttle = 1500, BASE_THROTTLE

        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(Point, '/drone_goal', self.goal_callback, 10)

        self.arm()
        self.create_timer(self.sample_time, self.controller)

    def arm(self):
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    def goal_callback(self, msg):
        self.desired_state[0] = msg.x / 10.0
        self.desired_state[1] = msg.y / 10.0
        self.desired_state[2] = -msg.z / 10.0
        self.get_logger().info(
            f"🎯 New desired state: ({self.desired_state[0]:.2f}, {self.desired_state[1]:.2f}, {self.desired_state[2]:.2f})"
        )

    def whycon_callback(self, msg):
        self.prev_state = self.current_state.copy()
        if not hasattr(self, "z_filtered"): self.z_filtered = 0.0
        z_measured = -(msg.poses[0].position.z / 10.0)
        alpha_z = 0.6
        self.z_filtered = (1-alpha_z)*self.z_filtered + alpha_z*z_measured
        self.current_state[0] = msg.poses[0].position.x / 10.0
        self.current_state[1] = msg.poses[0].position.y / 10.0
        self.current_state[2] = self.z_filtered

        if self.first_cb:
            self.vel = [0.0, 0.0, 0.0]; self.first_cb = False
        else:
            measured_v = [(self.current_state[i]-self.prev_state[i])/self.sample_time for i in range(3)]
            alpha_v = 0.2
            self.vel = [(1-alpha_v)*v_prev + alpha_v*v_new for v_prev, v_new in zip(self.vel, measured_v)]
        self.current_state[3:6] = self.vel

    def linear_map(self, x): return (200/(math.pi/12))*x + 1500

    def force_to_throttle_linear(self, thrust_force):
        hover_thrust = self.m*self.g
        total_thrust = hover_thrust + thrust_force
        max_thrust = 2.0*hover_thrust
        slope = (MAX_THROTTLE - BASE_THROTTLE)/(max_thrust - hover_thrust)
        throttle = BASE_THROTTLE + slope*(total_thrust - hover_thrust) + 34
        return int(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))

    def controller(self):
        error = np.array(self.current_state) - np.array(self.desired_state)
        u = -self.K @ error
        roll, pitch, thrust = np.clip(u[0],-math.pi/12,math.pi/12), np.clip(u[1],-math.pi/12,math.pi/12), u[2]
        self.cmd.rc_roll = int(self.linear_map(roll))
        self.cmd.rc_pitch = int(self.linear_map(pitch))
        self.cmd.rc_throttle = self.force_to_throttle_linear(thrust)
        margin=0.04
        if all(abs(error[i])<margin for i in range(3)): self.get_logger().info("perfect")
        else:
            self.get_logger().info(f"x_err={error[0]:+.2f}, y_err={error[1]:+.2f}, z_err={error[2]:+.2f}, roll={self.cmd.rc_roll}, pitch={self.cmd.rc_pitch}, throttle={self.cmd.rc_throttle}")
        self.command_pub.publish(self.cmd)
        e=Error(); e.roll_error=error[1]; e.pitch_error=error[0]; e.throttle_error=error[2]; self.error_pub.publish(e)

def main(args=None):
    rclpy.init(args=args)
    node = Swift_Pico()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
