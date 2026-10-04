#!/usr/bin/env python3
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from swift_msgs.msg import SwiftMsgs
from error_msg.msg import Error
from waypoint_navigation.action import NavToWaypoint
from control.matlab import c2d, ss
from control import dlqr

# ===========================
# Constants
# ===========================
MIN_ROLL, MAX_ROLL = 1000, 2000
MIN_PITCH, MAX_PITCH = 1000, 2000
MIN_THROTTLE, MAX_THROTTLE = 1250, 2000
BASE_ROLL = BASE_PITCH = BASE_THROTTLE = 1500
BASE_YAW = 1500

TOL_RADIUS = 0.08       # spherical tolerance (m)
HOLD_SEC = 2.0          # must stay stable inside sphere for 2s
TIMEOUT_SEC = 18.0      # hard cap per waypoint
AXIS_TOL = 0.04         # per-axis tolerance (m)
PROGRESS_LOG_PERIOD = 1.0  # seconds

# ===========================
# PicoServer Node
# ===========================
class PicoServer(Node):
    def __init__(self):
        super().__init__('pico_server')

        # --- Physical constants ---
        self.m, self.g = 0.152, 9.81
        self.sample_time = 0.01666

        # --- State tracking ---
        self.desired_state = [0.0]*6
        self.current_state = [0.0]*6
        self.prev_state = [0.0]*6
        self.raw_state = [0.0, 0.0, 0.0]
        self.first_cb = True
        self.goal_active = False
        self.completed_early = False

        # --- Hold/sphere tracking ---
        self.in_hold_zone = False
        self.stable_start_time_ros = None
        self.min_dist_current_wp = float('inf')
        self.hold_start_time = None
        self.max_hold_time = 0.0
        self.total_hold_time = 0.0
        self._last_whycon_ros_time = None

        # --- Vision and yaw ---
        self.last_vision_time = None
        self.marker_visible = False
        self.current_yaw = 0.0
        self.desired_yaw = None
        self.prev_rc_yaw = BASE_YAW
        self.max_yaw_step = 12
        self.K_yaw = 280

        # --- Subscriptions ---
        self.create_subscription(Odometry, '/rotors/odometry', self.odom_callback, 10)
        self.pose_array_sub = self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)

        # --- Hover ---
        self.hover_est = self.m * self.g

        # >>> Adaptive hover bias (NEW) <<<
        self.hover_bias = 0.0               # in m/s^2 equivalent (scaled below)
        self.hover_adapt_rate = 0.02        # adaptation per second
        self.last_hover_update = time.time()

        # --- LQR setup ---
        A = np.array([
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ], float)

        B = np.array([
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,self.g,0],
            [self.g,0,0],
            [0,0,1/self.m]
        ], float)

        # Keep your Q, but make Z less "snappy" by reducing throttle penalty in R
        Q = np.diag([45, 50, 45, 30, 30, 300])
        R = np.diag([140, 140, 260])  # was 260 → 180 (gentler throttle)

        sysd = c2d(ss(A, B, np.eye(6), np.zeros((6,3))), self.sample_time, method='zoh')
        self.K, _, _ = dlqr(sysd.A, sysd.B, Q, R)

        self.get_logger().info(f"LQR Gain Matrix K:\n{self.K}")

        # --- ROS Setup ---
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = self.cmd.rc_pitch = BASE_ROLL
        self.cmd.rc_yaw = BASE_YAW
        self.cmd.rc_throttle = BASE_THROTTLE
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.error_pub = self.create_publisher(Error, '/pos_error', 10)
        self._action_server = ActionServer(self, NavToWaypoint, 'waypoint_navigation', self.execute_callback)

        self.create_timer(self.sample_time, self.controller)
        self.arm()

        self.get_logger().info('✅ Pico LQR Server ready (filtered control + RAW hold logic + 20s timeout).')
        self.metrics = None
        self._last_progress_log = 0.0

    # ------------------------------------------------------------------
    def arm(self):
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)
        self.get_logger().info('🚁 Drone armed.')

    # ------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        sin_yaw = 2.0*(q.w*q.z + q.x*q.y)
        cos_yaw = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        self.current_yaw = math.atan2(sin_yaw, cos_yaw)
        if self.desired_yaw is None:
            self.desired_yaw = self.current_yaw
            self.get_logger().info(f"🎯 Yaw locked at {math.degrees(self.desired_yaw):.1f}°")

    # ------------------------------------------------------------------
    def whycon_callback(self, msg: PoseArray):
        if not msg.poses:
            return

        # --- Timestamp ---
        msg_time_ros = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_whycon_ros_time is None:
            dt_ros = 0.0
        else:
            dt_ros = max(0.0, msg_time_ros - self._last_whycon_ros_time)
        self._last_whycon_ros_time = msg_time_ros

        # --- Extract pose ---
        p = msg.poses[0].position
        self.update_state_from_pose(p.x, p.y, p.z)
        if not self.goal_active:
            return

        # --- Euclidean distance check ---
        cx, cy, cz = self.raw_state
        dx, dy, dz = self.desired_state[:3]
        dist = math.sqrt((dx - cx)**2 + (dy - cy)**2 + (dz - cz)**2)
        self.min_dist_current_wp = min(self.min_dist_current_wp, dist)
        inside = dist <= TOL_RADIUS

        # === Total hold time accumulation ===
        if inside:
            if self.hold_start_time is None:
                self.hold_start_time = msg_time_ros
                self.total_hold_time += dt_ros
            else:
                current_hold = msg_time_ros - self.hold_start_time
                if current_hold > self.max_hold_time:
                    self.max_hold_time = current_hold
                self.total_hold_time += dt_ros
        else:
            self.hold_start_time = None

        # === Success logic (2s stable inside sphere) ===
        if inside:
            if not self.in_hold_zone:
                self.in_hold_zone = True
                self.stable_start_time_ros = msg_time_ros
            else:
                held = msg_time_ros - self.stable_start_time_ros
                if held >= self.time_for_wp:
                    self.get_logger().info(
                        f"✅ Stabilized {held:.2f}s within {TOL_RADIUS:.3f}m (min_dist={self.min_dist_current_wp:.3f} m)"
                    )
                    self.completed_early = True
                    self.goal_active = False
                    self.in_hold_zone = False
        else:
            self.in_hold_zone = False
            self.stable_start_time_ros = None

    # ------------------------------------------------------------------
    def update_state_from_pose(self, x, y, z):
        x, y, z = x/10.0, y/10.0, -(z/10.0)
        self.raw_state = [x, y, z]
        self.prev_state = self.current_state.copy()

        if not hasattr(self, "pos_filtered"):
            self.pos_filtered = [x, y, z]

        alpha_pos = 0.2# smaller = smoother, 0.3–0.5 is typical
        self.pos_filtered = [
            (1 - alpha_pos)*p_f + alpha_pos*p_new
            for p_f, p_new in zip(self.pos_filtered, [x, y, z])
        ]
        self.current_state[0:3] = self.pos_filtered



        if self.first_cb:
            self.vel = [0.0, 0.0, 0.0]
            self.first_cb = False
        else:
            measured_v = [(self.current_state[i] - self.prev_state[i]) / self.sample_time for i in range(3)]

            # Compute overall speed (for adaptive smoothing)
            speed = math.sqrt(sum(v**2 for v in measured_v))
            alpha_vel = 0.15 if speed > 0.05 else 0.08  # smooth more when slow

            self.vel = [
                (1 - alpha_vel)*v_prev + alpha_vel*v_new
                for v_prev, v_new in zip(self.vel, measured_v)
            ]


        self.current_state[3:6] = self.vel
        self.last_vision_time = self.get_clock().now().nanoseconds / 1e9
        self.marker_visible = True

    # ------------------------------------------------------------------
    async def execute_callback(self, goal_handle):
        tx, ty, tz = goal_handle.request.x, goal_handle.request.y, goal_handle.request.z
        self.desired_state = [tx/10.0, ty/10.0, -tz/10.0, 0, 0, 0]
        self.time_for_wp = goal_handle.request.hold_time
        self.goal_active = True
        self.completed_early = False
        self.in_hold_zone = False
        self.min_dist_current_wp = float('inf')
        self.hold_start_time = None
        self.max_hold_time = 0.0
        self.total_hold_time = 0.0
        self._last_whycon_ros_time = None

        self.metrics = {
            'start_wall': time.time(),
            'max_abs_err': [0.0, 0.0, 0.0],
            'zone_prev': [0, 0, 0],
            'count_over': [0, 0, 0],
            'count_under': [0, 0, 0],
        }
        self.get_logger().info(
            f"➡️ New goal (WhyCon units): x={tx:.2f} cm, y={ty:.2f} cm, z={tz:.2f} cm "
            f"→ meters target: ({self.desired_state[0]:.3f}, {self.desired_state[1]:.3f}, {self.desired_state[2]:.3f})"
        )

        start_time = time.time()
        feedback = NavToWaypoint.Feedback()

        while rclpy.ok() and self.goal_active:
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT_SEC:
                self.goal_active = False
                break

            feedback.current_x = self.current_state[0]*10.0
            feedback.current_y = self.current_state[1]*10.0
            feedback.current_z = -self.current_state[2]*10.0
            goal_handle.publish_feedback(feedback)
            rclpy.spin_once(self, timeout_sec=0.05)

        result = NavToWaypoint.Result()
        result.stabilization_time = self.max_hold_time
        goal_handle.succeed()
        self.finalize_goal(succeeded=self.completed_early, elapsed_s=time.time() - start_time)
        return result

    # ------------------------------------------------------------------
    def _update_axis_metrics(self, error_vec):
        if self.metrics is None:
            return
        for i in range(3):
            e = float(error_vec[i])
            if abs(e) > self.metrics['max_abs_err'][i]:
                self.metrics['max_abs_err'][i] = abs(e)
            zone = 1 if e > AXIS_TOL else (-1 if e < -AXIS_TOL else 0)
            if zone != self.metrics['zone_prev'][i]:
                if zone == 1:
                    self.metrics['count_over'][i] += 1
                elif zone == -1:
                    self.metrics['count_under'][i] += 1
                self.metrics['zone_prev'][i] = zone

    # ------------------------------------------------------------------
    def controller(self):
        if not self.goal_active:
            return

        now_ros = self.get_clock().now().nanoseconds / 1e9
        age = now_ros - (self.last_vision_time or now_ros)
        if age > 0.35:
            for i in range(3):
                self.current_state[i] += self.current_state[3+i]*self.sample_time
            if age > 1.0:
                self.current_state[3:6] = [0.0, 0.0, 0.0]

        error = np.array(self.current_state) - np.array(self.desired_state)
        self._update_axis_metrics(error)

        error_for_ctrl = error.copy()
        error_for_ctrl[5] *= 0.7

        # Soft Z deadband
        Z_DEADBAND = 0.020
        if abs(error[2]) < Z_DEADBAND:
            scale = 1 - 0.6*(abs(error[2])/Z_DEADBAND)
            error_for_ctrl[2] *= scale
            error_for_ctrl[5] *= scale

        # >>> Adaptive hover bias update (NEW) <<<
        now = time.time()
        dt_bias = now - self.last_hover_update
        self.last_hover_update = now
        # Positive error[2] = current higher than target → reduce hover
        bias_correction = -float(error[2]) * self.hover_adapt_rate * max(0.0, dt_bias)
        self.hover_bias += bias_correction
        # Limit bias to a small range (±0.4 m/s^2 equivalent)
        self.hover_bias = float(np.clip(self.hover_bias, -0.4, 0.4))

        # LQR control
        u = -self.K @ error_for_ctrl
       

        roll  = float(np.clip(u[0], -math.pi/12, math.pi/12))
        pitch = float(np.clip(u[1], -math.pi/12, math.pi/12))
        thrust = float(u[2])

        # Apply hover + adaptive bias; bias is scaled by g so it's a fraction of hover
        base_hover = self.hover_est * (1.0 + self.hover_bias / self.g)

        cos_term = max(0.8, min(1.0, math.cos(roll)*math.cos(pitch)))
        Ft = (base_hover / cos_term) + thrust
        Ft = float(np.clip(Ft, 0.6*self.hover_est, 1.8*self.hover_est))
        target_throttle = self.force_to_throttle_total(Ft)

        self.cmd.rc_roll = int(self.linear_map(roll))
        self.cmd.rc_pitch = int(self.linear_map(pitch))
        self.cmd.rc_throttle = int(target_throttle)

        # Yaw control
        if self.desired_yaw is not None:
            yaw_err = math.atan2(math.sin(self.desired_yaw - self.current_yaw),
                                 math.cos(self.desired_yaw - self.current_yaw))
            yaw_cmd = BASE_YAW - self.K_yaw*yaw_err
            yaw_cmd = int(np.clip(yaw_cmd, 1100, 1900))
            yaw_cmd = int(np.clip(yaw_cmd, self.prev_rc_yaw - self.max_yaw_step,
                                  self.prev_rc_yaw + self.max_yaw_step))
            self.prev_rc_yaw = yaw_cmd
            self.cmd.rc_yaw = yaw_cmd

        self.command_pub.publish(self.cmd)

        err_msg = Error()
        err_msg.roll_error = error[1]
        err_msg.pitch_error = error[0]
        err_msg.throttle_error = error[2]
        self.error_pub.publish(err_msg)

    # ------------------------------------------------------------------
    def finalize_goal(self, succeeded: bool, elapsed_s: float):
        label = "✅ SUCCESS" if succeeded else "⏰ TIMEOUT"
        mx = self.metrics['max_abs_err'] if self.metrics else [0,0,0]
        co = self.metrics['count_over'] if self.metrics else [0,0,0]
        cu = self.metrics['count_under'] if self.metrics else [0,0,0]

        self.get_logger().info(
            f"{label} @ {elapsed_s:.2f}s | min_3D_dist={self.min_dist_current_wp:.3f} m\n"
            f"   Max |error| (m): x={mx[0]:.3f}, y={mx[1]:.3f}, z={mx[2]:.3f}\n"
            f"   Crossings beyond ±{AXIS_TOL:.2f} m:"
            f"  x(over={co[0]}, under={cu[0]}),"
            f" y(over={co[1]}, under={cu[1]}),"
            f" z(over={co[2]}, under={cu[2]})\n"
            f"   Max continuous hold inside {TOL_RADIUS:.2f} m sphere: {self.max_hold_time:.2f}s\n"
            f"   Total time inside {TOL_RADIUS:.2f} m sphere: {self.total_hold_time:.2f}s\n"
            f"   Final PWM: roll={self.cmd.rc_roll}, pitch={self.cmd.rc_pitch}, throttle={self.cmd.rc_throttle}, yaw={self.cmd.rc_yaw}"
        )

        # Reset
        self.goal_active = False
        self.in_hold_zone = False
        self.stable_start_time_ros = None
        self.hold_start_time = None
        self.max_hold_time = 0.0
        self.total_hold_time = 0.0
        self.metrics = None
        self.completed_early = False

    # ------------------------------------------------------------------
    def linear_map(self, x):
        return (200/(math.pi/12))*x + 1500

    def force_to_throttle_total(self, total_force):
        hover = self.m*self.g
        max_total = 2.0*hover
        slope = (MAX_THROTTLE - BASE_THROTTLE)/(max_total - hover)
        throttle = BASE_THROTTLE + slope*(total_force - hover) + 28
        return int(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))


# ===========================
# Main
# ===========================
def main(args=None):
    rclpy.init(args=args)
    node = PicoServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()