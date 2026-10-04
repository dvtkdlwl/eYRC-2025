#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints
from geometry_msgs.msg import Point
from std_msgs.msg import String


class PicoClient(Node):
    def __init__(self):
        super().__init__('pico_client')

        # --- Waypoint service client ---
        self.cli = self.create_client(GetWaypoints, 'get_waypoints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for waypoint service...')

        self.req = GetWaypoints.Request()
        self.req.get_all = True

        # --- Action client (to pico_server) ---
        self._action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')

        # --- Optional publishers ---
        self.goal_pub = self.create_publisher(Point, '/drone_goal', 10)  # visualization/scaling
        self.inf_pub  = self.create_publisher(String, '/infected_plants', 10)  # if you want to hard-set labels

        # --- Stats ---
        self.completed_waypoints = 0
        self.total_stabilization_time = 0.0

    # ------------------------------------------------------------------
    def publish_infected(self, b1_label: str, b2_label: str):
        msg = String()
        msg.data = f'{b1_label},{b2_label}'
        self.inf_pub.publish(msg)
        self.get_logger().info(f'📡 Published /infected_plants="{msg.data}"')

    # ------------------------------------------------------------------
    def get_waypoints(self):
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is None:
            raise RuntimeError("GetWaypoints service call failed (no response).")
        return resp.waypoints

    # ------------------------------------------------------------------
    def send_goal(self, wp):
        """
        Send a single waypoint to the pico_server action with the waypoint's hold_time.
        wp is waypoint_navigation.msg.Waypoint (fields: x, y, z, hold_time) in *cm*.
        """
        # Publish scaled goal for any visualization you have
        p = Point()
        p.x, p.y, p.z = wp.x / 10.0, wp.y / 10.0, -wp.z / 10.0
        self.goal_pub.publish(p)

        # Build action goal (server converts cm->m and flips z internally)
        goal_msg = NavToWaypoint.Goal()
        goal_msg.x = wp.x
        goal_msg.y = wp.y
        goal_msg.z = wp.z
        goal_msg.hold_time = getattr(wp, 'hold_time', 0.2)  # safety default

        self.get_logger().info(
            f"▶️  Sending goal: ({wp.x:.2f}, {wp.y:.2f}, {wp.z:.2f}) cm, hold={goal_msg.hold_time:.2f}s"
        )

        # Send and wait for result
        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('⚠️ Goal rejected by server.')
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        # Stats/logs
        self.completed_waypoints += 1
        self.total_stabilization_time += float(result.stabilization_time)

        self.get_logger().info(
            f'✅ Waypoint #{self.completed_waypoints} done — server reports stabilization {result.stabilization_time:.2f}s'
        )

    # ------------------------------------------------------------------
    def feedback_callback(self, feedback_msg):
        # If you want live telemetry, uncomment this:
        # fb = feedback_msg.feedback
        # self.get_logger().info(f'Feedback → ({fb.current_x:.2f}, {fb.current_y:.2f}, {fb.current_z:.2f})')
        pass


# ------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = PicoClient()
    start_time = time.time()
    node.get_logger().info(f"🕐 Mission started at {time.strftime('%H:%M:%S')}")

    try:
        # If your separate vision node already published /infected_plants, skip.
        # Otherwise, you can hard-set (uncomment if needed):
        # node.publish_infected('P1C', 'P2A')

        waypoints = node.get_waypoints()
        node.get_logger().info(f"📦 Received {len(waypoints)} waypoints.")

        for wp in waypoints:
            node.send_goal(wp)

        # Completed all waypoints
        total_time = time.time() - start_time
        avg_stab = node.total_stabilization_time / max(node.completed_waypoints, 1)
        node.get_logger().info(
            f"🏁 Mission complete: {node.completed_waypoints} waypoints in {total_time:.2f}s "
            f"(avg stabilization: {avg_stab:.2f}s)"
        )

    except KeyboardInterrupt:
        total_time = time.time() - start_time
        if node.completed_waypoints > 0:
            avg_stab = node.total_stabilization_time / node.completed_waypoints
            node.get_logger().warn(
                f"⚠️ Mission aborted by user after {node.completed_waypoints} waypoints. "
                f"Elapsed: {total_time:.2f}s, Avg stabilization: {avg_stab:.2f}s"
            )
        else:
            node.get_logger().warn(f"⚠️ Mission aborted before completion. Elapsed: {total_time:.2f}s")
    finally:
        node.get_logger().info("🔻 Shutting down PicoClient...")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()