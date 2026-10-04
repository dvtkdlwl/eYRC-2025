#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from waypoint_navigation.srv import GetWaypoints
from waypoint_navigation.msg import Waypoint
from std_msgs.msg import String

class WaypointService(Node):
    def __init__(self):
        super().__init__('waypoint_service')
        self.srv = self.create_service(GetWaypoints, 'get_waypoints', self.get_waypoints_callback)
        self.create_subscription(String, '/infected_plants', self.infected_cb, 10)
        self.infected_b1 = 'P1C'
        self.infected_b2 = 'P2A'
        self.get_logger().info('✅ Waypoint service ready.')

        # ------------------- Tables (cm) -------------------
        # Ground + hover
        self.GROUND = (-6.90, 0.09, 32.16, 0)
        self.HOVER  = (-7.00, 0.00, 30.22, 0)

        # Pesticide stations
        self.PS1_PATH = [
            (-7.64,  3.06, 30.22, 0),
            (-8.22,  6.02, 30.22, 0),
            (-9.11,  9.27, 31.27, 1)  # destination
        ]
        self.PS1_PICKUP = (-9.07, 9.23, 32.57, 1)

        self.PS2_PATH = [
            (-7.49, -2.83, 30.22, 0),
            (-7.82, -5.55, 30.22, 0),
            (-8.62, -8.60, 29.60, 1)  # destination
        ]
        self.PS2_PICKUP = (-8.53, -8.52, 30.64, 1)

        # Hula hoops (path between blocks/stations)
        self.HH2_ENTRY = (-3.26,  8.41, 29.88, 0)
        self.HH2_EXIT  = ( 0.87,  8.18, 29.05, 0)
        self.HH1_ENTRY = (-3.26, -8.12, 29.48, 0)
        self.HH1_EXIT  = ( 0.80, -8.04, 29.20, 0)

        # Blocks centers
        self.BLOCK1 = ( 6.60,  6.62, 30.20, 0)
        self.BLOCK2 = ( 7.15, -7.00, 30.98, 0)

        # Block1 helper waypoints
        self.B1_WP = [
            (6.7375,  3.3200, 30.6450, 0),
            (6.8750, -0.1900, 30.0900, 0),
            (7.0125, -3.5950, 28.5350, 0)
        ]
        # Block2 helper from east side
        self.B2_WP_FROM_EAST = [
            (4.0, -7.62, 30.59, 0),
        ]

        # Block1 plant hover / application
        self.P1_HOVER = {
            'P1A': (4.39,  8.98, 29.29, 1),
            'P1B': (6.45,  8.79, 28.67, 1),
            'P1C': (8.62,  8.80, 28.69, 1),
            'P1D': (4.37,  4.57, 29.19, 1),
            'P1E': (6.76,  4.49, 28.77, 1),
            'P1F': (8.70,  4.53, 28.97, 1),
        }
        self.P1_APPLY = {
            'P1A': (4.50,  8.88, 30.59, 1),
            'P1B': (6.85,  8.89, 30.62, 1),
            'P1C': (9.13,  8.80, 30.40, 1),
            'P1D': (4.53,  4.57, 30.79, 1),
            'P1E': (6.45,  4.57, 30.79, 1),
            'P1F': (9.10,  4.53, 28.97, 1),
        }

        # Block2 plant hover / application
        self.P2_HOVER = {
            'P2A': (4.43, -4.30, 29.74, 1),
            'P2B': (6.68, -4.30, 29.74, 1),
            'P2C': (8.93, -4.30, 29.74, 0),
            'P2D': (4.58, -8.78, 30.62, 1),
            'P2E': (6.76, -8.61, 30.03, 1),
            'P2F': (9.17, -8.61, 30.03, 1),
        }
        self.P2_APPLY = {
            'P2A': (4.43, -4.30, 31.10, 1),
            'P2B': (7.05, -4.30, 31.52, 1),
            'P2C': (9.32, -4.30, 31.06, 1),
            'P2D': (4.53, -8.59, 30.76, 1),
            'P2E': (7.05, -8.81, 31.52, 1),
            'P2F': (9.08, -8.92, 31.93, 1),
        }

        # Return path to ground
        self.RETURN_PATH = [
            (-7.82, -5.55, 30.22, 0),
            (-7.49, -2.83, 30.22, 0),
            self.GROUND
        ]

    # ------------------------------------------------------------------
    def infected_cb(self, msg: String):
        try:
            text = msg.data.strip()
            # Accept formats: "P1C,P2A" or "P1C;P2A" or "P1C P2A"
            parts = [p.strip() for p in text.replace(';', ',').replace(' ', ',').split(',') if p.strip()]
            if len(parts) >= 1 and parts[0].startswith('P1'):
                self.infected_b1 = parts[0]
            if len(parts) >= 2 and parts[1].startswith('P2'):
                self.infected_b2 = parts[1]
            self.get_logger().info(f'🌿 Infected plants set → {self.infected_b1}, {self.infected_b2}')
        except Exception as e:
            self.get_logger().warn(f'Failed to parse /infected_plants: {e}')

    # ------------------------------------------------------------------
    def get_waypoints_callback(self, request, response):
        # Build full mission list (in cm), always go to Block 1 first.
        b1 = self.infected_b1 if self.infected_b1 in self.P1_HOVER else 'P1A'
        b2 = self.infected_b2 if self.infected_b2 in self.P2_HOVER else 'P2A'

        coords = []

        # 1) Ground hover (takeoff staging)
        coords += [self.GROUND, self.HOVER]

        # 2) To Pesticide Station 1 (for Block 1)
        coords += self.PS1_PATH + [self.PS1_PICKUP]

        # 3) To Block 1 via HH2 (entry→exit), then Block 1 center
        coords += [
            (-5.976, 8.810, 31.27, 0),
            self.HH2_ENTRY,
            self.HH2_EXIT,
            (3.929, 7.347, 29.05, 0),
            self.BLOCK1
        ]

        # 4) Block 1 infected: hover then application (critical)
        coords += [
            self.P1_HOVER[b1],
            self.P1_APPLY[b1]
        ]

        # 5) Traverse across to Block 2 (helper WPs)
        coords += self.B1_WP + [self.BLOCK2]

        # 6) Block 2 infected: hover then application (critical)
        coords += [
            self.P2_HOVER[b2],
            self.P2_APPLY[b2]
        ]

        # 7) Go to Pesticide Station 2 (through HH1 exit→entry side), then pickup
        coords += self.B2_WP_FROM_EAST + [self.HH1_EXIT, self.HH1_ENTRY]
        coords += [
            (-5.94, -8.36, 29.54, 0),
            self.PS2_PATH[-1],  # destination of PS2
            self.PS2_PICKUP
        ]

        # 8) Return path to ground
        coords += self.RETURN_PATH

        # Fill response
        response.waypoints = []
        for x, y, z, priority in coords:
            wp = Waypoint()
            wp.x, wp.y, wp.z = float(x), float(y), float(z)  # cm as before
            if priority == 1:
                wp.hold_time = 2.0
            else:
                wp.hold_time = 0.0
            response.waypoints.append(wp)

        if request.get_all:
            return response

def main(args=None):
    rclpy.init(args=args)
    node = WaypointService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()