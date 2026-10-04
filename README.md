# eYRC-2025
This is my repository detailing and containing all my work related to the IIT-Bombay eYantra 2025 Robotics Challenge. My team's problem statement was the autonomous control of a microdrone deployed in a greenhouse farm through visual localisation methods.

<img width="1280" height="960" alt="WhatsApp Image 2026-10-04 at 16 54 05" src="https://github.com/user-attachments/assets/fb5552f4-2da3-4893-986f-40a11a036c04" />

## Problem Statement
Our detailed problem statement was as follows:
1. Detect the WhyCon marker visibly attached to the drone's center,
2. Detect the ArUco markers representing the corners of the arena,
3. Crop out the excess and apply perspective transforms in order to straighten the arena for the ceiling-mounted camera,
4. Use OpenCV to detect the 2 infected (yellow) plants in the farm,
5. Find their coordinates, and generate a waypoint trajectory for the drone to follow,
7. The drone takes off and hovers over its homebase ("H" on the arena),
8. Make the drone stabilise using the fine-tuned PID (LQR for simulation) controller at all of the received checkpoints in order (such as pesticide pickup, pesticide drop, arena entrance, etc.)
9. Make it return to base and land.
