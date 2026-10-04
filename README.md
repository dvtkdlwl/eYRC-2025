# eYRC-2025
This is my repository detailing and containing all my work related to the IIT-Bombay eYantra 2025 Robotics Challenge. My team's problem statement was the autonomous control of a microdrone deployed in a greenhouse farm through visual localisation methods. The first stage of the competition consisted of doing everything on simulation, and on the basis of our (almost-perfect) performance in stage 1, we qualified for stage-2- where we received hardware kits to execute our code on.

<div>
  <img src="https://github.com/user-attachments/assets/fb5552f4-2da3-4893-986f-40a11a036c04"
       width="49%" />
  <img src="https://github.com/user-attachments/assets/ef71d45a-52c7-40d0-b613-b5dc23182b73"
       width="49%" />
</div>

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

## Technologies Used
The broad concepts explored throughout the project include:
1. Camera calibration
2. Image coordinate systems
3. OpenCV techniques
4. Homography and perspective transformation
5. ArUco marker detection
6. WhyCon-based localisation
7. Coordinate-frame transformations
8. ROS 2 communication
9. PID control
10. LQR control
11. Closed-loop autonomous navigation
12. Gazebo simulation

> **Attribution:** The base simulation environments, robot models, and boilerplate code are provided by **e-Yantra, IIT Bombay** for eYRC-2025. This repository primarily contains our team's implementations, modifications, and solutions built upon that framework.
