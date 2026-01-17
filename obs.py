#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_node')
        
        # Subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.lidar_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/mavros/setpoint_velocity/cmd_vel_unstamped',
            10
        )
        
        # Initialize variables
        self.current_pose = None
        self.lidar_data = None
        self.target_position = [10.0, 10.0, 5.0]  # Example target
        
        # Control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
    def lidar_callback(self, msg):
        """Process lidar scan data"""
        self.lidar_data = msg
        
    def pose_callback(self, msg):
        """Update current position"""
        self.current_pose = msg
        
    def detect_obstacles(self):
        """Convert lidar data to obstacle positions"""
        if not self.lidar_data:
            return []
            
        obstacles = []
        
        for i, distance in enumerate(self.lidar_data.ranges):
            if distance < self.lidar_data.range_max and distance > self.lidar_data.range_min:
                angle = self.lidar_data.angle_min + i * self.lidar_data.angle_increment
                
                # Convert to x, y coordinates
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                
                if distance < 3.0:  # Only consider close obstacles
                    obstacles.append([x, y])
                    
        return obstacles
        
    def potential_field_navigation(self, obstacles):
        """Potential field-based path planning"""
        if not self.current_pose:
            return Twist()
            
        current_pos = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y
        ])
        
        target_pos = np.array(self.target_position[:2])
        
        # Attractive force toward target
        target_vector = target_pos - current_pos
        distance_to_target = np.linalg.norm(target_vector)
        
        if distance_to_target > 0.1:
            attractive_force = target_vector / distance_to_target
        else:
            attractive_force = np.array([0.0, 0.0])
            
        # Repulsive forces from obstacles
        repulsive_force = np.array([0.0, 0.0])
        
        for obstacle in obstacles:
            obs_vector = current_pos - np.array(obstacle)
            obs_distance = np.linalg.norm(obs_vector)
            
            if obs_distance < 2.0 and obs_distance > 0.1:
                repulsive_force += (obs_vector / obs_distance) * (1.0 / obs_distance)
                
        # Combine forces
        total_force = attractive_force + repulsive_force * 3.0
        
        # Create Twist message
        cmd = Twist()
        cmd.linear.x = np.clip(total_force[0], -2.0, 2.0)
        cmd.linear.y = np.clip(total_force[1], -2.0, 2.0)
        cmd.linear.z = 0.0  # Maintain altitude
        
        return cmd
        
    def control_loop(self):
        """Main control loop"""
        obstacles = self.detect_obstacles()
        cmd_vel = self.potential_field_navigation(obstacles)
        self.cmd_vel_pub.publish(cmd_vel)

def main():
    rclpy.init()
    node = ObstacleAvoidanceNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()