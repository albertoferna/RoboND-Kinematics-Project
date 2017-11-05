#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
import numpy as np


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        ### Your FK code here
        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])
     
            ### Your IK code here 
            # Calculate position of wrist center
            # Transform from 0 to end effector:
            T_0_eff = np.zeros((4, 4))
            T_0_eff[:3, :3] = calc_orient(roll, pitch, yaw)
            T_0_eff[:, 3] = np.array([px, py, pz, 1.0])
            T_0_eff = np.dot(T_0_eff, corr)
            wc_pos = np.dot(T_0_eff, np.array([0.0, 0.0, -0.303, 1.0]))
            theta1 = np.arctan2(wc_pos[1], wc_pos[0])
            # Two sides of the triangle come directly from the DH table
            side_b = 1.501 # sqrt(d4 ** 2 + a3 **2)
            side_a = 1.25 # a2
            # point at joint 2:
            x_p2 = get_point2(theta1)
            # Distance vector between joint and wrist
            distance = (wc_pos - x_p2[:, 0])
            # calculate side c
            side_c = np.linalg.norm(distance)
            # apply law of cosines to solve angles:
            # alpha = acos((side_b ** 2 + side_c ** 2 - side_a ** 2) / (2 * side_b * side_c))
            beta = acos((side_a ** 2 + side_c ** 2 - side_b ** 2) / (2 * side_a * side_c))
            gamma = acos((side_b ** 2 + side_a ** 2 - side_c ** 2) / (2 * side_b * side_a))
            # solve relations based on distances and angles
            theta2 = float(pi / 2 - beta - atan2(distance[2],
                                                 sqrt(distance[0] ** 2 + distance[1] ** 2)))
            theta3 = float(pi / 2 - (gamma + 0.036))
            # calculate transform 3-End effector
            # could not get it to work with LU decomposition. Using default method(gauss elim)
            T_0_3 = get_T_0_3(theta1, theta2, theta3)
            T_3_E = np.dot(np.linalg.inv(T_0_3[:3, :3]), T_0_eff[:3, :3])
            # calc final thetas
            theta4 = float(atan2(T_3_E[2, 2], -T_3_E[0, 2]))
            theta5 = float(atan2(sqrt(T_3_E[0, 2] ** 2 + T_3_E[2, 2] ** 2), T_3_E[1, 2]))
            theta6 = float(atan2(-T_3_E[1, 1], T_3_E[1, 0]))
            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()


def transform(alpha, a, d, q):
    """
    Helper function. Given the values for a tranformation between two joints it returns
    the tranformation matrix
    """
    T = Matrix([[cos(q), -sin(q), 0.0, a],
                [sin(q) * cos(alpha), cos(q) * cos(alpha), -sin(alpha), -sin(alpha) * d],
                [sin(q) * sin(alpha), cos(q) * sin(alpha),  cos(alpha),  cos(alpha) * d],
                [0.0, 0.0, 0.0, 1.0]])
    return T


def calc_Ts(dh, table):
    """
    Based on DH configuration return a list of transformations from one link to the next
    """
    T_s = []
    for alpha, a, d, q in table:
        T_s.append(transform(alpha, a, d, q).subs(dh))
    return T_s

def build_dh():
    """
    Builds the dh table for the robot as a dictionary and a list of tuples
    for the transformation matrices
    """
    # working with lists would make it easier latter on
    q_s = symbols('q1:8', real=True)
    d_s = symbols('d1:8', real=True)
    a_s = symbols('a0:7', real=True)
    alpha_s = symbols('alpha0:7', real=True)

    # DH Table configuration
    alpha = [0.0, -pi/2, 0.0, -pi/2, pi/2, -pi/2, 0.0]
    a = [0.0, 0.35, 1.25, -0.054, 0.0, 0.0, 0.0]
    d = [0.75, 0.0, 0.0, 1.5, 0.0, 0.0, 0.303]
    q = list(q_s)
    q[1] = q_s[1] - pi/2
    q[6] = 0.0
    dh = dict(zip(a_s, a))
    dh.update(dict(zip(alpha_s, alpha)))
    dh.update(dict(zip(d_s, d)))
    dh.update(dict(zip(q_s, q)))
    cols = zip(alpha_s, a_s, d_s, q_s)
    return dh, cols

def calc_orient(roll, pitch, yaw):
    """
    Starting from a RPY configuration, return the transformation matrix
    """
    R_x = np.array([[1, 0, 0],
                    [0, cos(roll), -sin(roll)],
                    [0, sin(roll),  cos(roll)]])
    R_y = np.array([[cos(pitch), 0, sin(pitch)],
                    [0, 1, 0],
                    [-sin(pitch), 0,  cos(pitch)]])
    R_z = np.array([[cos(yaw), -sin(yaw), 0],
                    [sin(yaw),  cos(yaw), 0],
                    [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y,R_x))


if __name__ == "__main__":
    # Some parameters to use globaly:
    # wrist center as homogeneous transform
    x_wc = symbols(('x_wc', 'y_wc', 'z_wc'))
    # Set up the math before running to avoid multiple execs
    dh_conf, dh_table = build_dh()
    Ts = calc_Ts(dh_conf, dh_table)
    # convinience variable:
    q_s = [dh_table[i][3] for i in range(len(dh_table))]
    # precalc useful symbolic matrices:
    T = Matrix.eye(4)
    for t in Ts:
        T *= t
    # correction matrix (its constant, calc numerically and keep)
    corr = np.zeros((4,4))
    corr[:3, :3] = calc_orient(0, -pi/2,pi)
    corr[3, 3] = 1
    # base frame of reference:
    origin = Matrix([0.0, 0.0, 0.0, 1.0])
    # Precalculate values and lambdify for speed
    # point at joint 2:
    point2 = (Ts[0] * Ts[1] * origin)
    get_point2 = lambdify(dh_table[0][3], point2, 'numpy')
    T_0_3 = Ts[0] * Ts[1] * Ts[2]
    get_T_0_3 = lambdify(q_s[0:3], T_0_3, 'numpy')
    

    IK_server()
