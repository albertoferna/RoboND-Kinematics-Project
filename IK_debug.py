from sympy import *
from time import time
from mpmath import radians
import tf
import numpy as np

'''
Format of test case is [ [[EE position],[EE orientation as quaternions]],[WC location],[joint angles]]
You can generate additional test cases by setting up your kuka project and running `$ roslaunch kuka_arm forward_kinematics.launch`
From here you can adjust the joint angles to find thetas, use the gripper to extract positions and orientation (in quaternion xyzw) and lastly use link 5
to find the position of the wrist center. These newly generated test cases can be added to the test_cases dictionary.
'''

test_cases = {1:[[[2.16135,-1.42635,1.55109],
                  [0.708611,0.186356,-0.157931,0.661967]],
                  [1.89451,-1.44302,1.69366],
                  [-0.65,0.45,-0.36,0.95,0.79,0.49]],
              2:[[[-0.56754,0.93663,3.0038],
                  [0.62073, 0.48318,0.38759,0.480629]],
                  [-0.638,0.64198,2.9988],
                  [-0.79,-0.11,-2.33,1.94,1.14,-3.68]],
              3:[[[-1.3863,0.02074,0.90986],
                  [0.01735,-0.2179,0.9025,0.371016]],
                  [-1.1669,-0.17989,0.85137],
                  [-2.99,-0.12,0.94,4.06,1.29,-4.12]],
              4:[[[0.743086, -0.618487, 0.0619249],
                  [0.684884,-0.510046, -0.34054, -0.393472]],
                  [1.89451046, -1.44302032, 1.69366545],
                  [-0.630919088657, 0.5503197815830001, 1.1023063526474997, -5.9156191294200005, -0.59952061455, -2.68658539149]],
              5:[]}


def test_code(test_case):
    ## Set up code
    ## Do not modify!
    x = 0
    class Position:
        def __init__(self,EE_pos):
            self.x = EE_pos[0]
            self.y = EE_pos[1]
            self.z = EE_pos[2]
    class Orientation:
        def __init__(self,EE_ori):
            self.x = EE_ori[0]
            self.y = EE_ori[1]
            self.z = EE_ori[2]
            self.w = EE_ori[3]

    position = Position(test_case[0][0])
    orientation = Orientation(test_case[0][1])

    class Combine:
        def __init__(self,position,orientation):
            self.position = position
            self.orientation = orientation

    comb = Combine(position,orientation)

    class Pose:
        def __init__(self,comb):
            self.poses = [comb]

    req = Pose(comb)
    start_time = time()
    
    ########################################################################################
    ## 

    ## Insert IK code here!
    """# Set needed symbols
    # working with lists would make it easier latter on
    q_s = symbols('q1:8', real=True)
    d_s = symbols('d1:8', real=True)
    a_s = symbols('a0:7', real=True)
    alpha_s = symbols('alpha0:7', real=True)
    # wrist center as homogeneous transform
    x_wc = symbols(('x_wc', 'y_wc', 'z_wc'), real=True)
    wrist_center = Matrix(x_wc).row_insert(3, Matrix([1.0]))
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
    dh.update(dict(zip(q_s, q)))"""

    # Effector orientation. Transform quaternion into roll, pitch, yaw
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])
    
    # inverse kinematics configuration:
    ik_conf = {'ee_x': req.poses[x].position.x, 'ee_y': req.poses[x].position.y, 'ee_z': req.poses[x].position.z,
               'roll': roll, 'pitch': pitch, 'yaw': yaw}

    # Calculate position of wrist center
    # Transform from 0 to end effector:
    T_0_eff = np.zeros((4, 4))
    T_0_eff[:3, :3] = calc_orient(roll, pitch, yaw)
    T_0_eff[:, 3] = np.array([ik_conf['ee_x' ], ik_conf['ee_y'], ik_conf['ee_z'], 1.0])
    T_0_eff = np.dot(T_0_eff, corr)
    wc_pos = np.dot(T_0_eff, np.array([0.0, 0.0, -0.303, 1.0]))
    ik_conf.update(dict(zip(x_wc, list(wc_pos[:3]))))
    theta1 = np.arctan2(wc_pos[1], wc_pos[0])
    # Two sides of the triangle come directly from the DH table
    side_b = 1.501 # sqrt(d4 ** 2 + a3 **2)
    side_a = 1.25 # a2
    # point at joint 2:
    x_p2 = get_point2(theta1)
    # Distance vector between joint and wrist
    distance = (wc_pos - x_p2[:,0])
    # calculate side c
    side_c = np.linalg.norm(distance)
    # apply law of cosines to solve angles:
    # alpha angle not needed
    # alpha = acos((side_b ** 2 + side_c ** 2 - side_a ** 2) / (2 * side_b * side_c))
    beta = acos((side_a ** 2 + side_c ** 2 - side_b ** 2) / (2 * side_a * side_c))
    gamma = acos((side_b ** 2 + side_a ** 2 - side_c ** 2) / (2 * side_b * side_a))
    # solve relations based on distances and angles
    theta2 = float(pi / 2 - beta - atan2(distance[2], sqrt(distance[0] ** 2 + distance[1] ** 2)))
    theta3 = float(pi / 2 - (gamma + 0.036))
    # update ik_conf with calculated values
    theta_1_3 = (theta1, theta2, theta3)
    ik_conf.update(dict(zip(q_s[:3], theta_1_3)))
    # calculate transform 3-End effector
    # could not get it to work with LU decomposition. Using default method(gauss elim)
    T_0_3 = get_T_0_3(theta1, theta2, theta3)
    T_3_E = np.dot(np.linalg.inv(T_0_3[:3, :3]), T_0_eff[:3, :3])
    # calc final thetas
    theta4 = float(atan2(T_3_E[2, 2], -T_3_E[0, 2]))
    theta5 = float(atan2(sqrt(T_3_E[0, 2] ** 2 + T_3_E[2, 2] ** 2), T_3_E[1, 2]))
    theta6 = float(atan2(-T_3_E[1, 1], T_3_E[1, 0]))
    # update ik_conf
    ik_conf.update(dict(zip(q_s[3:6], [float(theta4), float(theta5), float(theta6)])))

    ## 
    ########################################################################################
    
    ########################################################################################
    ## For additional debugging add your forward kinematics here. Use your previously calculated thetas
    ## as the input and output the position of your end effector as your_ee = [x,y,z]

    ## (OPTIONAL) YOUR CODE HERE!
    
    T_wc = Matrix.eye(4)
    for t in Ts[:4]:
        T_wc *= t

    corrected_T = (T.subs(ik_conf) * corr)
    origin = Matrix([0.0, 0.0, 0.0, 1.0])

    ## End your code input for forward kinematics here!
    ########################################################################################

    ## For error analysis please set the following variables of your WC location and EE location in the format of [x,y,z]
    your_wc = (T_wc.subs(ik_conf) * origin)[:3] # <--- Load your calculated WC values in this array
    your_ee = (corrected_T * origin)[:3] # <--- Load your calculated end effector value from your forward kinematics
    ########################################################################################

    ## Error analysis
    print ("\nTotal run time to calculate joint angles from pose is %04.4f seconds" % (time()-start_time))

    # Find WC error
    if not(sum(your_wc)==3):
        wc_x_e = abs(your_wc[0]-test_case[1][0])
        wc_y_e = abs(your_wc[1]-test_case[1][1])
        wc_z_e = abs(your_wc[2]-test_case[1][2])
        wc_offset = sqrt(wc_x_e**2 + wc_y_e**2 + wc_z_e**2)
        print ("\nWrist error for x position is: %04.8f" % wc_x_e)
        print ("Wrist error for y position is: %04.8f" % wc_y_e)
        print ("Wrist error for z position is: %04.8f" % wc_z_e)
        print ("Overall wrist offset is: %04.8f units" % wc_offset)

    # Find theta errors
    t_1_e = abs(theta1-test_case[2][0])
    t_2_e = abs(theta2-test_case[2][1])
    t_3_e = abs(theta3-test_case[2][2])
    t_4_e = abs(theta4-test_case[2][3])
    t_5_e = abs(theta5-test_case[2][4])
    t_6_e = abs(theta6-test_case[2][5])
    print ("\nTheta 1 error is: %04.8f" % t_1_e)
    print ("Theta 2 error is: %04.8f" % t_2_e)
    print ("Theta 3 error is: %04.8f" % t_3_e)
    print ("Theta 4 error is: %04.8f" % t_4_e)
    print ("Theta 5 error is: %04.8f" % t_5_e)
    print ("Theta 6 error is: %04.8f" % t_6_e)
    print ("\n**These theta errors may not be a correct representation of your code, due to the fact \
           \nthat the arm can have muliple positions. It is best to add your forward kinmeatics to \
           \nconfirm whether your code is working or not**")
    print (" ")

    # Find FK EE error
    if not(sum(your_ee)==3):
        ee_x_e = abs(your_ee[0]-test_case[0][0][0])
        ee_y_e = abs(your_ee[1]-test_case[0][0][1])
        ee_z_e = abs(your_ee[2]-test_case[0][0][2])
        ee_offset = sqrt(ee_x_e**2 + ee_y_e**2 + ee_z_e**2)
        print ("\nEnd effector error for x position is: %04.8f" % ee_x_e)
        print ("End effector error for y position is: %04.8f" % ee_y_e)
        print ("End effector error for z position is: %04.8f" % ee_z_e)
        print ("Overall end effector offset is: %04.8f units \n" % ee_offset)


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
    Ts = []
    for alpha, a, d, q in table:
        Ts.append(transform(alpha, a, d, q).subs(dh))
    return Ts

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
    R_x = np.array([[ 1, 0, 0],
                [ 0, cos(roll), -sin(roll)],
                [ 0, sin(roll),  cos(roll)]])
    R_y = np.array([[ cos(pitch), 0, sin(pitch)],
                [ 0, 1, 0],
                [-sin(pitch), 0,  cos(pitch)]])
    R_z = np.array([[ cos(yaw), -sin(yaw), 0],
                [ sin(yaw),  cos(yaw), 0],
                [ 0, 0, 1]])
    return np.dot(R_z, np.dot(R_y,R_x))

if __name__ == "__main__":
    # Change test case number for different scenarios
    test_case_number = 1


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
    # point at joint 2:
    point2 = (Ts[0] * Ts[1] * origin)
    get_point2 = lambdify(dh_table[0][3], point2, 'numpy')
    T_0_3 = Ts[0] * Ts[1] * Ts[2]
    get_T_0_3 = lambdify(q_s[0:3], T_0_3, 'numpy')
    
    test_code(test_cases[test_case_number])
