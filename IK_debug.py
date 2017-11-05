from sympy import *
from time import time
from mpmath import radians
import tf

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
              4:[],
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
    # base frame of reference:
    origin = Matrix([0.0, 0.0, 0.0, 1.0])
    # Set needed symbols
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
    dh.update(dict(zip(q_s, q)))
    # end effector symbols:
    roll_s = symbols('r', real=True)
    pitch_s = symbols('p', real=True)
    yaw_s = symbols('y', real=True)
    ee_x = symbols('ee_x', real=True)
    ee_y = symbols('ee_y', real=True)
    ee_z = symbols('ee_z', real=True)
    # Generic rotation matrices:
    rot_1, rot_2, rot_3 = symbols('rot_1:4')
    R_x = Matrix([[ 1, 0, 0],
                [ 0, cos(rot_1), -sin(rot_1)],
                [ 0, sin(rot_1),  cos(rot_1)]])
    R_y = Matrix([[ cos(rot_2), 0, sin(rot_2)],
                [ 0, 1, 0],
                [-sin(rot_2), 0,  cos(rot_2)]])
    R_z = Matrix([[ cos(rot_3), -sin(rot_3), 0],
                [ sin(rot_3),  cos(rot_3), 0],
                [ 0, 0, 1]])

    # Effector orientation. Transform quaternion into roll, pitch, yaw
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])
    
    # inverse kinematics configuration:
    ik_conf = {ee_x: req.poses[x].position.x, ee_y: req.poses[x].position.y, ee_z: req.poses[x].position.z,
               roll_s: roll, pitch_s: pitch, yaw_s: yaw}

    # Calculate position of wrist center
    # Correction of orientation difference between urdf and DH:
    corr = (R_z.subs(rot_3, pi) * R_y.subs(rot_2, -pi/2))
    corr = corr.row_insert(3, Matrix([0,0,0]).transpose())
    corr = corr.col_insert(3, Matrix([0,0,0,1]))
    # Transform from 0 to end effector:
    T_0_eff = R_z.subs(rot_3, yaw) * R_y.subs(rot_2, pitch) * R_x.subs(rot_1, roll)
    T_0_eff = T_0_eff.row_insert(3, Matrix([0, 0, 0]).transpose())
    T_0_eff = T_0_eff.col_insert(3, Matrix([ee_x, ee_y, ee_z,1])) * corr
    wrist_position = T_0_eff * Matrix([0.0, 0.0, -0.303, 1.0])
    wc_pos = wrist_position.subs(ik_conf)
    ik_conf.update(dict(zip(x_wc, list(wc_pos[:3]))))
    # list of transformations
    T_s = []
    for alpha, a, d, q in zip(alpha_s, a_s, d_s, q_s):
        T_s.append(transform(alpha, a, d, q).subs(dh))

    theta_1 = atan2(x_wc[1], x_wc[0])
    # Two sides of the triangle come directly from the DH table
    side_b = 1.501 # sqrt(d4 ** 2 + a3 **2)
    side_a = 1.25 # a2
    # point at joint 2:
    x_p2 = (T_s[0] * T_s[1] * origin).subs(q_s[0], theta_1)
    # Distance vector between joint and wrist
    distance = (wrist_center - x_p2)
    # It is simpler to keep the square here. Apply sqrt to cos law
    side_c_sq = simplify((distance.transpose() * distance)[0])
    # apply law of cosines to solve angles:
    alpha = acos((side_b ** 2 + side_c_sq - side_a ** 2) / (2 * side_b * sqrt(side_c_sq)))
    beta = acos((side_a ** 2 + side_c_sq - side_b ** 2) / (2 * side_a * sqrt(side_c_sq)))
    gamma = acos((side_b ** 2 + side_a ** 2 - side_c_sq) / (2 * side_b * side_a))
    # solve relations based on distances and angles
    theta_2 = pi / 2 - beta - atan2(distance[2], sqrt(simplify(distance[0] ** 2 + distance[1] ** 2)))
    theta_3 = pi / 2 - (gamma + 0.036)
    # ik_conf = dict(zip(x_wc, position))
    theta1 = float(theta_1.subs(ik_conf))
    theta2 = float(theta_2.subs(ik_conf))
    theta3 = float(theta_3.subs(ik_conf))
    # update ik_conf with calculated values
    theta_1_3 = (theta1, theta2, theta3)
    ik_conf.update(dict(zip(q_s[:3], theta_1_3)))
    # calculate transform 3-End effector
    # could not get it to work with LU decomposition. Using default method(gauss elim)
    T_3_E = ((T_s[0].subs(ik_conf) * T_s[1].subs(ik_conf) * T_s[2].subs(ik_conf)).inv()[:3, :3] *
              T_0_eff.subs(ik_conf)[:3, :3])
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
    # total transform
    T = Matrix.eye(4)
    for t in T_s:
        T *= t
    
    T_wc = Matrix.eye(4)
    for t in T_s[:4]:
        T_wc *= t
    
    # Correction matrix in homogeneous form
    corr = (R_z.subs(rot_3, pi) * R_y.subs(rot_2, -pi/2))
    corr = corr.row_insert(3, Matrix([0,0,0]).transpose())
    corr = corr.col_insert(3, Matrix([0,0,0,1]))
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




if __name__ == "__main__":
    # Change test case number for different scenarios
    test_case_number = 2

    test_code(test_cases[test_case_number])
