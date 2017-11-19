## Project: Kinematics Pick & Place

---
### General Approach

Most of the description of the project is in the notebook [Writeup.ipynb](https://github.com/albertoferna/RoboND-Kinematics-Project/blob/master/Writeup.ipynb). A would give here a summary of why I went the way I did. There is also an [static html](https://github.com/albertoferna/RoboND-Kinematics-Project/blob/master/Writeup.html) version of the same notebook

When I started trying to put together what we did in the lectures, my main struggle was to make sure I was doing the transformations correctly. The best way I found to approach it was to plot each frame of reference in 3d. Particularly important was to see what the reference points were in the urdf file versus the way frames are defined in DH notation.

With those plots, most of the information needed can be easily found graphically. The only difficulty is to solve a triangle given two sides and the angle between them. That can be easily solve using the law of cosines.

At that point I had all transformations from one frame to the next in my T matrices. It was just a matter of composing them in the right order. And for the orientation of the effector solve the corresponding matrix equation.

### Improvements

I did a first round of optimizations. For that reason I pre-calculated some matrices outside the main loop. There is room for speed improvements for sure because I have just seen several cosines that are repeatedly calculated.

### Final Video

I have added a [video](https://github.com/albertoferna/RoboND-Kinematics-Project/blob/master/Robo_ND_pick_and_place_sim.mp4) of the pick and place gazebo world running in my local setup of ROS.
