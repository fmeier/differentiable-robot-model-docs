# Tutorials


## Using the Differentiable Robot Model as a ground truth model

As with most simulated robots, a robot model can be created from an URDF file.
Officially supported robot URDF files can also be found in `diff_robot_data/`.
```py
from differentiable_robot_model.robot_model import  DifferentiableRobotModel

urdf_path = "path/to/robot/urdf"
robot = DifferentiableRobotModel(urdf_path)
```

Once the robot model has been successfully instatiated with the URDF, we now have access to the properties and rigid body mechanics of the robot.
The following example assumes that the robot model is instatiated with a 7 degree-of-freedom Kuka iiwa arm URDF:
```py
# Values to query the model with
joint_pos = torch.rand(7)
joint_vel = torch.rand(7)
joint_acc_desired = torch.rand(7)
torques = torch.rand(7)
ee_link_name = "iiwa_link_ee"

# Robot properties
robot.get_joint_limits()
robot.get_link_names()

# Robot kinematics
ee_pos, ee_quat = robot.compute_forward_kinematics(joint_pos, ee_link_name)
J_linear, J_angular = robot.compute_endeffector_jacobian(joint_pos, ee_link_name)

# Robot dynamics
joint_acc = robot.compute_forward_dynamics(joint_pos, joint_vel, torques)
torques_desired = robot.compute_inverse_dynamics(joint_pos, joint_vel, joint_acc_desired)
```

For more details see the [API docs](modules.diff_robot_model).


## Differentiating through the Differentiable Robot Model



## Learning the parameters of the Differentiable Robot Model

