# Tutorials

As with most simulated robots, a robot model can be created from an URDF file.
Officially supported robot URDF files can also be found in `diff_robot_data/`.
```py
from differentiable_robot_model.robot_model import DifferentiableRobotModel

urdf_path = "path/to/robot/urdf"
robot = DifferentiableRobotModel(urdf_path)
```

For the remainder of the tutorial, we will assume that the robot model is instatiated with a 7 degree-of-freedom Kuka iiwa arm URDF, which can be found at `diff_robot_data/kuka_iiwa/urdf/iiwa7.urdf`.


## Using the Differentiable Robot Model as a ground truth model

Once the robot model has been successfully instatiated with the URDF, we now have access to the properties and rigid body mechanics of the robot.
```py
import torch

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


## Learning the parameters of the Differentiable Robot Model

The class `DifferentialRobotModel` is actually derived from [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), and thus gradients of the inputs and/or parameters can be obtained as with any other Pytorch module. 
This allows users to differentiate through kinematics/dynamics calls.
```py
# The following is equivalent to robot.compute_jacobian(joint_pos, ee_link_name)[0:3, :]
ee_pos, ee_quat = robot.compute_forward_kinematics(joint_pos, ee_link_name)
pos_jacobian = torch.autograd.grad(ee_pos, joint_pos)
```
The example in `examples/run_kinematic_trajectory_opt.py` demonstrates a trajectory optimization algorithm by differentiating through the model kinematics in a similar manner.

By assigning parametrizations to model parameters, we can also directly learn the model parameters. Several parametrizations schemes are provided in `differentiable_robot_model/rigid_body_params.py`.
```py
# potential mass parametrizations
from differentiable_robot_model.rigid_body_params import (
    UnconstrainedScalar,
    PositiveScalar,
    UnconstrainedTensor,
)

# potential inertia matrix parametrizations
from differentiable_robot_model.rigid_body_params import (
    CovParameterized3DInertiaMatrixNet,
    Symm3DInertiaMatrixNet,
    SymmPosDef3DInertiaMatrixNet,
    TriangParam3DInertiaMatrixNet,
)

robot.make_link_param_learnable(
"iiwa_link_1", "mass", PositiveScalar()
)
robot.make_link_param_learnable(
"iiwa_link_1", "com", UnconstrainedTensor(dim1=1, dim2=3)
)
robot.make_link_param_learnable(
"iiwa_link_1", "inertia_mat", UnconstrainedTensor(dim1=3, dim2=3)
)
```


## Putting it all together
The following code snippet shows how to learn parameters of a link in a robot model using data from a ground truth model.
This example script can also be found in `examples/learn_forward_dynamics_iiwa.py`.

```py
import numpy as np
import os
import torch
import random
from torch.utils.data import DataLoader

from differentiable_robot_model.robot_model import DifferentiableKUKAiiwa
from differentiable_robot_model.data_utils import (
    generate_sine_motion_forward_dynamics_data,
)
import diff_robot_data


class NMSELoss(torch.nn.Module):
    def __init__(self, var):
        super(NMSELoss, self).__init__()
        self.var = var

    def forward(self, yp, yt):
        err = (yp - yt) ** 2
        werr = err / self.var
        return werr.mean()

# Setup learnable robot model
urdf_path = os.path.join(diff_robot_data.__path__[0], "kuka_iiwa/urdf/iiwa7.urdf")

learnable_robot_model = DifferentiableRobotModel(
urdf_path, "kuka_iiwa", device=device
)
learnable_robot_model.make_link_param_learnable(
"iiwa_link_1", "mass", PositiveScalar()
)
learnable_robot_model.make_link_param_learnable(
"iiwa_link_1", "com", UnconstrainedTensor(dim1=1, dim2=3)
)
learnable_robot_model.make_link_param_learnable(
"iiwa_link_1", "inertia_mat", UnconstrainedTensor(dim1=3, dim2=3)
)

# Generate training data via ground truth model
gt_robot_model = DifferentiableKUKAiiwa(device=device)

train_data = generate_sine_motion_forward_dynamics_data(
gt_robot_model, n_data=n_data, dt=1.0 / 250.0, freq=0.1
)
train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=False)

# Optimize learnable params
optimizer = torch.optim.Adam(learnable_robot_model.parameters(), lr=1e-2)
loss_fn = NMSELoss(train_data.var())
for i in range(n_epochs):
losses = []
for batch_idx, batch_data in enumerate(train_loader):
	q, qd, qdd, tau = batch_data
	optimizer.zero_grad()
	qdd_pred = learnable_robot_model.compute_forward_dynamics(
	q=q, qd=qd, f=tau, include_gravity=True, use_damping=True
	)
	loss = loss_fn(qdd_pred, qdd)
	loss.backward()
	optimizer.step()
	losses.append(loss.item())

print(f"i: {i} loss: {np.mean(losses)}")
```
