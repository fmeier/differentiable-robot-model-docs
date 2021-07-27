Differentiable Robot Model Documentation
========================================

Overview
-------------

Our differentiable robot model library implements computations such as
forward kinematics and inverse dynamics, in a fully differentiable way. We also allow to specify
parameters (kinematics or dynamics parameters), which can then be identified from data (see examples folder).

Currently, our code should work with any kinematic trees. This package comes with wrappers specifically for:

   * Kuka iiwa
   * Franka Panda
   * Allegro Hand
   * Fetch Arm
   * a 2-link toy robot

Getting Started
-----------------
You can find examples of how to use the library
    * in **examples/run_kinematic_trajectory_opt.py**:
      creating a differentiable model of the Franka Panda and perform kinematic trajectory optimization

    * in **examples/learn_dynamics_iiwa.py**:
      create a differentiable Kuka IIWA model, and make a subset of the dynamics parameters learnable, and learn them from data

    * in **examples/learn_kinematics_of_iiwa.py**:
      create a differentiable Kuka IIWA model, and make a subset of the kinematics parameters learnable, and learn them from data


.. toctree::
   :maxdepth: 2
   :caption: API:

   modules/index




Indices and tables
==================

* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`
