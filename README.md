# Otter USV – API, Control Methods, and Simulator Extensions

This repository combines work from several courses and a bachelor thesis, all centered around the Otter Unmanned Surface Vehicle (USV). It provides an Application Programming Interface (API), multiple control approaches, and modified simulation environments. The project is built on top of the Python Vehicle Simulator developed by [Thor I. Fossen (Autumn 2023 version)](https://github.com/cybergalactic/PythonVehicleSimulator), and it expands that framework for academic experimentation, controller development, and reinforcement learning research.

---

## Repository Overview

### ELVE 3160 – Introduction to Robotics
Provides an API for interacting with the Otter USV. It includes modules for accessing sensor data, maintaining an updated internal state, computing derived quantities, and supporting the development of custom control algorithms. The structure is designed to be easy to read and extend, reflecting the original assignment goals.

---

### Bachelor Thesis Contributions
Integrates the Otter API with a full simulation setup. It includes support for pattern creation, scenario handling, and extended data structures for analyzing vessel behavior. The thesis version preserves the modular layout from the course but adds refinement in control flow, data management, and state updating.

---

### ACIT 4830 – Special Robotics and Control
Deep Reinforcement Learning (DRL) for USV control.  
Its contributions include:

- A DRL version of the Otter simulator (`Otter_simulator_DRL`)  
- A Gym-like environment for training  
- Implementations based on PPO and related DRL methods  
- Normalized observations and action spaces  
- Modified stepwise simulation logic for RL interaction  

All DRL-related work is contained under the `Otter_dl` directory, with simulator adjustments stored in the corresponding `Otter_simulator_DRL` folder.

---

## Control Methods Included in the Repository

### Proportional–Integral–Derivative (PID) Control
Supports traditional PID-based surge and yaw controllers, built using the API’s state dictionary and the underlying Fossen model for dynamics.

### Deep Reinforcement Learning (DRL)
The DRL section includes:

- A vectorized environment for training multiple parallel simulations  
- Custom reward shaping functions for navigation and tracking tasks  
- Policy structures compatible with modern RL libraries  
- Modifications enabling step-by-step interaction rather than continuous simulation  

These components provide a platform for experiments in autonomous control and policy learning.

### Nonlinear Model Predictive Control (NMPC)
Contains the foundations needed for NMPC experiments on the Otter USV model. Cost functions, horizon updates, and constraints compatible with the dynamics from Fossen’s simulator reduced to 3DOF.

---

## Otter API

The API is designed to simplify access to the vessel’s state and control interface. Its responsibilities include:

- Collecting data from the simulator or gps at each update  
- Maintaining a structured dictionary with positions, velocities, attitudes, and derived values  
- Handling control allocation by exposing functions for sending thrusts  
- Offering helper methods for tasks such as resetting, logging, and managing simulation steps  

The API acts as the central layer connecting control algorithms with the underlying simulator.

---

## Simulator Foundation

All simulation components are based on the **Python Vehicle Simulator (Autumn 2023)** by Thor I. Fossen.  
The simulator provides:

- Full six–degree–of–freedom vessel dynamics  
- Hydrodynamic and hydrostatic forces  
- Added mass and damping  
- Environmental effects  
- Control allocation and thruster models  

This repository extends the simulator with additional features such as RL-compatible stepping, detailed state tracking, and easier access to intermediate simulation values.

---
