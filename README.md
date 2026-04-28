# Otter USV – API, Control Methods, and Simulator Extensions

This repository combines work from several courses and a master thesis, all centered around the Otter Unmanned Surface Vehicle (USV). It provides an Application Programming Interface (API), multiple control approaches, and modified simulation environments. The project is built on top of the Python Vehicle Simulator developed by Thor I. Fossen (Autumn 2023 version), and it expands that framework for academic experimentation, controller development, and reinforcement learning research.

---

## Repository Overview

### ELVE 3160 – Introduction to Robotics
Provides an API for interacting with the Otter USV. It includes modules for accessing sensor data, maintaining an updated internal state, computing derived quantities, and supporting the development of custom control algorithms. The structure is designed to be easy to read and extend, reflecting the original assignment goals.

---

### ACIT 4830 – Special Robotics and Control
Introduces Deep Reinforcement Learning (DRL) for USV control.  

Its contributions include:

- A DRL version of the Otter simulator (`Otter_simulator_DRL`)  
- A Gym-like environment for training  
- Implementations based on PPO and related DRL methods  
- Normalized observations and action spaces  
- Modified stepwise simulation logic for RL interaction  

All DRL-related work is contained under the `Otter_dl` directory, with simulator adjustments stored in the corresponding `Otter_simulator_DRL` folder.

---

### Master Thesis Contributions
Builds upon the course foundations and integrates the Otter API with the full simulation and control framework. The thesis introduces significant extensions in control, environment modeling, and learning-based methods.

Key contributions include:

#### Control and Reference Models
- Surge and yaw PID control for target tracking  
- Foundations for Nonlinear Model Predictive Control (NMPC)  
- **Third-order trajectory reference models** for smooth position, velocity, and acceleration profiles  

#### Environmental Modeling
- **Wind disturbance model** for simulating external forces acting on the vessel  
- **Bretschneider wave spectrum model** for realistic irregular sea states and wave-induced disturbances  

#### Deep Reinforcement Learning Enhancements
- Custom **reward functions** for navigation and tracking  
- Tuned **DRL hyperparameters** for stable and efficient learning  
- **Training randomization strategies** to improve robustness and generalization  
- Extended DRL-compatible simulator with improved step interaction and state handling  

#### System Integration
- Unified framework combining API, simulator, and control methods  
- Improved data structures and state management for analysis and debugging  
- Scenario handling and pattern generation for testing  

---

## Control Methods Included in the Repository

### Proportional–Integral–Derivative (PID) Control
Supports traditional PID-based surge and yaw controllers, built using the API’s state dictionary and the underlying Fossen model for dynamics.

### Deep Reinforcement Learning (DRL)
The DRL framework includes:

- Vectorized environments for parallel training  
- Custom reward shaping for control objectives  
- Policy structures compatible with modern RL libraries  
- Step-based simulation interface for training  

### Nonlinear Model Predictive Control (NMPC)
Contains the foundations needed for NMPC experiments on the Otter USV model. Cost functions, horizon updates, and constraints compatible with the dynamics from Fossen’s simulator reduced to 3DOF.

---

## Controller Deployment

The implemented controllers are designed to be platform-independent and can be used in multiple setups:

- **Simulation environment** (Python Vehicle Simulator and DRL extensions)  
- **Socket-based communication interface** for real-time or external system integration  

This allows the same control logic to be reused across simulation and real-time applications.

---

## Otter API

The API is designed to simplify access to the vessel’s state and control interface. Its responsibilities include:

- Collecting data from the simulator or GPS at each update  
- Maintaining a structured dictionary with positions, velocities, attitudes, and derived values  
- Handling control allocation by exposing functions for sending thrusts  
- Offering helper methods for tasks such as resetting, logging, and managing simulation steps  

The API acts as the central layer connecting control algorithms with the underlying simulator.

---

## Simulator Foundation

All simulation components are based on the **Python Vehicle Simulator (Autumn 2023)** by Thor I. Fossen.  

The simulator provides:

- Full six-degree-of-freedom vessel dynamics  
- Hydrodynamic and hydrostatic forces  
- Added mass and damping  
- Environmental effects  

This repository extends the simulator with:

- Wind disturbance modeling  
- Wave excitation using the Bretschneider spectrum  
- Third-order reference trajectory generation  
- RL-compatible stepping and control interaction  
- Detailed state tracking and improved access to intermediate simulation values  

These extensions make the simulator suitable for realistic marine control experiments and advanced research scenarios.
