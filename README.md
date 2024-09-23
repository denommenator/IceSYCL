# ICESYCL - A GPU MPM Implementation
This repository contains an experimental/exploratory implementation of the Material Point Method for simulating bulk solids using the recent SYCL GPU acceleration standard. 

# Goals
-Formulate efficient GPU kernels that implement explicit as well as implicit MPM methods
-Implement the required small linear algebra libraries needed for the MPM interpolations as well as Elasto-plastic constitutive models
-Compare performance to CPU implementations using standard C++17 CPU parallelism

# Status and Results (preliminary)

[Screencast from 2024-09-23 14-50-43_small_trim.webm](https://github.com/user-attachments/assets/a0387a7c-0052-42fd-acd5-717c8ed66f3c)

A fluid simulation using fixed corotated constitutive model with 0 shear modulus. 3,700 particles simulated with fully implicit time integration in real time. 

[Screencast from 2024-09-20 16-22-29.webm](https://github.com/user-attachments/assets/0865a29e-a54d-4057-83ac-33fd826a9686)



## Constitutive Models Implemented:
- Ideal Gas Law
- Tait Equation of State (A stiff compressive model of water)
- Fixed Corotated Model*
- Snow Elasto-Plasticity*
(* = requisite SVD implementation is currently missing for 3D)

## Time integration Schemes
- Semi-implicit forwards euler for elastic and elasto-plastic
- Descent based (Fletcher-Reeves) backwards euler for elastic models

## Collisions
- Under Development

## Momentum Transfer
- Both standard and APIC transfers of momentum are implemented


