# ICESYCL - A GPU MPM Implementation
This repository contains an experimental/exploratory implementation of the Material Point Method for simulating bulk solids using the recent SYCL GPU acceleration standard. 

# Goals
-Formulate efficient GPU kernels that implement explicit as well as implicit MPM methods
-Implement the required small linear algebra libraries needed for the MPM interpolations as well as Elasto-plastic constitutive models
-Compare performance to CPU implementations using standard C++17 CPU parallelism

# Status and Results (preliminary)
[output.webm](https://github.com/user-attachments/assets/87bf3e25-8b8a-4282-9e93-1a07ffdd853a)


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


