# ICESYCL - A GPU MPM Implementation
This repository contains an experimental/exploratory implementation of the Material Point Method for simulating bulk solids using the recent SYCL GPU acceleration standard. 

# Goals
-Formulate efficient GPU kernels that implement explicit as well as implicit MPM methods
-Implement the required small linear algebra libraries needed for the MPM interpolations as well as Elasto-plastic constitutive models
-Compare performance to CPU implementations using standard C++17 CPU parallelism

# Status and Results (preliminary)

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


