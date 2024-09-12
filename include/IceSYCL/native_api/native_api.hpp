//
// Created by robert-denomme on 8/27/24.
//

#ifndef NATIVE_API_HPP
#define NATIVE_API_HPP

#include <IceSYCL/engine.hpp>
#include <IceSYCL/constitutive_models.h>
#include "buffer_helpers.hpp"
#include <small_la/small_matrix.hpp>

//#include <iostream>

#define EXPORT_API extern "C"
//
// struct ParticleState
// {
//     double* positions;
//     double* velocities;
//     size_t particle_count;
//
// };

using Engine2D = iceSYCL::Engine<iceSYCL::CubicInterpolationScheme<iceSYCL::Double2DCoordinateConfiguration>>;

EXPORT_API Engine2D* create_engine(
    size_t particle_count,
    const double* positions,
    const double* velocities,
    const double h, const double wall_stiffness)
{
    using namespace iceSYCL;
    using namespace raw_buffer_utility;
    using CoordinateConfiguration = Engine2D::CoordinateConfiguration;
    using Coordinate_t = Engine2D::Coordinate_t;
    using scalar_t = Engine2D::scalar_t;
    using Wall_t = ElasticCollisionWall<CoordinateConfiguration>;

    const std::vector<Coordinate_t> positions_vec = to_coordinate_vector<CoordinateConfiguration>(particle_count, positions);
    const std::vector<Coordinate_t> velocities_vec = to_coordinate_vector<CoordinateConfiguration>(particle_count, velocities);

    std::vector<ElasticCollisionWall<CoordinateConfiguration>> walls = {
            Wall_t{Coordinate_t(0.0, 1.0), Coordinate_t(0.0, -50.0), wall_stiffness},
            Wall_t{Coordinate_t(1.0, 0.0), Coordinate_t(-100.0, 0.0), wall_stiffness},
            Wall_t{Coordinate_t(-1.0, 0.0), Coordinate_t(100.0, 0.0), wall_stiffness}
    };

    Engine2D engine = Engine2D::FromInitialState(
        Engine2D::InterpolationScheme(h),
        positions_vec,
        velocities_vec,
        walls
        );

    return new Engine2D(engine);
}

EXPORT_API void copy_current_positions(Engine2D* engine, double* positions_raw_pointer)
{
    using Coordinate_t = Engine2D::Coordinate_t;
    size_t particle_count = engine->particle_count;
    {
        sycl::host_accessor positions_acc(engine->particle_data.positions);
        raw_buffer_utility::copy<iceSYCL::Double2DCoordinateConfiguration, sycl::host_accessor<Coordinate_t>>(positions_acc, positions_raw_pointer);
    }

    //current_state_raw_ptr[0] = particle_count;
}

// EXPORT_API void copy_deformation_gradients(Abominable::Engine* engine, double* deformation_gradients_raw_ptr)
// {
//     Abominable::ParticleMatrix& particle_deformation_gradients = engine->get_particle_deformation_gradients();
//     size_t particle_count = particle_deformation_gradients.index_count();
//     for(auto iter = particle_deformation_gradients.begin(); iter != particle_deformation_gradients.end(); iter++)
//     {
//         size_t pid = iter.id();
//         Abominable::CoordinateMatrix& F = *iter;
//
//         deformation_gradients_raw_ptr[Abominable::Dimension * pid * 2 + 0] = F(0,0);
//         deformation_gradients_raw_ptr[Abominable::Dimension * pid * 2 + 1] = F(1,0);
//         deformation_gradients_raw_ptr[Abominable::Dimension * pid * 2 + 2] = F(0,1);
//         deformation_gradients_raw_ptr[Abominable::Dimension * pid * 2 + 3] = F(1,1);
//         /*
//         Abominable::Coordinate p = engine->get_current_particles().positions_.coordinate(pid);
//         Abominable::Coordinate v = engine->get_current_particles().velocities_.coordinate(pid);
//
//         current_state_raw_ptr[Abominable::Dimension * pid + 0] = p(0);
//         current_state_raw_ptr[Abominable::Dimension * pid + 1] = p(1);
//
//         //std::cout << p(0) << " " << p(1) << std::endl;
//
//         current_state_raw_ptr[Abominable::Dimension * particle_count + Abominable::Dimension * pid + 0] = v(0);
//         current_state_raw_ptr[Abominable::Dimension * particle_count + Abominable::Dimension * pid + 1] = v(1);
//         */
//     }
//
//     //current_state_raw_ptr[0] = particle_count;
//  }

//EXPORT_API void step_frame(Engine2D* engine, int num_steps_per_frame, double mu_constitutive, double lambda_constitutive, double mu_damping, double gravity)
//{
//    using namespace iceSYCL;
//    using CoordinateConfiguration = Engine2D::CoordinateConfiguration;
//    //using ConstitutiveModel = DensityBasedConstitutiveModel<TaitPressureFromDensity<CoordinateConfiguration>>;
//
//    using ConstitutiveModel = FixedCorotated<CoordinateConfiguration>;
//    ConstitutiveModel Psi{FixedCorotated<CoordinateConfiguration>{mu_constitutive, lambda_constitutive}};
//
////    using ConstitutiveModel = DensityBasedConstitutiveModel<IdealGasFromDensity<CoordinateConfiguration>>;
////    ConstitutiveModel Psi{IdealGasFromDensity<CoordinateConfiguration>{1.0, c_speed_of_sound}};
//
//
//    engine->step_frame(Psi, num_steps_per_frame, mu_damping, gravity);
//}

EXPORT_API void step_frame_implicit(Engine2D* engine, int num_steps_per_frame, int num_descent_steps, double mu_constitutive, double lambda_constitutive, double mu_damping, double gravity)
{
    using namespace iceSYCL;
    using CoordinateConfiguration = Engine2D::CoordinateConfiguration;
    //using ConstitutiveModel = DensityBasedConstitutiveModel<TaitPressureFromDensity<CoordinateConfiguration>>;

    using ConstitutiveModel = FixedCorotated<CoordinateConfiguration>;
    ConstitutiveModel Psi{FixedCorotated<CoordinateConfiguration>{mu_constitutive, lambda_constitutive}};

//    using ConstitutiveModel = DensityBasedConstitutiveModel<IdealGasFromDensity<CoordinateConfiguration>>;
//    ConstitutiveModel Psi{IdealGasFromDensity<CoordinateConfiguration>{1.0, c_speed_of_sound}};


    engine->step_frame_implicit(Psi, num_steps_per_frame, num_descent_steps, mu_damping, gravity);
}


EXPORT_API void delete_engine(Engine2D* engine)
{
    engine->~Engine();
}


#endif //NATIVE_API_HPP
