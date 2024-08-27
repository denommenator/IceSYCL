//
// Created by robert-denomme on 8/27/24.
//

#ifndef NATIVE_API_HPP
#define NATIVE_API_HPP

#include <IceSYCL/engine.hpp>
#include "buffer_helpers.hpp"

//#include <iostream>

#define EXPORT_API extern "C"

struct ParticleState
{
    double* positions;
    double* velocities;
    size_t particle_count;

};

using Engine2D = iceSYCL::Engine<iceSYCL::CubicInterpolationScheme<iceSYCL::Double2DCoordinateConfiguration>>;

EXPORT_API Engine2D* create_2D_engine(
    const ParticleState* initial_state,
    const double* masses,
    const double h)
{
    using namespace iceSYCL;
    using CoordinateConfiguration = Engine2D::CoordinateConfiguration;
    using Coordinate_t = Engine2D::Coordinate_t;
    using scalar_t = Engine2D::scalar_t;

    const size_t particle_count = initial_state->particle_count;

    const std::vector<Coordinate_t> positions_vec = to_coordinate_vector<CoordinateConfiguration>(particle_count, initial_state->positions);
    const std::vector<Coordinate_t> velocities_vec = to_coordinate_vector<CoordinateConfiguration>(particle_count, initial_state->velocities);
    const std::vector<scalar_t> masses_vec = to_scalar_vector<CoordinateConfiguration>(particle_count, masses);

    Engine2D engine = Engine2D::FromInitialState(
        Engine2D::InterpolationScheme(h),
        positions_vec,
        velocities_vec,
        masses_vec
        );

    return new Engine2D(engine);
}

// EXPORT_API void copy_current_state(Abominable::Engine* engine, double* current_state_raw_ptr)
// {
//     size_t particle_count = engine->get_current_particles().positions_.index_count();
//     for(size_t pid = 0; pid < particle_count; pid++)
//     {
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
//     }
//
//     //current_state_raw_ptr[0] = particle_count;
// }
//
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
//
// EXPORT_API void step_frame(Abominable::Engine* engine)
// {
//     engine->step_frame();
// }


EXPORT_API void delete_engine(Engine2D* engine)
{
    engine->~Engine();
}


#endif //NATIVE_API_HPP
