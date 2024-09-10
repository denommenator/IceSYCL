//
// Created by robert-denomme on 8/22/24.
//

#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <cmath>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <sycl/sycl.hpp>

#include "utility.hpp"
#include "coordinates.hpp"
#include "interpolation.hpp"
#include "particle_grid_interactions.hpp"
#include "particle_grid_operations.hpp"
#include "collision.h"
#include "constitutive_models.h"



namespace iceSYCL
{

template<class TInterpolationScheme>
class Engine
{
public:
    using InterpolationScheme = TInterpolationScheme;
    using CoordinateConfiguration = typename InterpolationScheme::CoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using NodeIndex_t = typename CoordinateConfiguration::NodeIndex_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    using CoordinateMatrix_t = typename CoordinateConfiguration::CoordinateMatrix_t;
    static constexpr int Dimension = CoordinateConfiguration::Dimension;

public:
    struct ParticleInitialState
    {
        std::vector<Coordinate_t> positions;
        std::vector<Coordinate_t> velocities;
        std::vector<scalar_t> masses;
        std::vector<scalar_t> rest_volumes;
    };

    InterpolationScheme interpolator;
    ParticleGridInteractionManager<InterpolationScheme> pgi_manager;

public:
    static ParticleInitialState MakeUniformDensityInitialState(
        std::vector<Coordinate_t> positions,
        std::vector<Coordinate_t> velocities,
        ParticleGridInteractionManager<InterpolationScheme>& interaction_manager,
        InterpolationScheme interpolator,
        const scalar_t density
        )
    {

        size_t particle_count = positions.size();
        std::vector<scalar_t> particle_rest_volumes(particle_count, 0.0);
        std::vector<scalar_t> particle_masses(particle_count, 0.0);

        //pre-dividing by node volume so the number density calc can use a standard particle-grid-operation
        std::vector<scalar_t> ones_per_particle(particle_count, 1.0 / interpolator.node_volume());
        std::vector<scalar_t> node_number_densities(particle_count * InterpolationScheme::num_interactions_per_particle, 0.0);

        //Determine rest volumes of the particles
        {
            sycl::buffer node_number_densitiesB(node_number_densities);
            sycl::buffer particle_rest_volumesB(particle_rest_volumes);
            sycl::buffer particle_massesB(particle_masses);
            sycl::buffer ones_per_particleB(ones_per_particle);
            sycl::buffer particle_positionsB(positions);

            sycl::queue q;

            auto& pgi_manager = interaction_manager;

            pgi_manager.update_particle_locations(q, particle_positionsB, interpolator);
            particle_grid_operations::transfer_data_particles_to_grid(q, pgi_manager, interpolator, ones_per_particleB, node_number_densitiesB, particle_positionsB, 0.0);
            //node_number_densitiesB actually has number density times volume instead of just number density :-(
            auto interaction_access = pgi_manager.kernel_accessor;
            q.submit([&](sycl::handler& h)
            {
                sycl::accessor particle_mass_acc(particle_massesB, h);
                sycl::accessor node_number_densities_acc(node_number_densitiesB, h);
                sycl::accessor particle_position_acc(particle_positionsB, h);
                sycl::accessor particle_rest_volumes_acc(particle_rest_volumesB, h);
                interaction_access.give_kernel_access(h);

                h.parallel_for(particle_count, [=](sycl::id<1> idx)
                {
                    size_t pid = idx[0];

                    scalar_t number_density_p = 0.0;
                    Coordinate_t x_p = particle_position_acc[pid];
                    for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                        interaction_it != interaction_access.particle_interactions_end(pid);
                        ++interaction_it)
                    {
                        ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                        size_t node_id = interaction.node_id;
                        scalar_t density_i = node_number_densities_acc[node_id];

                        number_density_p += density_i * interpolator.value(interaction.node_index, x_p);

                    }


                    scalar_t volume_p = 1.0 / number_density_p;
                    particle_rest_volumes_acc[pid] = volume_p;
                    particle_mass_acc[pid] = volume_p * density;
                });

            });
            q.wait();
        }


        return {
            positions,
            velocities,
            particle_masses,
            particle_rest_volumes
        };
    }

public:
    struct ParticleData
    {
    private:
        explicit ParticleData(size_t particle_count):
        particle_count{particle_count},
        positions{particle_count},
        velocities{particle_count},
        positions_prev{particle_count},
        velocities_prev{particle_count},
        masses{particle_count},
        rest_volumes{particle_count},
        B_matrices{particle_count},
        D_matrices{particle_count},
        deformation_gradients{particle_count},
        del_v_del_x{particle_count},
        deformation_gradients_prev{particle_count}
        {}
    public:
        static ParticleData InitialStateFactory(ParticleInitialState& initial_state)
        {
            size_t particle_count = initial_state.positions.size();
            ParticleData ret(particle_count);
            iceSYCL::host_copy_all(initial_state.positions, ret.positions);
            iceSYCL::host_copy_all(initial_state.positions, ret.positions_prev);
            iceSYCL::host_copy_all(initial_state.velocities, ret.velocities);
            iceSYCL::host_copy_all(initial_state.velocities, ret.velocities_prev);
            iceSYCL::host_copy_all(initial_state.masses, ret.masses);
            iceSYCL::host_copy_all(initial_state.rest_volumes, ret.rest_volumes);

            CoordinateMatrix_t Identity = CoordinateMatrix_t::Identity();
            iceSYCL::host_fill_all(ret.B_matrices, CoordinateMatrix_t::Zero());
            iceSYCL::host_fill_all(ret.deformation_gradients, CoordinateMatrix_t::Identity());
            iceSYCL::host_fill_all(ret.deformation_gradients_prev, CoordinateMatrix_t::Identity());

            return ret;
        }

        const size_t particle_count;
        sycl::buffer<Coordinate_t> positions;
        sycl::buffer<Coordinate_t> velocities;
        sycl::buffer<Coordinate_t> positions_prev;
        sycl::buffer<Coordinate_t> velocities_prev;
        sycl::buffer<scalar_t> masses;
        sycl::buffer<scalar_t> rest_volumes;
        sycl::buffer<CoordinateMatrix_t> B_matrices;
        sycl::buffer<CoordinateMatrix_t> D_matrices;
        sycl::buffer<CoordinateMatrix_t> deformation_gradients;
        sycl::buffer<CoordinateMatrix_t> del_v_del_x;
        sycl::buffer<CoordinateMatrix_t> deformation_gradients_prev;



    };

    ParticleData particle_data;
    const size_t particle_count;
public:
    Engine(InterpolationScheme interpolator_instance, ParticleGridInteractionManager<InterpolationScheme>&& interaction_manager_instance, ParticleData&& particle_data, sycl::buffer<ElasticCollisionWall<CoordinateConfiguration>>&& walls) :
    particle_count{particle_data.particle_count},
    interpolator{interpolator_instance},
    pgi_manager{std::move(interaction_manager_instance)},
    particle_data{std::move(particle_data)},
    node_data{particle_count * InterpolationScheme::num_interactions_per_particle},
    descent_data{node_data.max_node_count},
    collision_walls{std::move(walls)}
    {}

    static Engine FromInitialState(
        InterpolationScheme interpolator,
        std::vector<Coordinate_t> positions,
        std::vector<Coordinate_t> velocities,
        std::vector<ElasticCollisionWall<CoordinateConfiguration>> walls)
    {
        ParticleGridInteractionManager<InterpolationScheme> pgi_manager(positions.size());
        const scalar_t unit_density = 1.0;
        ParticleInitialState initial_state = MakeUniformDensityInitialState(positions, velocities, pgi_manager,
                                                                            interpolator, unit_density);
        sycl::buffer<ElasticCollisionWall<CoordinateConfiguration>> walls_b(walls.size());
        host_copy_all(walls, walls_b);

        return Engine(interpolator, std::move(pgi_manager), std::move(ParticleData::InitialStateFactory(initial_state)), std::move(walls_b));
    }
public:
    struct NodeData
    {
        explicit NodeData(size_t max_node_count) :
        max_node_count{max_node_count},
        masses{max_node_count},
        momenta{max_node_count},
        predicted_positions{max_node_count},
        inertial_positions{max_node_count},
        velocities{max_node_count}
        {}
        const size_t max_node_count;
        sycl::buffer<scalar_t> masses;
        sycl::buffer<Coordinate_t> momenta;
        sycl::buffer<Coordinate_t> predicted_positions;
        sycl::buffer<Coordinate_t> inertial_positions;
        sycl::buffer<Coordinate_t> velocities;
    };

    NodeData node_data;

    struct DescentData
    {
        explicit DescentData(size_t max_node_count) :
        max_node_count{max_node_count},
        descent_direction{max_node_count},
        descent_direction_prev{max_node_count},
        gradient{max_node_count},
        gradient_prev{max_node_count}
        {}
        const size_t max_node_count;
        sycl::buffer<Coordinate_t> descent_direction;
        sycl::buffer<Coordinate_t> descent_direction_prev;
        sycl::buffer<Coordinate_t> gradient;
        sycl::buffer<Coordinate_t> gradient_prev;
    };

    DescentData descent_data;

    sycl::buffer<ElasticCollisionWall<CoordinateConfiguration>> collision_walls;

public:
    template<typename ConstitutiveModel>
    void step_frame(const ConstitutiveModel Psi, const size_t num_steps_per_frame = 50, const double mu_velocity_damping = 1.0, const double gravity = 981.0)
    {step_frame_explicit(Psi, num_steps_per_frame, mu_velocity_damping, gravity);}
    template<typename ConstitutiveModel>
    void step_frame_explicit(const ConstitutiveModel Psi, const size_t num_steps_per_frame = 50, const double mu_velocity_damping = 1.0, const double gravity = 981.0);
    void transer_mass_particles_to_nodes(sycl::queue& q);
    void transfer_momentum_particles_to_nodes_APIC(sycl::queue& q);
    void apply_particle_forces_to_grid(sycl::queue& q, sycl::buffer<ElasticCollisionWall<CoordinateConfiguration>>& walls, scalar_t dt, const scalar_t gravity);
    template<typename ConstitutiveModel>
    void apply_mpm_hyperelastic_forces_to_grid(sycl::queue& q, const ConstitutiveModel Psi, const scalar_t dt);
    void compute_node_velocities(sycl::queue& q);
    void transfer_velocity_nodes_to_particles(sycl::queue& q);
    void transfer_velocity_nodes_to_particles_APIC(sycl::queue& q);
    void update_particle_deformation_gradients(sycl::queue& q, scalar_t dt);


    void setup_descent_objective(sycl::queue& q, const scalar_t dt);
    template<typename ConstitutiveModel>
    void step_frame_implicit(const ConstitutiveModel Psi, const size_t num_steps_per_frame = 50, const double mu_velocity_damping = 1.0);

    void update_particle_deformation_gradients_implicit(sycl::queue& q, scalar_t dt);
    template<typename ConstitutiveModel>
    void compute_descent_gradient(sycl::queue& q, const ConstitutiveModel Psi, scalar_t dt, const double gravity);

};

}


#include "engine_impl.ipp"
#include "descent_impl.ipp"

#endif //ENGINE_HPP
