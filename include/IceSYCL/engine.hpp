//
// Created by robert-denomme on 8/22/24.
//

#ifndef ENGINE_HPP
#define ENGINE_HPP


#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include <sycl/sycl.hpp>

#include "utility.hpp"
#include "coordinates.hpp"
#include "interpolation.hpp"
#include "particle_grid_interactions.hpp"
#include "particle_grid_operations.hpp"



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
    static ParticleInitialState MakeInitialState(
        std::vector<Coordinate_t> positions,
        std::vector<Coordinate_t> velocities,
        std::vector<scalar_t> masses,
        ParticleGridInteractionManager<InterpolationScheme>& interaction_manager,
        InterpolationScheme interpolator
        )
    {

        size_t particle_count = positions.size();
        std::vector<scalar_t> particle_rest_volumes(particle_count, 0.0);
        std::vector<scalar_t> node_masses(particle_count * InterpolationScheme::num_interactions_per_particle, 0.0);

        //Determine rest volumes of the particles
        {
            sycl::buffer node_massesB(node_masses);
            sycl::buffer particle_rest_volumesB(particle_rest_volumes);
            sycl::buffer particle_massesB(masses);
            sycl::buffer particle_positionsB(positions);

            sycl::queue q;

            auto& pgi_manager = interaction_manager;

            pgi_manager.update_particle_locations(q, particle_positionsB, interpolator);
            particle_grid_operations::transfer_data_particles_to_grid(q, pgi_manager, interpolator, particle_massesB, node_massesB, particle_positionsB, 0.0);

            auto interaction_access = pgi_manager.kernel_accessor;
            q.submit([&](sycl::handler& h)
            {
                sycl::accessor particle_mass_acc(particle_massesB, h);
                sycl::accessor node_mass_acc(node_massesB, h);
                sycl::accessor particle_position_acc(particle_positionsB, h);
                sycl::accessor particle_rest_volumes_acc(particle_rest_volumesB, h);
                interaction_access.give_kernel_access(h);

                h.parallel_for(particle_count, [=](sycl::id<1> idx)
                {
                    size_t pid = idx[0];

                    scalar_t mass_p = particle_mass_acc[pid];
                    Coordinate_t x_p = particle_position_acc[pid];
                    scalar_t denominator = 0.0;
                    for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                        interaction_it != interaction_access.particle_interactions_end(pid);
                        ++interaction_it)
                    {
                        ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                        size_t node_id = interaction.node_id;
                        scalar_t mass_i = node_mass_acc[node_id];

                        denominator += mass_i * interpolator.value(interaction.node_index, x_p);
                    }
                    particle_rest_volumes_acc[pid] = mass_p * interpolator.node_volume() / denominator;
                });

            });
            q.wait();
        }


        return {
            positions,
            velocities,
            masses,
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
    Engine(InterpolationScheme interpolator_instance, ParticleGridInteractionManager<InterpolationScheme>&& interaction_manager_instance, ParticleData&& particle_data) :
    particle_count{particle_data.particle_count},
    interpolator{interpolator_instance},
    pgi_manager{std::move(interaction_manager_instance)},
    particle_data{std::move(particle_data)},
    node_data{}
    {}

    static Engine FromInitialState(
        InterpolationScheme interpolator,
        std::vector<Coordinate_t> positions,
        std::vector<Coordinate_t> velocities,
        std::vector<scalar_t> masses)
    {
        ParticleNodeInteraction<CoordinateConfiguration> interaction_manager(positions.size());
        ParticleInitialState initial_state = MakeInitialState(positions, velocities, masses, interaction_manager);

        return Engine(interpolator, interaction_manager, ParticleData::InitialStateFactory(initial_state));
    }
public:
    struct NodeData
    {
        explicit NodeData(size_t max_node_count) :
        max_node_count{max_node_count},
        masses{max_node_count},
        momenta{max_node_count},
        velocities{max_node_count}
        {}
        const size_t max_node_count;
        sycl::buffer<scalar_t> masses;
        sycl::buffer<Coordinate_t> momenta;
        sycl::buffer<Coordinate_t> velocities;
    };

    NodeData node_data;

public:
    void step_frame(sycl::queue& q);
    void transer_mass_particles_to_nodes(sycl::queue& q);
    void transfer_momentum_particles_to_nodes_APIC(sycl::queue& q);
    void apply_particle_forces_to_grid(sycl::queue& q, scalar_t dt);
    void apply_mpm_hyperelastic_forces_to_grid(sycl::queue& q);
    void compute_node_velocities(sycl::queue& q);
    void transfer_velocity_nodes_to_particles(sycl::queue& q);
    void transfer_velocity_nodes_to_particles_APIC(sycl::queue& q);
    void update_particle_deformation_gradients(sycl::queue& q, scalar_t dt);

};

}


#include "engine_impl.ipp"

#endif //ENGINE_HPP
