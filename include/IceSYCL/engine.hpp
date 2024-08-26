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
        std::vector<Coordinate_t> masses)
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
    void transer_mass_particles_to_nodes(sycl::queue& q);
    void transfer_momentum_particles_to_nodes_APIC(sycl::queue& q);
    void apply_particle_forces_to_grid(sycl::queue& q);
    void apply_mpm_hyperelastic_forces_to_grid(sycl::queue& q);
    void compute_node_velocities(sycl::queue& q);
    void transfer_velocity_nodes_to_particles(sycl::queue& q);
    void transfer_velocity_nodes_to_particles_APIC(sycl::queue& q);
    void update_particle_deformation_gradients(sycl::queue& q, scalar_t dt);

};


template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::transer_mass_particles_to_nodes(sycl::queue& q)
{
    particle_grid_operations::transfer_data_particles_to_grid(q,
        pgi_manager,
        interpolator,
        particle_data.masses,
        node_data.masses,
        particle_data.positions,
        0.0);
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::transfer_momentum_particles_to_nodes_APIC(sycl::queue& q)
{

    auto interaction_access = pgi_manager.kernel_accessor;

    //particles
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor particle_mass_acc(particle_data.masses, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);
        sycl::accessor particle_D_matrices_acc(particle_data.D_matrices, h);
        interaction_access.give_kernel_access(h);

        h.parallel_for(particle_count,[=](sycl::id<1> idx)
        {
            size_t pid = idx[0];

            Coordinate_t x_p = particle_positions_acc[pid];

            CoordinateMatrix_t D_p = CoordinateMatrix_t::Zero();
            for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                interaction_it != interaction_access.particle_interactions_end(pid);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;


                NodeIndex_t node_index = interaction.node_index;
                scalar_t n_value = interpolator.value(node_index, x_p);

                D_p += n_value * (interpolator.position(node_index) - x_p) * (interpolator.position(node_index) - x_p).transpose();
            }

            particle_D_matrices_acc[pid] = D_p;
        });
    });

    //nodes
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor particle_mass_acc(particle_data.masses, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);
        sycl::accessor particle_velocity_acc(particle_data.velocities, h);
        sycl::accessor particle_D_matrices_acc(particle_data.D_matrices, h);
        sycl::accessor particle_B_matrices_acc(particle_data.B_matrices, h);
        sycl::accessor node_momenta_acc(node_data.momenta, h);

        interaction_access.give_kernel_access(h);

        h.parallel_for(node_data.max_node_count,[=](sycl::id<1> idx)
        {
            const size_t node_count = interaction_access.node_count();
            const size_t node_id = idx[0];
            if(node_id >= node_count)
                return;

            Coordinate_t momentum_i = Coordinate_t::Zero();
            for(auto interaction_it = interaction_access.node_interactions_begin(node_id);
                interaction_it != interaction_access.node_interactions_end(node_id);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                size_t pid = interaction.particle_id;
                scalar_t mass_p = particle_mass_acc[pid];
                Coordinate_t x_p = particle_positions_acc[pid];
                Coordinate_t v_p = particle_velocity_acc[pid];
                CoordinateMatrix_t B_p = particle_B_matrices_acc[pid];
                CoordinateMatrix_t D_p = particle_D_matrices_acc[pid];

                NodeIndex_t node_index = interaction.node_index;

                momentum_i += interpolator.value(node_index, x_p) * mass_p * (v_p + B_p * D_p.inverse() * (interpolator.position(node_index) - x_p));
            }

            node_momenta_acc[node_id] = momentum_i;
        });
    });
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::compute_node_velocities(sycl::queue& q)
{

    //nodes
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor node_mass_acc(node_data.masses, h);
        sycl::accessor node_velocity_acc(node_data.velocities, h);
        sycl::accessor node_momenta_acc(node_data.momenta, h);
        sycl::accessor node_count_acc(pgi_manager.node_count);

        h.parallel_for(node_data.max_node_count,[=](sycl::id<1> idx)
        {
            const size_t node_count = node_count_acc[0];
            const size_t node_id = idx[0];
            if(node_id >= node_count)
                return;


            Coordinate_t momentum_i = node_momenta_acc[node_id];
            scalar_t mass_i = node_mass_acc[node_id];

            Coordinate_t velocity_i = Coordinate_t::Zero();
            if(mass_i > 0.0)
                velocity_i = 1.0 / (mass_i) * momentum_i;

            node_velocity_acc[node_id] = velocity_i;
        });
    });
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::transfer_velocity_nodes_to_particles(sycl::queue& q)
{
    auto interaction_access = pgi_manager.kernel_accessor;
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor node_velocity_acc(node_data.velocities, h);
        sycl::accessor particle_velocity_acc(particle_data.velocities, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);

        interaction_access.give_kernel_access(h);

        h.parallel_for(particle_count,[=](sycl::id<1> idx)
        {
            size_t pid = idx[0];
            Coordinate_t x_p = particle_positions_acc[pid];

            Coordinate_t v_p = Coordinate_t::Zero();
            for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                interaction_it != interaction_access.particle_interactions_end(pid);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                size_t node_id = interaction.node_id;
                Coordinate_t v_i = node_velocity_acc[node_id];
                NodeIndex_t node_index = interaction.node_index;

                v_p += interpolator.value(node_index, x_p) * v_i;
            }

            particle_velocity_acc[pid] = v_p;
        });
    });

}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::transfer_velocity_nodes_to_particles_APIC(sycl::queue& q)
{
    transfer_velocity_nodes_to_particles(q);

    auto interaction_access = pgi_manager.kernel_accessor;
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor node_velocity_acc(node_data.velocities, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);
        sycl::accessor particle_B_matrices_acc(particle_data.B_matrices, h);

        interaction_access.give_kernel_access(h);

        h.parallel_for(particle_count,[=](sycl::id<1> idx)
        {
            size_t pid = idx[0];
            Coordinate_t x_p = particle_positions_acc[pid];

            CoordinateMatrix_t B_p = CoordinateMatrix_t::Zero();
            for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                interaction_it != interaction_access.particle_interactions_end(pid);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                size_t node_id = interaction.node_id;
                Coordinate_t v_i = node_velocity_acc[node_id];
                NodeIndex_t node_index = interaction.node_index;
                Coordinate_t x_i = interpolator.position(node_index);

                B_p += interpolator.value(node_index, x_p) * v_i * (x_i - x_p).transpose();
            }

            particle_B_matrices_acc[pid] = B_p;
        });
    });

}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::update_particle_deformation_gradients(sycl::queue& q, Engine<TInterpolationScheme>::scalar_t dt)
{
    auto interaction_access = pgi_manager.kernel_accessor;
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor del_v_del_x_acc(particle_data.del_v_del_x, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);
        sycl::accessor node_velocity_acc(node_data.velocities, h);

        interaction_access.give_kernel_access(h);

        h.parallel_for(particle_count,[=](sycl::id<1> idx)
        {
            size_t pid = idx[0];
            Coordinate_t x_p = particle_positions_acc[pid];

            CoordinateMatrix_t del_v_del_x_p = CoordinateMatrix_t::Identity();
            for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                interaction_it != interaction_access.particle_interactions_end(pid);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                size_t node_id = interaction.node_id;
                Coordinate_t v_i = node_velocity_acc[node_id];
                NodeIndex_t node_index = interaction.node_index;
                Coordinate_t x_i = interpolator.position(node_index);

                Coordinate_t grad_n_i = interpolator.gradient(node_index, x_p);

                del_v_del_x_p += dt * v_i * grad_n_i.transpose();
            }
            del_v_del_x_acc[pid] = del_v_del_x_p;
        });
    });
}

}

#endif //ENGINE_HPP
