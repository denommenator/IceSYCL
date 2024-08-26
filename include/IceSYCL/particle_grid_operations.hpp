//
// Created by robert-denomme on 8/26/24.
//

#ifndef PARTICLEGRIDOPERATIONS_HPP
#define PARTICLEGRIDOPERATIONS_HPP




#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>

#include<sycl/sycl.hpp>

#include <IceSYCL/coordinates.hpp>
#include <IceSYCL/interpolation.hpp>
#include <IceSYCL/particle_grid_interactions.hpp>


namespace iceSYCL
{


template<class TInterpolationScheme, class TData>
void transfer_data_particles_to_grid(
    sycl::queue q,
    ParticleGridInteractionManager<TInterpolationScheme>& interaction_manager,
    TInterpolationScheme interpolator,
    sycl::buffer<TData> data_per_particle,
    sycl::buffer<TData> data_per_node,
    sycl::buffer<typename TInterpolationScheme::Coordinate_t> particle_positions,
    TData zero
)
{
    using Coordinate_t = typename TInterpolationScheme::Coordinate_t;
    using NodeIndex_t = typename TInterpolationScheme::NodeIndex_t;

    auto interaction_access = interaction_manager.kernel_accessor;
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor<TData> data_per_particle_acc(data_per_particle, h);
        sycl::accessor<TData> data_per_node_acc(data_per_node, h);
        sycl::accessor<Coordinate_t> particle_positions_acc(particle_positions, h);

        interaction_access.give_kernel_access(h);

        h.parallel_for(data_per_node_acc.size(),[=](sycl::id<1> idx)
        {
            const size_t node_count = interaction_access.node_count();
            const size_t node_id = idx[0];
            if(node_id >= node_count)
                return;

            TData result = zero;
            for(auto interaction_it = interaction_access.node_interactions_begin(node_id);
                interaction_it != interaction_access.node_interactions_end(node_id);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                size_t pid = interaction.particle_id;
                TData value_p = data_per_particle_acc[pid];
                Coordinate_t x_p = particle_positions_acc[pid];

                NodeIndex_t node_index = interaction.node_index;

                result += interpolator.value(node_index, x_p) * value_p;
            }

            data_per_node_acc[node_id] = result;
        });
    });
}

}

#endif //PARTICLEGRIDOPERATIONS_HPP
