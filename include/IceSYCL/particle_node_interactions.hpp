//
// Created by robert-denomme on 8/16/24.
//

#ifndef PARTICLE_NODE_INTERACTIONS_HPP
#define PARTICLE_NODE_INTERACTIONS_HPP

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>

#include "coordinates.hpp"
#include "interpolation.hpp"

namespace iceSYCL
{

template<class TInterpolationScheme>
class ParticleNodeInteractionManager
{
public:
    using InterpolationScheme = TInterpolationScheme;
    using CoordinateConfiguration = typename InterpolationScheme::CoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using NodeIndex_t = typename CoordinateConfiguration::NodeIndex_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;

    ParticleNodeInteractionManager(const size_t particle_count) :
    particle_count_{particle_count},
    interactions_by_particle_{particle_count},
    interactions_by_node_{particle_count}
    {}

    size_t particle_count_;
    sycl::buffer<ParticleNodeInteraction<CoordinateConfiguration>> interactions_by_particle_;
    sycl::buffer<ParticleNodeInteraction<CoordinateConfiguration>> interactions_by_node_;

    void generate_particle_node_interactions(
        sycl::queue q,
        sycl::buffer<Coordinate_t>& particle_locations,
        const InterpolationScheme interpolator
    )
    {
        q.submit([&](sycl::handler& h)
        {
            sycl::accessor interactions_by_particle_acc(interactions_by_particle_, h);
            sycl::accessor particle_locations_acc(particle_locations, h);

            h.parallel_for(particle_count_,[=](sycl::id<1> i)
            {
                Coordinate_t p = particle_locations_acc[i];
                const size_t pid = i[0];
                auto interactions_begin = interactions_by_particle_acc.begin() + i[0] * InterpolationScheme::num_interactions_per_particle;

                auto converter = [=](typename InterpolationScheme::Interaction interaction)
                {
                    return local_interaction_to_global(pid, interaction);
                };

                interpolator.generate_particle_node_interactions(p, interactions_begin, converter);
            });

        });



        auto dpl_policy = dpl::execution::make_device_policy<class SortPolicy>(q);


        oneapi::dpl::copy(dpl_policy,
            dpl::begin(interactions_by_particle_),
            dpl::end(interactions_by_particle_),
            dpl::begin(interactions_by_node_));

    }

    size_t get_node_count()
    {
        return 0;
    }

private:
    static ParticleNodeInteraction<CoordinateConfiguration> local_interaction_to_global(
        size_t pid,
        typename InterpolationScheme::Interaction interaction)
    {
        ParticleNodeInteraction<CoordinateConfiguration> ret
            {
                pid,
                interaction.node_index,
                pid,
                interaction.particle_interaction_number
            };
        return ret;
    }

};

}


#endif //PARTICLE_NODE_INTERACTIONS_HPP
