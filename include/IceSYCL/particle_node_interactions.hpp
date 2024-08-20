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


template<class Input_t, class BinaryPred>
    void id_segments(
        sycl::queue& q,
        sycl::buffer<Input_t>& inputs,
        sycl::buffer<size_t>& is_segment_begin,
        sycl::buffer<size_t>& segment_ids,
        BinaryPred p
    )
{
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor is_segment_begin_acc(is_segment_begin, h);
        h.single_task([=](){is_segment_begin_acc[0] = 0;});
    });

    q.submit([&](sycl::handler& h)
    {
        sycl::accessor inputs_acc(inputs, h);
        sycl::accessor is_segment_begin_acc(is_segment_begin, h);

        h.parallel_for(inputs.size() - 1, [=](sycl::id<1> idx)
        {
            bool is_same = p(inputs_acc[idx[0]], inputs_acc[idx[0] + 1]);
            is_segment_begin_acc[idx[0] + 1] = is_same ? 0 : 1;
        });
    });


    dpl::inclusive_scan(dpl::execution::make_device_policy(q),
        dpl::begin(is_segment_begin),
        dpl::end(is_segment_begin),
        dpl::begin(segment_ids),
        std::plus<size_t>{}
        );

    q.submit([&](sycl::handler& h)
    {
        sycl::accessor is_segment_begin_acc(is_segment_begin, h);
        h.single_task([=](){is_segment_begin_acc[0] = 1;});
    });
}

template<class ExecutionPolicy, class InputIt, class OutputIt, class BinaryPred>
    void segment_id(
        ExecutionPolicy exec,
        InputIt first,
        InputIt last,
        OutputIt d_first,
        BinaryPred p
    )
{
    auto are_different_pair = [=](typename std::iterator_traits<InputIt>::value_type a, typename std::iterator_traits<InputIt>::value_type b)->size_t
    {
        return p(a, b) ? 0 : 1;
    };

    oneapi::dpl::zip_iterator adjacent = dpl::make_zip_iterator(first, dpl::next(first));
    using zip_t = decltype(adjacent);

    auto are_different_zip = [=](typename std::iterator_traits<zip_t>::value_type zipped_adjacent)->size_t
    {
        return are_different_pair(oneapi::dpl::get<0>(zipped_adjacent), oneapi::dpl::get<1>(zipped_adjacent));
    };

    oneapi::dpl::transform_iterator are_different_it = oneapi::dpl::make_transform_iterator(adjacent, are_different_zip);

    dpl::exclusive_scan(
        exec,
        are_different_it,
        are_different_it + (last - first),
        d_first,
        0
    );
}

template<class TInterpolationScheme>
class ParticleNodeInteractionManager
{
public:
    using InterpolationScheme = TInterpolationScheme;
    using CoordinateConfiguration = typename InterpolationScheme::CoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using NodeIndex_t = typename CoordinateConfiguration::NodeIndex_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;

    struct NodeData
    {
        NodeIndex_t node_index;
        size_t particle_interaction_begin;
        size_t particle_interaction_count;
    };

    ParticleNodeInteractionManager(const size_t particle_count) :
    particle_count_{particle_count},
    interactions_by_particle_{particle_count},
    interactions_by_node_{particle_count},
    node_id_by_node_interaction{particle_count},
    segment_begin{particle_count}
    {
        sycl::host_accessor segment_begin_acc(segment_begin);
        std::fill(oneapi::dpl::begin(segment_begin_acc), oneapi::dpl::end(segment_begin_acc), 0);
    }

    size_t particle_count_;
    sycl::buffer<ParticleNodeInteraction<CoordinateConfiguration>> interactions_by_particle_;
    sycl::buffer<ParticleNodeInteraction<CoordinateConfiguration>> interactions_by_node_;

    sycl::buffer<size_t> segment_begin;
    sycl::buffer<size_t> node_id_by_node_interaction;



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



        auto dpl_policy = dpl::execution::make_device_policy<class ParticleNodeInteractionGenerationPolicy>(q);

        auto node_index_comparer = [](NodeIndex_t& a, NodeIndex_t& b)->bool
        {
            for(size_t dim = 0; dim < CoordinateConfiguration::Dimension; dim++)
            {
                if(a(dim) < b(dim))
                    return true;
                else if(a(dim) > b(dim))
                    return false;
            }
            return false;
        };

        auto interaction_comparer = [=](ParticleNodeInteraction<CoordinateConfiguration>& a, ParticleNodeInteraction<CoordinateConfiguration>& b)->bool
        {
            return node_index_comparer(a.node_index, b.node_index);
        };

        oneapi::dpl::copy(dpl_policy,
            dpl::begin(interactions_by_particle_),
            dpl::end(interactions_by_particle_),
            dpl::begin(interactions_by_node_));

        /*
        oneapi::dpl::sort(dpl_policy,
            dpl::begin(interactions_by_node_),
            dpl::end(interactions_by_node_),
            interaction_comparer
        );




        segment_id(dpl_policy,
            dpl::begin(interactions_by_node_),
            dpl::end(interactions_by_node_),
            node_id_by_node_interaction,
            is_different_node_in_interaction);
        */


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
/*
    void mark_interaction_by_node_segments(sycl::queue q)
    {
        auto is_equal = [](NodeIndex_t& a, NodeIndex_t& b)->bool
        {
            for(size_t dim = 0; dim < CoordinateConfiguration::Dimension; dim++)
            {
                if(a(dim) != b(dim))
                    return false;
            }
            return true;
        };

        auto is_different_node_in_interaction = [=](ParticleNodeInteraction<CoordinateConfiguration>& a, ParticleNodeInteraction<CoordinateConfiguration>& b)->size_t
        {
            return is_equal(a.node_index, b.node_index) ? 0 : 1;
        };

        q.submit([&](sycl::handler h)
        {
            sycl::accessor interactions_by_node_acc(interactions_by_node_, h);
            sycl::accessor segment_begin_acc(segment_begin, h);

            q.parallel_for(particle_count_ - 1, [=](sycl::id<1> idx)
            {

            });

        });

    }
*/
};

}


#endif //PARTICLE_NODE_INTERACTIONS_HPP
