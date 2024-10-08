//
// Created by robert-denomme on 8/16/24.
//

#ifndef PARTICLE_NODE_INTERACTIONS_HPP
#define PARTICLE_NODE_INTERACTIONS_HPP

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>

#include "kernel_access_abstraction.hpp"

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

///TODO use ranges api to make a generic version of id_segments
// template<class ExecutionPolicy, class InputIt, class OutputIt, class BinaryPred>
//     void segment_id(
//         ExecutionPolicy exec,
//         InputIt first,
//         InputIt last,
//         OutputIt d_first,
//         BinaryPred p
//     )
// {
//     auto are_different_pair = [=](typename std::iterator_traits<InputIt>::value_type a, typename std::iterator_traits<InputIt>::value_type b)->size_t
//     {
//         return p(a, b) ? 0 : 1;
//     };
//
//     oneapi::dpl::zip_iterator adjacent = dpl::make_zip_iterator(first, dpl::next(first));
//     using zip_t = decltype(adjacent);
//
//     auto are_different_zip = [=](typename std::iterator_traits<zip_t>::value_type zipped_adjacent)->size_t
//     {
//         return are_different_pair(oneapi::dpl::get<0>(zipped_adjacent), oneapi::dpl::get<1>(zipped_adjacent));
//     };
//
//     oneapi::dpl::transform_iterator are_different_it = oneapi::dpl::make_transform_iterator(adjacent, are_different_zip);
//
//     dpl::exclusive_scan(
//         exec,
//         are_different_it,
//         are_different_it + (last - first),
//         d_first,
//         0
//     );
// }

template<class TInterpolationScheme>
class ParticleGridInteractionManager
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
    using NodeData_t = NodeData;

    ParticleGridInteractionManager(const size_t particle_count) :
    particle_count_{particle_count},
    particle_node_interaction_count_{particle_count * InterpolationScheme::num_interactions_per_particle},
    interactions_by_particle_{particle_node_interaction_count_},
    interactions_by_node_{particle_node_interaction_count_},
    node_id_by_node_interaction{particle_node_interaction_count_},
    segment_begin{particle_node_interaction_count_},
    node_count{1},
    node_data_{particle_node_interaction_count_},
    kernel_accessor{
        interactions_by_particle_,
        interactions_by_node_,
        node_data_,
        node_count
    }
    {
    }
public:
    size_t particle_count_;
    size_t particle_node_interaction_count_;
    sycl::buffer<ParticleNodeInteraction<CoordinateConfiguration>> interactions_by_particle_;
    sycl::buffer<ParticleNodeInteraction<CoordinateConfiguration>> interactions_by_node_;
    sycl::buffer<NodeData> node_data_;

    sycl::buffer<size_t> segment_begin;
    sycl::buffer<size_t> node_id_by_node_interaction;
    sycl::buffer<size_t> node_count;

public:
    void update_particle_locations(
        sycl::queue& q,
        sycl::buffer<Coordinate_t>& particle_locations,
        const InterpolationScheme& interpolator
    )
    {
        generate_particle_node_interactions(q, particle_locations, interpolator);
        identify_nodes(q);
        populate_node_data(q);
    }

    size_t get_node_count_host()
    {
        sycl::host_accessor node_count_acc(node_count);
        return (*std::prev(node_count_acc.end()));
    }

private:
    void generate_particle_node_interactions(
        sycl::queue& q,
        sycl::buffer<Coordinate_t>& particle_locations,
        const InterpolationScheme& interpolator
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
                auto interactions_begin = interactions_by_particle_acc.begin() + pid * InterpolationScheme::num_interactions_per_particle;

                auto converter = [=](typename InterpolationScheme::Interaction interaction)
                {
                    return local_interaction_to_global(pid, interaction);
                };

                interpolator.generate_particle_node_interactions(p, interactions_begin, converter);
            });

        });
    }

    void identify_nodes(
            sycl::queue& q
        )
    {
        auto dpl_policy = dpl::execution::make_device_policy<class ParticleNodeInteractionGenerationPolicy>(q);
        oneapi::dpl::copy(dpl_policy,
            dpl::begin(interactions_by_particle_),
            dpl::end(interactions_by_particle_),
            dpl::begin(interactions_by_node_));



        auto node_index_comparer = [](const NodeIndex_t& a, const NodeIndex_t& b)->bool
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

        auto interaction_comparer = [=](const ParticleNodeInteraction<CoordinateConfiguration>& a, const ParticleNodeInteraction<CoordinateConfiguration>& b)->bool
        {
            return node_index_comparer(a.node_index, b.node_index);
        };




        oneapi::dpl::sort(dpl_policy,
            dpl::begin(interactions_by_node_),
            dpl::end(interactions_by_node_),
            interaction_comparer
        );


        auto is_different_node_in_interaction = [=](ParticleNodeInteraction<CoordinateConfiguration>& a, ParticleNodeInteraction<CoordinateConfiguration>& b)->bool
        {
            return a.node_index == b.node_index;
        };

        id_segments(q,
            interactions_by_node_,
            segment_begin,
            node_id_by_node_interaction,
            is_different_node_in_interaction);

        q.submit([&](sycl::handler& h)
        {
            sycl::accessor node_id_by_node_interaction_acc(node_id_by_node_interaction, h);
            sycl::accessor node_count_acc(node_count, h);
            h.single_task([=](){node_count_acc[0] = (*std::prev(node_id_by_node_interaction_acc.end())) + 1;});
        });

    }

    void populate_node_data(sycl::queue& q)
    {
        q.submit([&](sycl::handler& h)
        {
            sycl::accessor node_id_by_node_interaction_acc(node_id_by_node_interaction, h);
            sycl::accessor interactions_by_node_acc(interactions_by_node_, h);
            sycl::accessor node_data_acc(node_data_, h);
            sycl::accessor interactions_by_particle_acc(interactions_by_particle_, h);
            sycl::accessor is_segment_begin_acc(segment_begin, h);

            h.parallel_for(particle_node_interaction_count_,
                [=](sycl::id<1> idx)
                {
                    ParticleNodeInteraction<CoordinateConfiguration>& interaction = interactions_by_node_acc[idx];
                    size_t node_id = node_id_by_node_interaction_acc[idx];
                    interaction.node_id = node_id;

                    size_t particle_interaction_offset = interaction.particle_id * InterpolationScheme::num_interactions_per_particle + interaction.particle_interaction_number;
                    interactions_by_particle_acc[particle_interaction_offset].node_id = node_id;

                    if(is_segment_begin_acc[idx])
                    {
                        NodeData node_data = {
                           interaction.node_index,
                           idx[0],
                           0
                       };
                       node_data_acc[node_id] = node_data;
                    }
                });

        });

        q.submit([&](sycl::handler& h)
        {
            sycl::accessor node_data_acc(node_data_, h);
            sycl::accessor node_count_acc(node_count, h);

            //TODO figure out how to do dynamic dispatch sizes based on GPU buffer data
            size_t particle_node_interaction_count = particle_node_interaction_count_;
            h.parallel_for(particle_node_interaction_count_,
                [=](sycl::id<1> idx)
                {
                    size_t node_id = idx[0];
                    size_t num_nodes = node_count_acc[0];
                    if(node_id >= num_nodes)
                        return;

                    size_t node_interaction_begin = node_data_acc[node_id].particle_interaction_begin;
                    if(node_id < num_nodes - 1)
                    {

                        size_t node_interaction_begin_next = node_data_acc[node_id + 1].particle_interaction_begin;
                        node_data_acc[node_id].particle_interaction_count = node_interaction_begin_next - node_interaction_begin;
                    }
                    else
                    {
                        node_data_acc[node_id].particle_interaction_count = particle_node_interaction_count - node_interaction_begin;
                    }
                });

        });

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

public:

    template<bool TIsDevice>
    struct KernelAccessor
    {
        using interaction_iterator_t = typename sycl::accessor<ParticleNodeInteraction<CoordinateConfiguration>>::iterator;
        template<bool TIsDevice_dummy = TIsDevice>
        std::enable_if_t<TIsDevice_dummy, void> give_kernel_access(sycl::handler& h)
        {
            h.require(interactions_by_particle_acc);
            h.require(interactions_by_node_acc);
            h.require(node_data_acc);

            h.require(node_count_acc);
        }

        interaction_iterator_t particle_interactions_begin(size_t pid) const
        {
            return interactions_by_particle_acc.begin() + pid * InterpolationScheme::num_interactions_per_particle;
        }

        interaction_iterator_t particle_interactions_end(size_t pid) const
        {
            return interactions_by_particle_acc.begin() + (pid + 1) * InterpolationScheme::num_interactions_per_particle;
        }

        interaction_iterator_t node_interactions_begin(size_t node_id) const
        {
            size_t begin = node_data_acc[node_id].particle_interaction_begin;
            return interactions_by_node_acc.begin() + begin;
        }

        interaction_iterator_t node_interactions_end(size_t node_id) const
        {
            size_t begin = node_data_acc[node_id].particle_interaction_begin;
            size_t count = node_data_acc[node_id].particle_interaction_count;
            return interactions_by_node_acc.begin() + begin + count;
        }

        NodeIndex_t get_node_index(size_t node_id) const
        {
            return node_data_acc[node_id].node_index;
        }

        const size_t node_count() const
        {
            return node_count_acc[0];
        }



        Accessor_t<ParticleNodeInteraction<CoordinateConfiguration>, TIsDevice> interactions_by_particle_acc;
        Accessor_t<ParticleNodeInteraction<CoordinateConfiguration>, TIsDevice> interactions_by_node_acc;
        Accessor_t<NodeData, TIsDevice> node_data_acc;

        Accessor_t<size_t, TIsDevice> node_count_acc;
    };

    KernelAccessor<true> kernel_accessor;
    KernelAccessor<false> get_host_accessor()
    {
        return KernelAccessor<false> {
            interactions_by_particle_,
            interactions_by_node_,
            node_data_,
            node_count
        };
    }


};

}


#endif //PARTICLE_NODE_INTERACTIONS_HPP
