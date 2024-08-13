//
// Created by robert-denomme on 8/12/24.
//

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <small_la/small_matrix.hpp>

namespace iceSYCL
{

static constexpr int compile_time_power(const int base, const int exponent)
{
    int ret = 1;
    for(int i = 0; i < exponent; i++)
    {
        ret *= base;
    }
    return ret;
}

template<class Tscalar_t, int TDimension>
class CoordinateConfiguration
{
public:
    static constexpr int Dimension = TDimension;
    using scalar_t = Tscalar_t;
    using Coordinate_t = small_la::small_matrix<scalar_t, Dimension, 1>;
    using NodeIndex_t = small_la::small_matrix<int, Dimension, 1>;

};

using Double2DCoordinateConfiguration = CoordinateConfiguration<double, 2>;
using Float2DCoordinateConfiguration = CoordinateConfiguration<float, 2>;
using Double3DCoordinateConfiguration = CoordinateConfiguration<double, 3>;
using Float3DCoordinateConfiguration = CoordinateConfiguration<float, 3>;




template<class TCoordinateConfiguration>
struct ParticleNodeInteraction
{
    int particle_id;
    typename TCoordinateConfiguration::NodeIndex_t node_index;
    int node_id;
    int particle_interaction_number;

};

template<class TCoordinateConfiguration>
class CubicInterpolationScheme
{
public:
    using CoordinateConfiguration = TCoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using NodeIndex_t = typename CoordinateConfiguration::NodeIndex_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    static constexpr int Dimension = CoordinateConfiguration::Dimension;

    static constexpr int _num_interactions_per_dimension = 4;
    static constexpr int num_interactions_per_particle = compile_time_power(_num_interactions_per_dimension, CoordinateConfiguration::Dimension);

    scalar_t h;

    template<class InputIt>
    void generate_particle_node_interactions(const int p_id, const Coordinate_t p, const InputIt begin) const
    {
        NodeIndex_t first_index = calculate_first_node(p);
        InputIt iter = begin;

        for(int i = 0; i < num_interactions_per_particle; ++i, ++iter)
        {
            ParticleNodeInteraction<CoordinateConfiguration> interaction;
            interaction.particle_id = p_id;
            interaction.node_id = 0;
            interaction.node_index = first_index + offset(i);
            interaction.particle_interaction_number = i;

            *iter = interaction;
        }
    }

private:
    static NodeIndex_t offset(int local_interaction_number)
    {
        NodeIndex_t ret;
        for(int dim = 0; dim < Dimension; dim++)
        {
            ret(dim) = local_interaction_number % _num_interactions_per_dimension;
            local_interaction_number /= _num_interactions_per_dimension;
        }

        return ret;
    }

    NodeIndex_t calculate_first_node(const Coordinate_t p) const
    {
        NodeIndex_t ret;

        for(int dim = 0; dim < Dimension; ++dim)
        {
            ret(dim) = static_cast <int>(std::floor(p(dim) / h)) - 1;
        }

        return ret;
    }

};

}

#endif //CONFIGURATION_HPP
