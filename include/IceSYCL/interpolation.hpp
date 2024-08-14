//
// Created by robert-denomme on 8/12/24.
//

#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include "coordinates.hpp"

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


template<class TCoordinateConfiguration>
class CubicInterpolationScheme
{
public:
    using CoordinateConfiguration = TCoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using NodeIndex_t = typename CoordinateConfiguration::NodeIndex_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;

    struct Interaction
    {
        NodeIndex_t node_index;
        int particle_interaction_number;
    };

    static constexpr int Dimension = CoordinateConfiguration::Dimension;

    static constexpr int _num_interactions_per_dimension = 4;
    static constexpr int num_interactions_per_particle = compile_time_power(_num_interactions_per_dimension, CoordinateConfiguration::Dimension);

    scalar_t h;

    template<class InputIt, class TTransform>
    void generate_particle_node_interactions(const Coordinate_t p, const InputIt begin, TTransform f) const
    {
        NodeIndex_t first_index = calculate_first_node(p);
        InputIt iter = begin;

        for(int i = 0; i < num_interactions_per_particle; ++i, ++iter)
        {
            Interaction interaction;
            interaction.node_index = first_index + offset(i);
            interaction.particle_interaction_number = i;

            *iter = f(interaction);
        }
    }

    template<class InputIt>
    void generate_particle_node_interactions(const Coordinate_t p, const InputIt begin) const
    {
        auto do_nothing = [](Interaction interaction){return interaction;};
        generate_particle_node_interactions(p, begin, do_nothing);
    }

    scalar_t value(NodeIndex_t i, Coordinate_t x_p)
    {
        scalar_t x_i_0 = i(0) * h;
        scalar_t x_i_1 = i(1) * h;

        return n(1.0 / h * (x_p(0) - x_i_0))
                * n(1.0 / h * (x_p(1) - x_i_1));
    }

    Coordinate_t gradient_impl(NodeIndex_t i, Coordinate_t x_p)
    {
        scalar_t x_i_0 = i(0) * h;
        scalar_t x_i_1 = i(1) * h;

        return Coordinate(
            1.0 / h * n_prime(1.0 / h * (x_p(0) - x_i_0)) * n(1.0 / h * (x_p(1) - x_i_1)),
            1.0 / h * n(1.0 / h * (x_p(0) - x_i_0)) * n_prime(1.0 / h * (x_p(1) - x_i_1)));

    }

    Coordinate_t position(NodeIndex_t i)
    {
        NodeIndex_t i_scalar;
        for(int dim = 0; dim < Dimension; dim++)
        {
            i_scalar(dim) = i(dim);
        }
        return h * i_scalar;
    }

private:
    static scalar_t n(scalar_t x)
    {
        if(std::abs(x) >= 2)
        {
            return 0;
        }
        else if(std::abs(x) >= 1)
        {
            return 1.0 / 6 * (2.0 - std::abs(x)) * (2.0 - std::abs(x)) * (2.0 - std::abs(x));
        }
        else
        {
            return 1.0 / 2 * std::abs(x) * std::abs(x) * std::abs(x) - std::abs(x) * std::abs(x) + 2.0 / 3.0;
        }
    }

    static scalar_t n_prime(scalar_t x)
    {
        if(std::abs(x) >= 2)
        {
            return 0;
        }
        else if(std::abs(x) >= 1)
        {
            return - 3.0 * 1.0 / 6 * (2.0 - std::abs(x)) * (2.0 - std::abs(x)) * (std::abs(x) / x);
        }
        else if(std::abs(x) > 1.0E-10)
        {
            return 3.0 * 1.0 / 2 * x * x * std::abs(x) / x - 2 * x;
        }
        else
        {
            return 0.0;
        }
    }

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

#endif //INTERPOLATION_HPP
