//
// Created by robert-denomme on 8/12/24.
//

#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include <cmath>

#include "coordinates.hpp"
#include "utility.hpp"

namespace iceSYCL
{

template<class TCoordinateConfiguration>
class CubicInterpolationScheme
{
public:
    using CoordinateConfiguration = TCoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using NodeIndex_t = typename CoordinateConfiguration::NodeIndex_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;

    explicit CubicInterpolationScheme(scalar_t h_grid_spacing) : h{h_grid_spacing}
    {}

    struct Interaction
    {
        NodeIndex_t node_index;
        size_t particle_interaction_number;
    };

    static constexpr int Dimension = CoordinateConfiguration::Dimension;

    static constexpr int _num_interactions_per_dimension = 4;
    static constexpr int num_interactions_per_particle = pow<CoordinateConfiguration::Dimension, int>(_num_interactions_per_dimension);

    scalar_t h;

    template<class InputIt, class TTransform>
    void generate_particle_node_interactions(const Coordinate_t p, const InputIt begin, TTransform f) const
    {
        NodeIndex_t first_index = calculate_first_node(p);
        InputIt iter = begin;

        for(size_t i = 0; i < num_interactions_per_particle; ++i, ++iter)
        {
            Interaction interaction =
                {
                    first_index + offset(i),
                    i
                };

            *iter = f(interaction);
        }
    }

    template<class InputIt>
    void generate_particle_node_interactions(const Coordinate_t p, const InputIt begin) const
    {
        NodeIndex_t first_index = calculate_first_node(p);
        InputIt iter = begin;

        for(size_t i = 0; i < num_interactions_per_particle; ++i, ++iter)
        {
            Interaction interaction =
                {
                first_index + offset(i),
                i
            };

            *iter = interaction;
        }
    }

    scalar_t value(NodeIndex_t i, Coordinate_t x_p) const
    {
        Coordinate_t x_i = position(i);
        scalar_t ret = 1.0;

        for(int dim = 0; dim < Dimension; ++dim)
        {
            ret *= n(1.0 / h * (x_p(dim) - x_i(dim)));
        }

        return ret;
    }

    Coordinate_t gradient_impl(NodeIndex_t i, Coordinate_t x_p) const
    {
        Coordinate_t x_i = position(i);
        Coordinate_t ret;
        for(int dim = 0; dim < Dimension; ++dim)
        {
            scalar_t ret_dim = 1.0;
            for(int pre_dim = 0; pre_dim < dim; ++pre_dim)
            {
                ret_dim *= n(1.0 / h * (x_p(pre_dim) - x_i(pre_dim)));
            }

            ret_dim *= 1.0 / h * n_prime(1.0 / h * (x_p(dim) - x_i(dim)));

            for(int post_dim = dim + 1; post_dim < Dimension; ++post_dim)
            {
                ret_dim *= n(1.0 / h * (x_p(post_dim) - x_i(post_dim)));
            }

            ret(dim) = ret_dim;
        }
        return ret;
    }

    Coordinate_t position(NodeIndex_t i) const
    {
        Coordinate_t i_scalar;
        for(int dim = 0; dim < Dimension; dim++)
        {
            i_scalar(dim) = i(dim);
        }
        return h * i_scalar;
    }

    scalar_t node_volume() const
    {
        return pow<Dimension>(h);
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
