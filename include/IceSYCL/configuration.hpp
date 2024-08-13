//
// Created by robert-denomme on 8/12/24.
//

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <small_la/small_matrix.hpp>

namespace iceSYCL
{

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
    static constexpr int _num_interactions_per_dimension = 4;
    static constexpr int num_interactions_per_particle = compile_time_power(_num_interactions_per_dimension, CoordinateConfiguration::Dimension);

private:

};

}

#endif //CONFIGURATION_HPP
