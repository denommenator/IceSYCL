//
// Created by robert-denomme on 8/14/24.
//

#ifndef COORDINATES_HPP
#define COORDINATES_HPP

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

template<class CoordinateConfiguration>
typename CoordinateConfiguration::Coordinate_t
MakeCoordinate(std::array<typename CoordinateConfiguration::scalar_t, CoordinateConfiguration::Dimension> entries)
{
    return small_la::MakeVector<typename CoordinateConfiguration::scalar_t, CoordinateConfiguration::Dimension>
    (entries);
}

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
}

#endif //COORDINATES_HPP
