//
// Created by robert-denomme on 8/22/24.
//

#ifndef ENGINE_HPP
#define ENGINE_HPP

namespace iceSYCL
{

template<class CoordinateConfiguration>
struct ParticleInitialState
{
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    using scalar_t = typename CoordinateConfiguration::scalar_t;

    std::vector<Coordinate_t> positions;
    std::vector<Coordinate_t> velocities;
    std::vector<scalar_t> masses;
};

template<class TInterpolationScheme>
class Engine
{
public:
    using InterpolationScheme = TInterpolationScheme;
    using CoordinateConfiguration = typename InterpolationScheme::CoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using NodeIndex_t = typename CoordinateConfiguration::NodeIndex_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    static constexpr int Dimension = CoordinateConfiguration::Dimension;
    using CoordinateMatrix_t = small_la::small_matrix<scalar_t, Dimension, Dimension>;

    Engine(InterpolationScheme interpolator_instance) :
    interpolator{interpolator_instance},
    interaction_manager{}
    {}

    InterpolationScheme interpolator;
    ParticleNodeInteractionManager<InterpolationScheme> interaction_manager;

public:
    struct ParticleData
    {

        sycl::buffer<Coordinate_t> positions;
        sycl::buffer<Coordinate_t> velocities;
        sycl::buffer<Coordinate_t> positions_prev;
        sycl::buffer<Coordinate_t> velocities_prev;
        sycl::buffer<scalar_t> masses;
        sycl::buffer<scalar_t> rest_volumes;
        sycl::buffer<CoordinateMatrix_t> B_matrices;
        sycl::buffer<CoordinateMatrix_t> D_matrices;
        sycl::buffer<CoordinateMatrix_t> deformation_gradients;
        sycl::buffer<CoordinateMatrix_t> deformation_gradients_prev;


    };

    ParticleData particle_data;

public:
    struct NodeData
    {

    };

    NodeData node_data;
};

};

#endif //ENGINE_HPP
