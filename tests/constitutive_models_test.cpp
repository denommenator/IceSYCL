//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>



#include <sycl/sycl.hpp>

#include <IceSYCL/coordinates.hpp>
#include <IceSYCL/interpolation.hpp>
#include <IceSYCL/particle_grid_interactions.hpp>
#include <IceSYCL/engine.hpp>
#include <IceSYCL/constitutive_models.h>

#include <vector>
#include <functional>
#include <cmath>

TEST_CASE( "constitutive model test", "[particle_node_operations]" )
{
    using namespace iceSYCL;
    using Cubic2d = CubicInterpolationScheme<Double2DCoordinateConfiguration>;
    using Coordinate_t = Cubic2d::Coordinate_t;
    using scalar_t = Cubic2d::scalar_t;


}