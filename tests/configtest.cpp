//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <IceSYCL/configuration.hpp>

TEST_CASE( "GPU Sort check", "[main]" )
{
    using namespace iceSYCL;
    using Cubic2d = CubicInterpolationScheme<Double2DCoordinateConfiguration>;
    REQUIRE( Cubic2d::num_interactions_per_particle == 4 * 4 );

    using Cubic3d = CubicInterpolationScheme<Double3DCoordinateConfiguration>;
    REQUIRE(Cubic3d::num_interactions_per_particle == 4 * 4 * 4);

}