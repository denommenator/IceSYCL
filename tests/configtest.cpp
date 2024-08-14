//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <IceSYCL/configuration.hpp>
#include <small_la/small_matrix.hpp>

TEST_CASE( "Interaction Generator Test", "[Configuration]" )
{
    using namespace iceSYCL;
    using Cubic2d = CubicInterpolationScheme<Double2DCoordinateConfiguration>;
    REQUIRE( Cubic2d::num_interactions_per_particle == 4 * 4 );

    using Cubic3d = CubicInterpolationScheme<Double3DCoordinateConfiguration>;
    REQUIRE(Cubic3d::num_interactions_per_particle == 4 * 4 * 4);

    Cubic2d interpolator;
    interpolator.h = 1.0;
    std::array<ParticleNodeInteraction<Double2DCoordinateConfiguration>, 16> interactions;

    Double2DCoordinateConfiguration::Coordinate_t p = small_la::MakeVector<double, 2>({1.0, 1.0});

    interpolator.generate_particle_node_interactions(0, p, interactions.begin());


    std::set<std::tuple<int, int>> indicesDesired =
        {{0,0}, {0,1}, {0, 2}, {0, 3},
        {1,0}, {1,1}, {1, 2}, {1, 3},
        {2,0}, {2,1}, {2, 2}, {2, 3},
        {3,0}, {3,1}, {3, 2}, {3, 3},
        };

    std::set<std::tuple<int, int>> indicesProduced;
    auto prducedInserter= std::inserter(indicesProduced, indicesProduced.begin());

    std::transform(interactions.begin(), interactions.end(), prducedInserter, [](ParticleNodeInteraction<Double2DCoordinateConfiguration> interaction)
    {
        return std::make_tuple(interaction.node_index(0), interaction.node_index(1));
    });


    std::set<std::tuple<int, int>> symmetric_difference;
    std::set_symmetric_difference(indicesProduced.begin(), indicesProduced.end(), indicesDesired.begin(), indicesDesired.end(), std::inserter(symmetric_difference, symmetric_difference.begin()));
    REQUIRE(symmetric_difference.empty());



}

TEST_CASE( "Interaction Generator Kernel Test", "[Configuration]" )
{
    using namespace iceSYCL;
    using Cubic2d = CubicInterpolationScheme<Double2DCoordinateConfiguration>;

    Cubic2d interpolator;
    interpolator.h = 1.0;
    std::array<ParticleNodeInteraction<Double2DCoordinateConfiguration>, 32> interactions;

    Double2DCoordinateConfiguration::Coordinate_t p0 = small_la::MakeVector<double, 2>({1.0, 1.0});
    Double2DCoordinateConfiguration::Coordinate_t p1 = small_la::MakeVector<double, 2>({1.0, 1.0});
    std::array<Double2DCoordinateConfiguration::Coordinate_t, 2> points = {p0, p1};
    sycl::buffer pointsB(points);
    {
        sycl::host_accessor pointsHostAcc(pointsB);
        std::copy(points.begin(), points.end(), pointsHostAcc.begin());
    }

    {
        sycl::buffer interactionB(interactions);

        sycl::queue q(sycl::gpu_selector_v);

        q.submit([&](sycl::handler& h)
        {
            sycl::accessor interactionAcc(interactionB, h);
            sycl::accessor pointsAcc(pointsB, h);
            h.parallel_for(sycl::range<1>(2), [=](sycl::id<1> idx)
            {
                auto p = pointsAcc[idx];
                auto begin = interactionAcc.begin() + 16 * idx[0];
                interpolator.generate_particle_node_interactions(idx[0], p, begin);
                //ParticleNodeInteraction<Double2DCoordinateConfiguration> interaction;
                //interaction.particle_id = 100;
                // *(interactionAcc.begin()) = interaction;
            });
        });
        q.wait();
    }



    std::set<std::tuple<int, int>> indicesDesired =
        {{0,0}, {0,1}, {0, 2}, {0, 3},
        {1,0}, {1,1}, {1, 2}, {1, 3},
        {2,0}, {2,1}, {2, 2}, {2, 3},
        {3,0}, {3,1}, {3, 2}, {3, 3}
        };

    std::vector<std::tuple<int, int>> indicesProduced;
    auto prducedInserter= std::back_inserter(indicesProduced);

    std::transform(interactions.begin(), interactions.end(), prducedInserter, [](ParticleNodeInteraction<Double2DCoordinateConfiguration> interaction)
    {
        return std::make_tuple(interaction.node_index(0), interaction.node_index(1));
    });

    std::set<std::tuple<int, int>> symmetric_difference;
    std::sort(indicesProduced.begin(), indicesProduced.begin() + 16);
    std::sort(indicesProduced.begin() + 16, indicesProduced.begin() + 32);

    std::set_symmetric_difference(indicesProduced.begin(), indicesProduced.begin() + 16, indicesDesired.begin(), indicesDesired.end(), std::inserter(symmetric_difference, symmetric_difference.begin()));
    REQUIRE(symmetric_difference.empty());

    std::set_symmetric_difference(indicesProduced.begin() + 16, indicesProduced.begin() + 32, indicesDesired.begin(), indicesDesired.end(), std::inserter(symmetric_difference, symmetric_difference.begin()));
    REQUIRE(symmetric_difference.empty());



}