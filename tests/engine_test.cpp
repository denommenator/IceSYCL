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

#include <vector>
#include <functional>

TEST_CASE( "Rest volume test", "[particle_node_operations]" )
{
    using namespace iceSYCL;
    using Cubic2d = CubicInterpolationScheme<Double2DCoordinateConfiguration>;
    using Coordinate_t = Cubic2d::Coordinate_t;
    using scalar_t = Cubic2d::scalar_t;

    Coordinate_t zero = Coordinate_t::Zero();
    std::vector<Coordinate_t> particle_positions = {
        Coordinate_t(0.0, 0.0),
        Coordinate_t(100.0, 100.0),
        Coordinate_t(100.0, 100.0),
        Coordinate_t(10.5, 10.5),
        Coordinate_t(110.5, 110.5),
        Coordinate_t(110.5, 110.5)
    };
    size_t particle_count = particle_positions.size();
    std::vector<scalar_t> particle_mass(particle_count, 1.0);
    std::vector<Coordinate_t>  particle_velocities(particle_count, zero);

    const scalar_t h{1.0};
    Cubic2d interpolator(h);
    ParticleGridInteractionManager<Cubic2d> pgi_manager(particle_count);

    auto initial_state = Engine<Cubic2d>::MakeInitialState(
        particle_positions,
        particle_velocities,
        particle_mass,
        pgi_manager,
        interpolator
        );


    auto rest_volumes = initial_state.rest_volumes;

    //for(auto v_p : rest_volumes)
    //    std::cout << v_p << ", " ;
    //std::cout << std::endl;

    CHECK(rest_volumes[1] == Approx(rest_volumes[2]));
    CHECK(rest_volumes[1] == Approx(0.5 * rest_volumes[0]));
    CHECK(rest_volumes[4] == Approx(rest_volumes[5]));
    CHECK(rest_volumes[4] == Approx(0.5 * rest_volumes[3]));
}

TEST_CASE( "First engine test!", "[particle_node_operations]" )
{
    using namespace iceSYCL;
    using Cubic2d = CubicInterpolationScheme<Double2DCoordinateConfiguration>;
    using Coordinate_t = Cubic2d::Coordinate_t;
    using scalar_t = Cubic2d::scalar_t;

    Coordinate_t zero = Coordinate_t::Zero();
    std::vector<Coordinate_t> particle_positions = {
        Coordinate_t(0.0, 0.0),
        Coordinate_t(100.0, 100.0),
        Coordinate_t(100.0, 100.0),
        Coordinate_t(10.5, 10.5),
        Coordinate_t(110.5, 110.5),
        Coordinate_t(110.5, 110.5)
    };
    size_t particle_count = particle_positions.size();
    std::vector<scalar_t> particle_mass(particle_count, 1.0);
    std::vector<Coordinate_t>  particle_velocities(particle_count, zero);

    const scalar_t h{1.0};
    Cubic2d interpolator(h);

    Engine<Cubic2d> engine = Engine<Cubic2d>::FromInitialState(interpolator,particle_positions, particle_velocities, particle_mass);
    for(int i = 0; i < 50 * 2; ++i)
        engine.step_frame();
}