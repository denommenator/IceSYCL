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

TEST_CASE( "Rest volume test", "[engine_test]" )
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

    const scalar_t unit_density = 1.0;
    auto initial_state = Engine<Cubic2d>::MakeUniformDensityInitialState(
            particle_positions,
            particle_velocities,
            pgi_manager,
            interpolator,
            unit_density
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

TEST_CASE( "First engine test!", "[engine_test]" )
{
    using namespace iceSYCL;
    using CoordinateConfiguration = Double2DCoordinateConfiguration;
    using Cubic2d = CubicInterpolationScheme<CoordinateConfiguration>;
    using Coordinate_t = Cubic2d::Coordinate_t;
    using scalar_t = Cubic2d::scalar_t;
    using ConstitutiveModel = DensityBasedConstitutiveModel<TaitPressureFromDensity<CoordinateConfiguration>>;

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

    std::vector<ElasticCollisionWall<Double2DCoordinateConfiguration>> walls{};
    Engine<Cubic2d> engine = Engine<Cubic2d>::FromInitialState(interpolator,particle_positions, particle_velocities, walls);

    ConstitutiveModel Psi{TaitPressureFromDensity<CoordinateConfiguration>{1.0, 3.0, 100}};
//    Psi.pressure.unit_density = 1.0;
//    Psi.pressure.gamma = 3.0;
//    Psi.pressure.c = 100;
    //for(int i = 0; i < 50 * 2; ++i)
        engine.step_frame(Psi);

    {
        sycl::host_accessor particle_positions_acc(engine.particle_data.positions);
        sycl::host_accessor particle_velocities_acc(engine.particle_data.velocities);
        for(size_t pid = 0; pid < particle_count; pid++)
        {
            Coordinate_t x_p = particle_positions_acc[pid];
            CHECK(!std::isnan(x_p(0)));
            CHECK(!std::isnan(x_p(1)));

            Coordinate_t v_p = particle_velocities_acc[pid];
            CHECK(!std::isnan(v_p(0)));
            CHECK(!std::isnan(v_p(1)));
        }

        sycl::host_accessor node_momenta_acc(engine.node_data.momenta);
        size_t node_count = engine.pgi_manager.get_node_count_host();
        for(size_t node_id = 0; node_id < node_count; node_id++)
        {
            Coordinate_t m_i = node_momenta_acc[node_id];
            CHECK(!std::isnan(m_i(0)));
            CHECK(!std::isnan(m_i(1)));
        }
    }


    Engine<Cubic2d>::CoordinateMatrix_t D(
            1.0, 0.0,
            1.0, 0.0);
    auto D_inv = inverse(D);
}

TEST_CASE( "First implicit engine test!", "[engine_test]" )
{
    using namespace iceSYCL;
    using CoordinateConfiguration = Double2DCoordinateConfiguration;
    using Cubic2d = CubicInterpolationScheme<CoordinateConfiguration>;
    using Coordinate_t = Cubic2d::Coordinate_t;
    using scalar_t = Cubic2d::scalar_t;
    using ConstitutiveModel = FixedCorotated<CoordinateConfiguration>;


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

    const scalar_t h{5.0};
    Cubic2d interpolator(h);

    std::vector<ElasticCollisionWall<Double2DCoordinateConfiguration>> walls{};
    Engine<Cubic2d> engine = Engine<Cubic2d>::FromInitialState(interpolator,particle_positions, particle_velocities, walls);

    ConstitutiveModel Psi{FixedCorotated<CoordinateConfiguration>{1000, 1000.0}};//    Psi.pressure.unit_density = 1.0;
//    Psi.pressure.gamma = 3.0;
//    Psi.pressure.c = 100;
    //for(int i = 0; i < 50 * 2; ++i)
    engine.step_frame_implicit(Psi, 1, 1);

    {
        sycl::host_accessor directional_gradient_acc(engine.descent_data.descent_direction_dot_grad);
        sycl::host_accessor directional_hessian_acc(engine.descent_data.directional_hessian);
        sycl::host_accessor alpha(engine.descent_data.alpha_step);
        sycl::host_accessor multiplier(engine.descent_data.line_search_multiplier);
        sycl::host_accessor value_0(engine.descent_data.descent_value_0);
        sycl::host_accessor value(engine.descent_data.descent_value);

        std::cout << "directional derivative: " << directional_gradient_acc[0] << std::endl;
        std::cout << "directional hessian: " << directional_hessian_acc[0] << std::endl;
        std::cout << "ratio: " << -directional_gradient_acc[0] / directional_hessian_acc[0] << std::endl;
        std::cout << "value_0: " << value_0[0] << std::endl;
        std::cout << "value: " << value[0] << std::endl;
        std::cout << "alpha: " << alpha[0] << std::endl;
        std::cout << "multiplier: " << multiplier[0] << std::endl;


    }

    {
        sycl::host_accessor particle_positions_acc(engine.particle_data.positions);
        sycl::host_accessor particle_velocities_acc(engine.particle_data.velocities);
        for(size_t pid = 0; pid < particle_count; pid++)
        {
            Coordinate_t x_p = particle_positions_acc[pid];
            CHECK(!std::isnan(x_p(0)));
            CHECK(!std::isnan(x_p(1)));

            Coordinate_t v_p = particle_velocities_acc[pid];
            CHECK(!std::isnan(v_p(0)));
            CHECK(!std::isnan(v_p(1)));
        }

        sycl::host_accessor node_momenta_acc(engine.node_data.momenta);
        size_t node_count = engine.pgi_manager.get_node_count_host();
        for(size_t node_id = 0; node_id < node_count; node_id++)
        {
            Coordinate_t m_i = node_momenta_acc[node_id];
            CHECK(!std::isnan(m_i(0)));
            CHECK(!std::isnan(m_i(1)));
        }
    }


}