//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>



#include <sycl/sycl.hpp>

#include <IceSYCL/coordinates.hpp>
#include <IceSYCL/interpolation.hpp>
#include <IceSYCL/particle_grid_interactions.hpp>
#include <IceSYCL/particle_grid_operations.hpp>

#include <vector>
#include <functional>

TEST_CASE( "one particle mass interpolation test", "[particle_node_operations]" )
{
    using namespace iceSYCL;
    using Cubic2d = CubicInterpolationScheme<Double2DCoordinateConfiguration>;
    using Coordinate_t = Cubic2d::Coordinate_t;
    using scalar_t = Cubic2d::scalar_t;

    std::vector<Coordinate_t> particle_positions = {MakeCoordinate<Cubic2d::CoordinateConfiguration>({0.0, 0.0})};
    size_t particle_count = particle_positions.size();
    scalar_t particle_initial_mass = 2.0;
    std::vector<scalar_t> particle_mass = {particle_initial_mass};
    std::vector<scalar_t> node_mass(Cubic2d::num_interactions_per_particle * particle_count, 0.0);

    ParticleGridInteractionManager<Cubic2d> pgi_manager(particle_count);

    const scalar_t h{1};
    Cubic2d interpolator(h);

    {
        sycl::buffer particle_positionsB(particle_positions);
        sycl::buffer particle_massB(particle_mass);
        sycl::buffer node_massB(node_mass);



        sycl::queue q;

        pgi_manager.update_particle_locations(q, particle_positionsB, interpolator);

        transfer_data_particles_to_grid(
            q,
            pgi_manager,
            interpolator,
            particle_massB,
            node_massB,
            particle_positionsB,
            0.0);

        q.wait();
    }

    scalar_t total_mass(0);
    for(auto m_i : node_mass)
    {
        total_mass += m_i;
        CHECK(m_i < particle_initial_mass);
    }

    CHECK(total_mass == Approx(particle_initial_mass));

}

TEST_CASE( "one particle mass interpolation and back test", "[particle_node_operations]" )
{
    using namespace iceSYCL;
    using Cubic2d = CubicInterpolationScheme<Double2DCoordinateConfiguration>;
    using Coordinate_t = Cubic2d::Coordinate_t;
    using scalar_t = Cubic2d::scalar_t;

    std::vector<Coordinate_t> particle_positions = {MakeCoordinate<Cubic2d::CoordinateConfiguration>({0.0, 0.0})};
    size_t particle_count = particle_positions.size();
    scalar_t particle_initial_mass = 2.0;
    std::vector<scalar_t> particle_mass = {particle_initial_mass};
    std::vector<scalar_t> particle_mass_back = {0};
    std::vector<scalar_t> node_mass(Cubic2d::num_interactions_per_particle * particle_count, 0.0);

    ParticleGridInteractionManager<Cubic2d> pgi_manager(particle_count);

    const scalar_t h{1};
    Cubic2d interpolator(h);

    {
        sycl::buffer particle_positionsB(particle_positions);
        sycl::buffer particle_massB(particle_mass);
        sycl::buffer particle_mass_backB(particle_mass_back);
        sycl::buffer node_massB(node_mass);



        sycl::queue q;

        pgi_manager.update_particle_locations(q, particle_positionsB, interpolator);

        transfer_data_particles_to_grid(
            q,
            pgi_manager,
            interpolator,
            particle_massB,
            node_massB,
            particle_positionsB,
            0.0);

        //transfer_data_grid_to_particles(
        //    q,
        //    pgi_manager,
        //    interpolator,
        //    node_massB,
        //    particle_mass_backB,
        //    particle_positionsB,
        //    0.0);

        q.wait();
    }

}