//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>

#include <IceSYCL/interpolation.hpp>
#include <IceSYCL/particle_node_interactions.hpp>


TEST_CASE( "Device sorting test", "[particle_node_interactions]" )
{
    const size_t count = 20;
    sycl::buffer<int> buf{count};

    sycl::queue q(sycl::gpu_selector_v);

    q.submit([&](sycl::handler& h)
    {
        sycl::accessor buf_acc(buf, h);
        h.parallel_for(count,[=](sycl::id<1> i)
            {
                buf_acc[i] = count - 1 - i[0];
            }
        );
    });

    auto dpl_policy = oneapi::dpl::execution::make_device_policy<class Sorterer>(q);
    oneapi::dpl::sort(dpl_policy,
        oneapi::dpl::begin(buf),
        oneapi::dpl::end(buf)
        );

    q.wait();

    sycl::host_accessor buf_acc(buf);
    for(size_t i = 0; i < count; i++)
    {
        REQUIRE(buf_acc[i] == i);
    }

}

TEST_CASE( "Particle Node Interaction test", "[particle_node_interactions]" )
{
    using namespace iceSYCL;
    using Cubic2d = CubicInterpolationScheme<Double2DCoordinateConfiguration>;

    size_t particle_count = 1;
    std::array<Cubic2d::Coordinate_t, 1> particles =
        {
        MakeCoordinate<Cubic2d::CoordinateConfiguration>({0.0, 0.0})
        };

    sycl::buffer particles_B(particles);

    const Cubic2d::scalar_t h = 1.0;
    Cubic2d interpolator{h};

    ParticleNodeInteractionManager<Cubic2d> pni(particle_count);
    sycl::queue q(sycl::gpu_selector_v);
    pni.generate_particle_node_interactions(q, particles_B, interpolator);
    q.wait();



}
