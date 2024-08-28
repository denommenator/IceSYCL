//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>



#include <sycl/sycl.hpp>

#include <IceSYCL/coordinates.hpp>

#include <vector>
#include <functional>

TEST_CASE( "Coordinates constructor test", "[particle_node_operations]" )
{
    using namespace iceSYCL;
    using Coordinate_t = Double2DCoordinateConfiguration::Coordinate_t;

    Coordinate_t zero = Coordinate_t::Zero();
    Coordinate_t ones(1.0, 1.0);
    Coordinate_t twos = Coordinate_t(1.0, 1.0);
    //Coordinate_t twos_error = {1.0};
    Coordinate_t ones_error;
    //REQUIRE_THROWS(ones_error = Coordinate_t(1.0, 1.0, 1.0));
    //REQUIRE_THROWS(ones_error = Coordinate_t(1.0));

    sycl::buffer<Coordinate_t> my_buf(10);

    sycl::queue q;

    q.submit([&](sycl::handler& h)
    {
        sycl::accessor my_buf_acc(my_buf, h);

        h.parallel_for(10, [=](sycl::id<1> idx)
        {
            my_buf_acc[idx[0]] = Coordinate_t(100, 100);
        });
    });
    q.wait();

}