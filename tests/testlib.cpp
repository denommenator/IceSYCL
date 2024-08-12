//
// Created by robert-denomme on 8/12/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <oneapi/dpl/algorithm>

TEST_CASE( "GPU Sort check", "[main]" ) {

    std::vector<int> vec = {3, 2, 1, 0};

    auto gpu = sycl::device(sycl::gpu_selector_v);
    std::cout << gpu.get_info<sycl::info::device::name>() << std::endl;

    dpl::execution::make_device_policy(gpu);
    std::sort(vec.begin(), vec.end());
    REQUIRE( vec[0] == 0 );
    REQUIRE( vec[1] == 1 );
    REQUIRE( vec[2] == 2 );
    REQUIRE( vec[3] == 3 );
}
