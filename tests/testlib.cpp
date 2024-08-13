//
// Created by robert-denomme on 8/12/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <oneapi/dpl/algorithm>
#include <small_la/small_matrix.hpp>

template<class iterator_t>
void fill_up(iterator_t begin, iterator_t end, int start_value)
{
    int i = 0;
    for(iterator_t iterator = begin; iterator != end; ++iterator)
    {
        *iterator = i + start_value;
        i++;
    }
}

TEST_CASE( "GPU Sort check", "[main]" ) {

    small_la::small_matrix<int, 2, 1> a;
    std::vector<int> vec = {3, 2, 1, 0};

    auto gpu = sycl::device(sycl::gpu_selector_v);
    std::cout << gpu.get_info<sycl::info::device::name>() << std::endl;

    dpl::execution::make_device_policy(gpu);
    std::sort(vec.begin(), vec.end());
    REQUIRE( vec[0] == 0 );
    REQUIRE( vec[1] == 1 );
    REQUIRE( vec[2] == 2 );
    REQUIRE( vec[3] == 3 );


    {
        sycl::buffer vecBuffer(vec);
        sycl::queue q(gpu);

        q.submit([&](sycl::handler& h)
        {
            sycl::accessor vecAcc(vecBuffer, h);
            h.single_task([=]()
            {
                fill_up(vecAcc.begin(), vecAcc.end(), 100);
            });

        });


        q.wait();
    }
    REQUIRE( vec[0] == 100 );
    REQUIRE( vec[1] == 101 );
    REQUIRE( vec[2] == 102 );
    REQUIRE( vec[3] == 103 );



}
