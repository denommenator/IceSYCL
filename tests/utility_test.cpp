//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>



#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>

#include <IceSYCL/utility.hpp>
#include <small_la/small_matrix.hpp>

#include <vector>



TEST_CASE( "Dot product test", "[utility_test]" )
{
    using namespace iceSYCL;
    using scalar_t = double;
    constexpr int dimension = 2;
    using Coordinate_t = small_la::small_matrix<scalar_t, dimension, 1>;

    size_t max_count = 10;
    size_t actual_count = 5;

    Coordinate_t ones = Coordinate_t::Ones();
    std::vector<Coordinate_t> v(max_count, ones);
    std::vector<Coordinate_t> w = v;
    std::vector<scalar_t> result = {10};

    std::vector<size_t> actual_count_vec = {actual_count};

    {
        sycl::buffer v_B(v);
        sycl::buffer w_B(w);
        sycl::buffer result_B(result);
        sycl::buffer actual_count_B(actual_count_vec);

        sycl::queue q{};

        initial_vec_dot(q, max_count, actual_count_B, v_B, w_B, result_B);
        q.wait();
    }

    CHECK(result[0] == actual_count * dimension);
}

TEST_CASE( "reduction without initialize", "[utility_test]" )
{
    using scalar_t = double;

    size_t count = 10;
    double init = 100;

    std::vector<scalar_t> v(count, 1.0);
    std::vector<scalar_t> result = {init};


    {
        sycl::buffer v_B(v);
        sycl::buffer result_B(result);

        sycl::queue q{};

        q.submit([&](sycl::handler& h){
            sycl::accessor v_acc(v_B, h);
            sycl::accessor result_acc(result_B, h);

            h.parallel_for(
                    count,
                    sycl::reduction(
                            result_B,
                            h,
                            std::plus<scalar_t>(),
                            {}
                            ),
                    [=](sycl::id<1> idx, auto& sum)
                    {
                        size_t id = idx[0];
                        sum += v_acc[id];
                    }
                    );
        });
        q.wait();
    }

    CHECK(result[0] == init + count);
}