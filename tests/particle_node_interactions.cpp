//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/ranges>

#include <IceSYCL/interpolation.hpp>
#include <IceSYCL/particle_node_interactions.hpp>

#include <vector>
#include <functional>

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

TEST_CASE( "ranges", "[particle_node_interactions]" )
{
    constexpr size_t count = 10;
    std::array<int, count> vec_list = {0, 0, 0, 1, 0, 0, 1, 1, 1, 0};
    std::array<size_t, count> segment_ids;
    std::array<size_t, count> segment_ids_desired = {0, 0, 0, 1, 1, 1, 2, 3, 4, 4};

    namespace rng = dpl::experimental::ranges;
    namespace views = oneapi::dpl::experimental::ranges::views;
    {
        sycl::buffer vec_listB(vec_list);
        sycl::buffer segment_idsB(segment_ids);
    }
}

TEST_CASE( "ranges device test", "[particle_node_interactions]" )
{
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result = data;
    std::fill(result.begin(), result.end(), 0);

    {
        namespace rng = dpl::experimental::ranges;

        sycl::buffer vec_listB(data);
        sycl::buffer resultB(result);

        //rng::all_view vec_list_range(vec_listB);
        //rng::all_view result_range(resultB);
        //rng::copy(dpl::execution::dpcpp_default, rng::views::all_read(vec_listB), rng::views::all_write(resultB));
        rng::transform(dpl::execution::dpcpp_default, rng::views::all_read(vec_listB), rng::views::all_write(resultB), [](const int& x){return x+1;});
    }

    for(int i = 0; i < data.size(); i++)
    {
        REQUIRE(result[i] == data[i] + 1);
    }

}


TEST_CASE( "ranges zip test", "[particle_node_interactions]" )
{
    std::vector<int> dataA = {1, 2, 3, 4, 5};
    std::vector<int> dataB = {10, 11, 12, 13};
    std::vector<int> result = dataB;

    {
        namespace rng = dpl::experimental::ranges;

        sycl::buffer bufferA(dataA);
        sycl::buffer bufferB(dataB);
        sycl::buffer resultB(result);

        auto zip = rng::zip_view(rng::views::all_read(bufferA), rng::views::all_read(bufferB));
        rng::transform(dpl::execution::dpcpp_default,
            zip,
            rng::views::all_write(resultB),
            [](std::tuple<const int&, const int&> el){return std::get<0>(el) + std::get<1>(el);}
            );
        //rng::all_view vec_list_range(vec_listB);
        //rng::all_view result_range(resultB);
        //rng::copy(dpl::execution::dpcpp_default, rng::views::all_read(vec_listB), rng::views::all_write(resultB));
        //rng::transform(dpl::execution::dpcpp_default, rng::views::all_read(vec_listB), rng::views::all_write(resultB), [](const int& x){return x+1;});
    }

    for(int i = 0; i < dataB.size(); i++)
    {
        REQUIRE(result[i] == dataA[i] + dataB[i]);
    }

}

/*
TEST_CASE( "inclusive scan device test", "[particle_node_interactions]" )
{
    constexpr size_t count = 10;
    std::array<int, count> vec_list = {0, 0, 0, 1, 0, 0, 1, 1, 1, 0};
    std::array<size_t, count> segment_ids;
    segment_ids[0] = 0;
    std::array<size_t, count> segment_ids_desired = {0, 0, 0, 1, 1, 1, 2, 3, 4, 4};

    namespace rng = dpl::experimental::ranges;
    namespace views = oneapi::dpl::experimental::ranges::views;
    {

        sycl::buffer vec_listB(vec_list);
        sycl::buffer segment_idsB(segment_ids);

        auto range_0 = rng::take_view(views::all_read(vec_listB), count - 1);
        auto range_1 = rng::drop_view(views::all_read(vec_listB), 1);

        auto zip_view = dpl::experimental::ranges::zip_view(range_0, range_1);

        auto is_different = [](const int& a, const int& b)->size_t{return a == b ? 1 : 0;};
        auto is_different_pair = [=](std::tuple<const int&, const int&> z)->size_t{return is_different(std::get<0>(z), std::get<1>(z));};
        //rng::transform(dpl::execution::dpcpp_default,
        //    zip_view,
        //    views::all_write(segment_ids) | views::drop(1),
        //    is_different_pair
        //    );
//
        //rng::inclusive_scan(dpl::execution::dpcpp_default,
        //    );

        rng::transform_inclusive_scan(dpl::execution::dpcpp_default,
            zip_view,
            views::all_write(segment_ids) | views::drop(1),
            std::plus<size_t>{},
            is_different_pair,
            0
        );

    }

    for(auto seg_id = segment_ids.begin(), seg_id_desired = segment_ids_desired.begin(); seg_id != segment_ids.end(); ++seg_id, ++seg_id_desired)
    {
        REQUIRE(*seg_id == *seg_id_desired);
    }
}
*/

TEST_CASE( "segment_id logic test", "[particle_node_interactions]" )
{
    using namespace iceSYCL;

    constexpr size_t count = 10;
    std::array<int, count> vec_list = {1, 1, 1, 7, 7, 7, 8, 9, 10, 10};
    std::array<size_t, count> segment_ids;
    std::array<size_t, count> segment_ids_desired = {0, 0, 0, 1, 1, 1, 2, 3, 4, 4};

    auto pred = [](int a, int b){return a == b;};
    auto are_different_pair = [=](int a, int b)->size_t
    {
        return pred(a, b) ? 0 : 1;
    };

    oneapi::dpl::zip_iterator adjacent = dpl::make_zip_iterator(vec_list.begin(), std::next(vec_list.begin()));
    using zip_t = decltype(adjacent);

    auto are_different_zip = [=](typename std::iterator_traits<zip_t>::value_type zipped_adjacent)->size_t
    {
        return are_different_pair(oneapi::dpl::get<0>(zipped_adjacent), oneapi::dpl::get<1>(zipped_adjacent));
    };

    oneapi::dpl::transform_iterator are_different_it = oneapi::dpl::make_transform_iterator(adjacent, are_different_zip);

    std::exclusive_scan(are_different_it, are_different_it + count, segment_ids.begin(), 0);

    for(size_t i = 0; i < count; i++)
    {
        REQUIRE(segment_ids[i] == segment_ids_desired[i]);
    }

}
/*
TEST_CASE( "segment_id function test", "[particle_node_interactions]" )
{
    using namespace iceSYCL;

    constexpr size_t count = 10;
    std::array<int, count> vec_list = {1, 1, 1, 7, 7, 7, 8, 9, 10, 10};
    std::array<size_t, count> segment_ids;
    std::array<size_t, count> segment_ids_desired = {0, 0, 0, 1, 1, 1, 2, 3, 4, 4};



    auto pred = [](int a, int b){return a == b;};

    {
        sycl::buffer vec_listB(vec_list);
        sycl::buffer segment_idsB(segment_ids);
        segment_id(dpl::execution::dpcpp_default,
            dpl::begin(vec_listB),
            dpl::end(vec_listB),
                dpl::begin(segment_idsB),
                pred
            );
    }

    for(size_t i = 0; i < count; i++)
    {
        REQUIRE(segment_ids[i] == segment_ids_desired[i]);
    }

}
*/

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
