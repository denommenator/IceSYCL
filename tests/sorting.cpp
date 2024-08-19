//
// Created by robert-denomme on 8/12/24.
//
#define CATCH_CONFIG_MAIN





#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>

#include <catch2/catch.hpp>


#include <sycl/sycl.hpp>

#include <IceSYCL/sorting.hpp>

#include <vector>

class Infinity {};
using ExtendedInt = std::variant<int, Infinity>;

bool is_infinite(int) {return false;}
bool is_infinite(Infinity) {return true;}

bool is_infinite_extended(ExtendedInt a)
{
    return std::visit([=](auto a){return is_infinite(a);}, a);
}

template<class TComparer>
bool compare(int a, int b, TComparer comparer)
{
    return comparer(a, b);
}
template<class TComparer>
bool compare(int, Infinity, TComparer comparer)
{
    return true;
}
template<class TComparer>
bool compare( Infinity, int, TComparer comparer)
{
    return false;
}
template<class TComparer>
bool compare( Infinity, Infinity, TComparer comparer)
{
    return false;
}

template<class TComparer>
bool compare_extended(ExtendedInt a, ExtendedInt b, TComparer comparer)
{
    return std::visit([](auto f, auto g, TComparer comparer_dummy){return compare(f, g, comparer_dummy);}, a, b, std::variant<TComparer>(comparer));
}

template<class TFn>
int my_apply(TFn f)
{
    return f();
}

TEST_CASE( "buffer of std::variant check", "[sorting]" ) {

    std::vector<ExtendedInt> vec = {Infinity{}, 1, 2, 3};

    auto comparer = [](int a, int b){return a < b;};
    auto comparer_extended = [=](ExtendedInt a, ExtendedInt b){return compare_extended(a, b, comparer);};

    std::sort(vec.begin(), vec.end(), comparer_extended);

    /*
    for(auto a : vec)
    {
        std::cout << (is_infinite_extended(a) ? "Infinite" : std::to_string(std::get<int>(a))) << ", ";
    }

    std::cout << std::endl;
    */

    REQUIRE(std::get<int>(vec[0]) == 1);
    REQUIRE(std::get<int>(vec[1]) == 2);
    REQUIRE(std::get<int>(vec[2]) == 3);
    REQUIRE(is_infinite_extended(vec[3]));


}

TEST_CASE( "Extended type check", "[sorting]" )
{
    using namespace iceSYCL::sorting;

    std::vector<Extended<int>> vec = { 1, 2, 3, iceSYCL::sorting::Infinity{}};

    auto comparer_unextended = [](int a, int b){return a < b;};
    auto comparer = Extended<int>::make_extended_comparer(comparer_unextended);
    std::sort(vec.begin(), vec.end(), comparer);


}

TEST_CASE("oneapi::dpl::sort", "[sorting]")
{
    size_t count = 100;
    sycl::buffer<int> buf(count);
    {
        sycl::host_accessor buf_acc(buf);
        std::iota(buf_acc.begin(), buf_acc.end(), 0);
    }

    auto policy = oneapi::dpl::execution::device_policy(sycl::device(sycl::gpu_selector_v));
    oneapi::dpl::transform(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), oneapi::dpl::begin(buf), [](int x){return -x;});

    oneapi::dpl::sort(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf));

    oneapi::dpl::counting_iterator<int> counter(-99);
    {
        sycl::host_accessor buf_acc(buf);
        for(auto x : buf_acc)
        {
            REQUIRE(x == *counter);
            ++counter;
        }
    }

    std::cout << std::endl;
}
/*
TEST_CASE("oneapi::dpl::scan", "[sorting]")
{
    size_t count = 100;
    sycl::buffer<int> buf(count);
    sycl::buffer<int> buf_prefix_sum(count);
    {
        sycl::host_accessor buf_acc(buf);
        std::fill(buf_acc.begin(), buf_acc.end(), 1);
    }

    auto policy = oneapi::dpl::execution::dpcpp_default;
    oneapi::dpl::exclusive_scan(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), oneapi::dpl::begin(buf_prefix_sum));

    //oneapi::dpl::sort(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf));

}

*/