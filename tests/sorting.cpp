//
// Created by robert-denomme on 8/12/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>

#include <IceSYCL/sorting.hpp>

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

    for(auto a : vec)
    {
        std::cout << (is_infinite_extended(a) ? "Infinite" : std::to_string(std::get<int>(a))) << ", ";
    }

    std::cout << std::endl;



}

TEST_CASE( "Extended type check", "[sorting]" )
{
    using namespace iceSYCL::sorting;

    std::vector<Extended<int>> vec = { 1, 2, 3, iceSYCL::sorting::Infinity{}};

    auto comparer_unextended = [](int a, int b){return a < b;};
    auto comparer = Extended<int>::make_extended_comparer(comparer_unextended);
    std::sort(vec.begin(), vec.end(), comparer);
}
