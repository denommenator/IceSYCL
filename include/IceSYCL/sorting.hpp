//
// Created by robert-denomme on 8/14/24.
//

#ifndef SORTING_HPP
#define SORTING_HPP

#include <small_la/small_matrix.hpp>


namespace iceSYCL::sorting
{
class Infinity{};

template<class TBase>
struct Extended
{
    using Base_t = TBase;
    using This_t = Extended<Base_t>;
    std::variant<TBase, Infinity> value;

    Extended(Base_t a) : value{a}{}
    Extended(Infinity) : value{Infinity{}}{}

    template<class Comparer>
    static bool compare(Base_t a, Base_t b, Comparer comparer)
    {
        return comparer(a, b);
    }

    template<class Comparer>
    static bool compare(Base_t, Infinity, Comparer comparer)
    {
        return true;
    }

    template<class Comparer>
    static bool compare(Infinity, Base_t, Comparer comparer)
    {
        return false;
    }

    template<class Comparer>
    static bool compare(Infinity, Infinity, Comparer comparer)
    {
        return false;
    }

    template<class Comparer_t>
    static bool compare_extended(This_t a, This_t b, Comparer_t comparer)
    {
        return std::visit([](auto f, auto g, Comparer_t comparer_dummy){return compare(f, g, comparer_dummy);}, a.value, b.value, std::variant<Comparer_t>(comparer));
    }

    template<class Comparer_t>
    static auto make_extended_comparer(Comparer_t comparer)
    {
        return [=](This_t a, This_t b){return compare_extended(a, b, comparer);};
    }

};


}



#endif //SORTING_HPP
