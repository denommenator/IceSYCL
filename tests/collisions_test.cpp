//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>



#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>

#include <IceSYCL/coordinates.hpp>

#include <vector>
#include <functional>
#include <variant>

struct A{double a;};
struct B{double b;};

double my_func(A a){return a.a;}
double my_func(B b){return b.b;}

//template<class X>
//double my_func_visitor(X x){return my_func(x);}



//
//TEST_CASE( "Coordinates constructor test", "[particle_node_operations]" )
//{
//    using Mixed_t = typename std::variant<A, B>;
//    std::vector<Mixed_t> mixed = {A{0.0}, B{1.0}, A{2.0}};
//    std::vector<double> result(mixed.size(), 0.0);
//
//    size_t count = mixed.size();
//
////    for(size_t i = 0 ; i < count; ++i)
////    {
////        Mixed_t m = mixed[i];
////        result[i] = std::visit([](auto x){return my_func(x);}, m);
////    }
//    {
//        sycl::buffer mixed_b(mixed);
//        sycl::buffer result_b(result);
//
//        sycl::queue q;
//        q.submit([&](sycl::handler& h){
//            sycl::accessor mixed_acc(mixed_b);
//            sycl::accessor result_acc(result_b);
//
//            h.parallel_for(count, [=](sycl::id<1> idx){
//                size_t id = idx[0];
////                try {
//                    result_acc[id] = std::visit([](auto x) { return my_func(x); }, mixed_acc[id]);
////                }
////                catch (const std::bad_variant_access& e)
////                {}
//            });
//        });
//    }
//
//    for(size_t i = 0; i < count; ++i)
//    {
//        CHECK(result[i] == Approx(i));
//    }
//
//}