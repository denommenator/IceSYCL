//
// Created by robert-denomme on 8/23/24.
//

#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <vector>

namespace iceSYCL
{
template<class Container_t, class TData_t>
void host_copy_all(Container_t container, sycl::buffer<TData_t> buffer)
{
    sycl::host_accessor buffer_acc(buffer);
    std::copy(container.begin(), container.end(), buffer_acc.begin());
}

template<class TData_t>
void host_fill_all(sycl::buffer<TData_t> buffer, TData_t value)
{
    sycl::host_accessor buffer_acc(buffer);
    std::fill(buffer_acc.begin(), buffer_acc.end(), value);
}

template<int exp, class scalar_t> constexpr scalar_t pow(scalar_t base);

template<>
inline constexpr double pow<1, double>(double base){ return base; }
template<>
inline constexpr float pow<1, float>(float base){ return base; }
template<>
inline constexpr int pow<1, int>(int base){ return base; }
template<>
inline constexpr double pow<2, double>(double base){ return base * base; }
template<>
inline constexpr float pow<2, float>(float base){ return base * base; }
template<>
inline constexpr int pow<2, int>(int base){ return base * base; }
template<>
inline constexpr double pow<3, double>(double base){ return base * base * base; }
template<>
inline constexpr float pow<3, float>(float base){ return base * base * base; }
template<>
inline constexpr int pow<3, int>(int base){ return base * base * base; }

template<class TCoordinate>
void initial_vec_dot(
        sycl::queue& q,
        size_t count_limit,
        sycl::buffer<size_t>& actual_count,
        sycl::buffer<TCoordinate>& v,
        sycl::buffer<TCoordinate>& w,
        sycl::buffer<typename TCoordinate::scalar_t>& result
        )
{
//    q.submit([&](sycl::handler& h)
//     {
//        sycl::accessor result_acc(result, h);
//        h.single_task([=]()
//          {
//              result_acc[0] = TCoordinate::Zero();
//          });
//     });

    q.submit([&](sycl::handler& h)
     {
        sycl::accessor v_acc(v, h);
        sycl::accessor w_acc(w, h);
        sycl::accessor count_acc(actual_count, h);

        h.parallel_for(
                count_limit,
                sycl::reduction(
                        result,
                        h,
                        std::plus<typename TCoordinate::scalar_t>(),
                        {sycl::property_list {sycl::property::reduction::initialize_to_identity()}}),
                [=](sycl::id<1> idx, auto& sum)
                {
                    size_t id = idx[0];
                    size_t count = count_acc[0];
                    if(id >= count)
                        return;
                    sum.combine(v_acc[id].dot(w_acc[id]));
                });
     });
}

}

#endif //UTILITY_HPP
