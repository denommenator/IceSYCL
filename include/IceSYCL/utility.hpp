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

}

#endif //UTILITY_HPP
