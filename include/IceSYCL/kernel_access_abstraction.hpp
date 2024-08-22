//
// Created by robert-denomme on 8/22/24.
//

#ifndef KERNEL_ACCESS_ABSTRACTION_HPP
#define KERNEL_ACCESS_ABSTRACTION_HPP

#include <sycl/sycl.hpp>

namespace iceSYCL
{

    template<class TData_t, bool TIsDevice>
    using Accessor_t = typename std::conditional<TIsDevice, sycl::accessor<TData_t>, sycl::host_accessor<TData_t>>::type;

}
#endif //KERNEL_ACCESS_ABSTRACTION_HPP
