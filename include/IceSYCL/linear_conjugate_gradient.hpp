//
// Created by robert-denomme on 9/23/24.
//

#ifndef ICESYCL_LINEAR_CONJUGATE_GRADIENT_HPP
#define ICESYCL_LINEAR_CONJUGATE_GRADIENT_HPP

#include <cmath>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <sycl/sycl.hpp>


#include "utility.hpp"
#include "coordinates.hpp"
namespace iceSYCL
{

template<class TCoordinate_t>
class IApplyMatrix
{
public:
    using Coordinate_t = TCoordinate_t;
    virtual void apply_matrix(sycl::queue &q, sycl::buffer<Coordinate_t>& x, sycl::buffer<Coordinate_t>& Ax) = 0;
    virtual ~IApplyMatrix() = default;
};

template<typename TCoordinate_t>
class LinearConjugateGradient
{
public:
    using Coordinate_t = TCoordinate_t;
    using scalar_t = typename Coordinate_t::scalar_t;

    explicit LinearConjugateGradient(size_t max_vec_size) :
            max_vec_size{max_vec_size},
            alpha{1},
            beta{1},
            r_k{max_vec_size},
            r_k_next{max_vec_size},
            p_k{max_vec_size},
            p_k_next{max_vec_size},
            x_k{max_vec_size},
            x_k_next{max_vec_size},
            Ap_k{max_vec_size},
            p_kAp_k{1},
            r_kdot2{1}
    {}

    size_t max_vec_size;
    sycl::buffer<scalar_t> alpha;
    sycl::buffer<scalar_t> beta;
    sycl::buffer<Coordinate_t> r_k;
    sycl::buffer<Coordinate_t> r_k_next;
    sycl::buffer<Coordinate_t> p_k;
    sycl::buffer<Coordinate_t> p_k_next;
    sycl::buffer<Coordinate_t> x_k;
    sycl::buffer<Coordinate_t> x_k_next;
    sycl::buffer<Coordinate_t> Ap_k;
    sycl::buffer<scalar_t> p_kAp_k;
    sycl::buffer<scalar_t> r_kdot2;
    sycl::buffer<scalar_t> r_k_nextdot2;



    //Solve Ax = y
    void Solve(
        sycl::queue &q,
        sycl::buffer<size_t>& actual_vec_size,
        IApplyMatrix<TCoordinate_t>& matrix_applier,
        sycl::buffer<Coordinate_t>& y,
        sycl::buffer<Coordinate_t>& x,
        int max_num_steps
    )
    {
        //initialize x_k = 0, r_0 = -y, p_k = -r_k
        q.submit([&](sycl::handler& h){
            sycl::accessor r_k_acc(r_k, h);
            sycl::accessor p_k_acc(p_k, h);
            sycl::accessor y_acc(y, h);
            sycl::accessor x_k_acc(x_k, h);
            sycl::accessor actual_vec_size_acc(actual_vec_size, h);

            h.parallel_for(max_vec_size, [=](sycl::id<1> idx)
            {
                const size_t actual_vec_size = actual_vec_size_acc[0];
                const size_t i = idx[0];
                if (i >= actual_vec_size)
                    return;
                Coordinate_t r_0_i = -y_acc[i];
                r_k_acc[i] = r_0_i;
                p_k_acc[i] = - r_0_i;
                x_k_acc[i] = Coordinate_t::Zero();
            });
        });

        for(int k = 0; k < max_num_steps; ++k)
        {
            matrix_applier.apply_matrix(q, p_k, Ap_k);

            initial_vec_dot(q, max_vec_size, actual_vec_size, p_k, Ap_k, p_kAp_k);
            initial_vec_dot(q, max_vec_size, actual_vec_size, r_k, r_k, r_kdot2);

            q.submit([&](sycl::handler& h)
             {
                 sycl::accessor alpha_acc(alpha, h);
                 sycl::accessor r_kdot2_acc(r_kdot2, h);
                 sycl::accessor p_kAp_k_acc(p_kAp_k, h);


                 h.single_task([=]()
                   {
                       alpha_acc[0] = r_kdot2_acc[0] / p_kAp_k_acc[0];
                   });
             });

            q.submit([&](sycl::handler& h){
                sycl::accessor r_k_acc(r_k, h);
                sycl::accessor r_k_next_acc(r_k_next, h);
                sycl::accessor p_k_acc(p_k, h);
                sycl::accessor x_k_acc(x_k, h);
                sycl::accessor x_k_next_acc(x_k_next, h);
                sycl::accessor alpha_acc(alpha, h);
                sycl::accessor Ap_k_acc(Ap_k, h);
                sycl::accessor actual_vec_size_acc(actual_vec_size, h);

                h.parallel_for(max_vec_size, [=](sycl::id<1> idx)
                {
                    const size_t actual_vec_size = actual_vec_size_acc[0];
                    const size_t i = idx[0];
                    if (i >= actual_vec_size)
                        return;
                    x_k_next_acc[i] = x_k_acc[i] + alpha[0] * p_k_acc[i];
                    r_k_next_acc[i] = r_k_acc[i] + alpha[0] * Ap_k_acc[i];
                });
            });

            initial_vec_dot(q, max_vec_size, actual_vec_size, r_k_next, r_k_next, r_k_nextdot2);

            q.submit([&](sycl::handler& h)
             {
                 sycl::accessor beta_acc(beta, h);
                 sycl::accessor r_kdot2_acc(r_kdot2, h);
                 sycl::accessor r_k_nextdot2_acc(r_k_nextdot2, h);


                 h.single_task([=]()
                   {
                       beta_acc[0] = r_k_nextdot2_acc[0] / r_kdot2_acc[0];
                   });
             });

            q.submit([&](sycl::handler& h){
                sycl::accessor beta_acc(beta, h);
                sycl::accessor r_k_next_acc(r_k_next, h);
                sycl::accessor p_k_acc(p_k, h);
                sycl::accessor p_k_next_acc(p_k_next, h);
                sycl::accessor actual_vec_size_acc(actual_vec_size, h);


                h.parallel_for(max_vec_size, [=](sycl::id<1> idx)
                {
                    const size_t actual_vec_size = actual_vec_size_acc[0];
                    const size_t i = idx[0];
                    if (i >= actual_vec_size)
                        return;
                    p_k_next_acc[i] = -r_k_next_acc[i] + beta[0] * p_k_acc[i];
                });
            });

            //reset for next loop iteration
            q.submit([&](sycl::handler& h){
                sycl::accessor r_k_acc(r_k, h);
                sycl::accessor r_k_next_acc(r_k_next, h);
                sycl::accessor p_k_acc(p_k, h);
                sycl::accessor p_k_next_acc(p_k_next, h);
                sycl::accessor x_k_acc(x_k, h);
                sycl::accessor x_k_next_acc(x_k_next, h);
                sycl::accessor actual_vec_size_acc(actual_vec_size, h);


                h.parallel_for(max_vec_size, [=](sycl::id<1> idx)
                {
                    const size_t actual_vec_size = actual_vec_size_acc[0];
                    const size_t i = idx[0];
                    if (i >= actual_vec_size)
                        return;
                    p_k_acc[i] = p_k_next_acc[i];
                    r_k_acc[i] = r_k_next_acc[i];
                    x_k_acc[i] = x_k_next_acc[i];
                });
            });

        }

        //store results!
        q.submit([&](sycl::handler& h){
            sycl::accessor x_acc(x, h);
            sycl::accessor x_k_next_acc(x_k_next, h);
            sycl::accessor actual_vec_size_acc(actual_vec_size, h);


            h.parallel_for(max_vec_size, [=](sycl::id<1> idx)
            {
                const size_t actual_vec_size = actual_vec_size_acc[0];
                const size_t i = idx[0];
                if (i >= actual_vec_size)
                    return;
                x_acc[i] = x_k_next_acc[i];
            });
        });
    }


};

}

#endif //ICESYCL_LINEAR_CONJUGATE_GRADIENT_HPP
