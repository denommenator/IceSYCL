//
// Created by robert-denomme on 9/3/24.
//

#ifndef ICESYCL_CONSTITUTIVE_MODELS_H
#define ICESYCL_CONSTITUTIVE_MODELS_H

#include <cmath>

#include "coordinates.hpp"


namespace iceSYCL
{


template<class TCoordinateConfiguration>
class IdealGasFromDensity
{
public:
    using CoordinateConfiguration = TCoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    scalar_t unit_density;
    scalar_t k_stiffness;

    scalar_t value(scalar_t density) const
    {
        return k_stiffness * density / unit_density;
    }

    scalar_t value_prime(scalar_t density) const
    {
        return k_stiffness / unit_density;
    }
};

template<class TCoordinateConfiguration>
class TaitPressureFromDensity
{
public:
    using CoordinateConfiguration = TCoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    scalar_t unit_density;
    scalar_t gamma;
    //c = speed of sound
    scalar_t c;

    scalar_t value(scalar_t density) const
    {
        return unit_density * c * c / gamma * (std::pow(density / unit_density, gamma) - 1.0);
    }

    scalar_t value_prime(scalar_t density) const
    {
        return c * c * std::pow(density / unit_density, gamma - 1);
    }
};

template<class PressureFromDensity>
class DensityBasedConstitutiveModel
{
public:
    using CoordinateConfiguration = typename PressureFromDensity::CoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    using CoordinateMatrix_t = typename CoordinateConfiguration::CoordinateMatrix_t;

    PressureFromDensity pressure;
    scalar_t value(CoordinateMatrix_t F) const
    {
        const scalar_t j = det(F);
        //assumes that rest_mass / rest_volume == unit_density
        //density = mass / volume = rest_mass / (j * rest_volume) = unit_density / j
        const scalar_t density = pressure.unit_density / j;

        return pressure.value(density);
    }

    CoordinateMatrix_t PK(CoordinateMatrix_t F) const
    {
        const scalar_t j = small_la::det(F);
        //assumes that rest_mass / rest_volume == unit_density
        //density = mass / volume = rest_mass / (j * rest_volume) = unit_density / j
        const scalar_t density = pressure.unit_density / j;

        /*
        CoordinateMatrix_t U,V;
        CoordinateMatrix_t Sigma;
        small_la::SVD(F, U, Sigma, V);
        //F = U * Sigma * V^t
        CoordinateMatrix_t D = CoordinateMatrix_t::Zero();
        if constexpr (CoordinateConfiguration::Dimension == 2)
        {
            D(0,0) = Sigma(1,1);
            D(1,1) = Sigma(0,0);
        }
        else if constexpr (CoordinateConfiguration::Dimension == 3)
        {
            scalar_t sigma_0 = Sigma(0,0);
            scalar_t sigma_1 = Sigma(1,1);
            scalar_t sigma_2 = Sigma(2,2);

            D(0,0) = sigma_1 * sigma_2;
            D(1,1) = sigma_0 * sigma_2;
            D(2,2) = sigma_0 * sigma_1;

        }
        else
        {
            static_assert((CoordinateConfiguration::Dimension == 2) || (CoordinateConfiguration::Dimension == 3), "Only dimensions 2 and 3 supported :-)");
        }

        CoordinateMatrix_t del_j_del_F = U * D * V.transpose();
        */
        CoordinateMatrix_t del_j_del_F = j * small_la::inverse(F).transpose();

        return pressure.value_prime(density) * (-pressure.unit_density / (j * j)) * del_j_del_F;
    }

};

template<class TCoordinateConfiguration>
class FixedCorotated {
public:
    using CoordinateConfiguration = TCoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    using CoordinateMatrix_t = typename CoordinateConfiguration::CoordinateMatrix_t;

    scalar_t mu;
    scalar_t lambda;


    scalar_t value(CoordinateMatrix_t F) const
    {
        CoordinateMatrix_t U, Sigma, V;
        small_la::SVD(F, U, Sigma, V);

        scalar_t ret = 0;

        for(int i = 0; i < CoordinateMatrix_t::num_rows; ++i)
        {
            ret += mu * (Sigma(i,i) - 1.0) * (Sigma(i,i) - 1.0);
        }

        scalar_t J = small_la::det(F);
        ret += lambda / 2.0 * (J - 1) * (J - 1);
        return ret;
    }

    CoordinateMatrix_t PK(CoordinateMatrix_t F) const
    {
        CoordinateMatrix_t R, S;
        small_la::PolarDecomposition(F, R, S);

        scalar_t J = small_la::det(F);

        return 2 * mu * (F - R) + lambda * (J - 1.0) * J * small_la::inverse(F).transpose();
    }

    CoordinateMatrix_t apply_hessian(CoordinateMatrix_t F, CoordinateMatrix_t delta_F) const
    {
        CoordinateMatrix_t R, S;
        small_la::PolarDecomposition(F, R, S);

        CoordinateMatrix_t LHS = R.transpose() * delta_F - delta_F.transpose() * R;

        if constexpr(CoordinateConfiguration::Dimension == 2)
        {
            scalar_t denom = S(0,0) + S(1,1);
            //TODO division by zero is possible here, but I don't know
            //how to compute R^T * delta_R when this happens
            scalar_t r = LHS(0,1) / denom;
            CoordinateMatrix_t R_trans_delta_R = CoordinateMatrix_t::Zero();
            R_trans_delta_R(0, 1) = r;
            R_trans_delta_R(1, 0) = -r;

            CoordinateMatrix_t delta_R = R * R_trans_delta_R;

            scalar_t J = small_la::det(F);

            CoordinateMatrix_t delta_J_F_mtrans = CoordinateMatrix_t::Zero();
            delta_J_F_mtrans(0, 0) = delta_F(1, 1);
            delta_J_F_mtrans(0, 1) = -delta_F(1, 0);
            delta_J_F_mtrans(1, 0) = -delta_F(0, 1);
            delta_J_F_mtrans(1, 1) = delta_F(0, 0);

            CoordinateMatrix_t J_F_mtrans = CoordinateMatrix_t::Zero();
            J_F_mtrans(0, 0) = F(1, 1);
            J_F_mtrans(0, 1) = -F(1, 0);
            J_F_mtrans(1, 0) = -F(0, 1);
            J_F_mtrans(1, 1) = F(0, 0);

            scalar_t delta_J =
                    J_F_mtrans(0, 0) * delta_F(0, 0) +
                    J_F_mtrans(0, 1) * delta_F(0, 1) +
                    J_F_mtrans(1, 0) * delta_F(1, 0) +
                    J_F_mtrans(1, 1) * delta_F(1, 1);

            return 2 * mu * delta_F - 2 * mu * delta_R +
                lambda * delta_J * J_F_mtrans + lambda * (J - 1) * delta_J_F_mtrans;


        } else
        {
            static_assert(CoordinateConfiguration::Dimension == 2, "Only dimension 2 is implemented currently.");
        }
    }
};

template<class TCoordinateConfiguration>
class SnowPlasticity {
public:
    using CoordinateConfiguration = TCoordinateConfiguration;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    using CoordinateMatrix_t = typename CoordinateConfiguration::CoordinateMatrix_t;

    scalar_t mu_0;
    scalar_t lambda_0;
    scalar_t xi;
    scalar_t theta_c;
    scalar_t theta_s;
    scalar_t max_exp;

    CoordinateMatrix_t F_P;
    CoordinateMatrix_t F_E;

    scalar_t get_mu(const scalar_t J_P) const
    {
        scalar_t exponent = std::min(xi * (1.0 - J_P), max_exp);
        return mu_0 * std::exp(exponent);
    }

    scalar_t get_lambda(const scalar_t J_P) const
    {
        scalar_t exponent = std::min(xi * (1.0 - J_P), max_exp);
        return lambda_0 * std::exp(exponent);
    }

    scalar_t get_sigma_E(const scalar_t sigma_E_prev) const
    {
        return std::clamp(sigma_E_prev, 1 - theta_c, 1 + theta_s);
    }

    void update(const CoordinateMatrix_t F)
    {
        CoordinateMatrix_t F_E_tilde_next = F * small_la::inverse(F_P);

        constexpr int dimension = CoordinateMatrix_t::num_rows;
        CoordinateMatrix_t U, Sigma_E = CoordinateMatrix_t::Zero(), Sigma_E_tilde, V;
        small_la::SVD(F_E_tilde_next, U, Sigma_E_tilde, V);
        for(int i = 0; i < dimension; ++i)
        {
            Sigma_E(i, i) = get_sigma_E(Sigma_E_tilde(i, i));
        }

        CoordinateMatrix_t F_E_next = U * Sigma_E * V.transpose();
        CoordinateMatrix_t F_P_next = small_la::inverse(F_E_next) * F;

        F_E = F_E_next;
        F_P = F_P_next;
    }


    CoordinateMatrix_t PK() const
    {
        CoordinateMatrix_t R, S;
        small_la::PolarDecomposition(F_E, R, S);

        scalar_t J_E = small_la::det(F_E);
        scalar_t J_P = small_la::det(F_P);

        scalar_t mu = get_mu(J_P);
        scalar_t lambda = get_lambda(J_P);

        return 2 * mu * (F_E - R) + lambda * (J_E - 1.0) * J_E * small_la::inverse(F_E).transpose();
    }
};

}
#endif //ICESYCL_CONSTITUTIVE_MODELS_H
