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
};

}
#endif //ICESYCL_CONSTITUTIVE_MODELS_H
