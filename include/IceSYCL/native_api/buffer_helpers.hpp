//
// Created by robert-denomme on 8/27/24.
//

#ifndef BUFFER_HELPERS_HPP
#define BUFFER_HELPERS_HPP

#include <vector>

#include <IceSYCL/coordinates.hpp>

//using Configuration2D = iceSYCL::Double2DCoordinateConfiguration;
//using Coordinate2D = Configuration2D::Coordinate_t;
//using MakeCoordinate2D = iceSYCL::MakeCoordinate<Configuration2D>;

template<class CoordinateConfiguration>
void copy(const typename CoordinateConfiguration::scalar_t* data_raw, std::vector<typename CoordinateConfiguration::Coordinate_t> coordinates_vec)
{
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    size_t count = coordinates_vec.size();

    for(size_t i = 0; i < count; ++i)
    {
        Coordinate_t coord_i = Coordinate_t::Zero();
        for(int dim = 0; dim < CoordinateConfiguration::Dimension; ++dim)
        {
            coord_i(dim) = *data_raw[CoordinateConfiguration::Dimension * i + dim];
        }
        coordinates_vec[i] = coord_i;
    }
}

template<class CoordinateConfiguration>
std::vector<typename CoordinateConfiguration::Coordinate_t> to_coordinate_vector(size_t count, const double* data_raw)
{
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    std::vector<Coordinate_t> res(count, Coordinate_t::Zero());
    copy(data_raw, res);
    return res;
}

template<class CoordinateConfiguration>
void copy(const std::vector<typename CoordinateConfiguration::Coordinate_t> coordinates_vec, typename CoordinateConfiguration::scalar_t* data_raw)
{
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    size_t count = coordinates_vec.size();

    for(size_t i = 0; i < count; ++i)
    {
        Coordinate_t coord_i = Coordinate_t::Zero();
        for(int dim = 0; dim < CoordinateConfiguration::Dimension; ++dim)
        {
            *data_raw[CoordinateConfiguration::Dimension * i + dim] = coordinates_vec[i](dim);
        }
    }
}

template<class CoordinateConfiguration>
std::vector<typename CoordinateConfiguration::scalar_t> to_scalar_vector(size_t count, const double* data_raw)
{
    using scalar_t = typename CoordinateConfiguration::scalar_t;
    std::vector<scalar_t> res(count, 0.0);
    std::copy(data_raw, data_raw + count, res.begin());
    return res;
}

#endif //BUFFER_HELPERS_HPP
