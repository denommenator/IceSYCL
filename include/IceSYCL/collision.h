//
// Created by robert-denomme on 8/30/24.
//

#ifndef ICESYCL_COLLISION_H
#define ICESYCL_COLLISION_H
namespace iceSYCL
{
template<class CoordinateConfiguration>
struct ElasticCollisionWall
{
public:
    using Coordinate_t = typename CoordinateConfiguration::Coordinate_t;
    using scalar_t = typename CoordinateConfiguration::scalar_t;
//public:
//    ElasticCollisionWall() = default;
//    ~ElasticCollisionWall() = default;
//    ElasticCollisionWall(const ElasticCollisionWall<CoordinateConfiguration>&) = default;
//    ElasticCollisionWall<CoordinateConfiguration>& operator=(const ElasticCollisionWall<CoordinateConfiguration>& ) = default;
//    ElasticCollisionWall(ElasticCollisionWall<CoordinateConfiguration>&&) = default;
//    ElasticCollisionWall<CoordinateConfiguration>& operator=(ElasticCollisionWall<CoordinateConfiguration>&& ) = default;
    
    scalar_t value(Coordinate_t p)
    {
        if(dot(p - p_0, n) > 0.0)
            return 0.0;
        return 0.5 * k_stiffness * dot(p - p_0, n) * dot(p - p_0, n);
    }

    Coordinate_t gradient(Coordinate_t p)
    {
        if(dot(p - p_0, n) > 0.0)
            return Coordinate_t ::Zero();
        return k_stiffness * dot(p - p_0, n) * n;
    }

    Coordinate_t apply_hessian(Coordinate_t p, Coordinate_t delta_p)
    {
        if(dot(p - p_0, n) > 0.0)
            return Coordinate_t ::Zero();
        return k_stiffness * dot(n, delta_p) * n;
    }


    Coordinate_t n;
    Coordinate_t p_0;
    scalar_t k_stiffness;
    
    
};
}

#endif //ICESYCL_COLLISION_H
