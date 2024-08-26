//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


#include <sycl/sycl.hpp>

#include <vector>


TEST_CASE( "basic access pattern", "[accessor abstraction]" )
{
    size_t count = 10;
    std::vector<int> A = {10, 10, 10, 10, 10};
    std::vector<float> B = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<float> C(A.size(), 0.0);

    {
        sycl::buffer<int> bufA(A);
        sycl::buffer<float> bufB(B);
        sycl::buffer<float> bufC(C);

        sycl::accessor bufA_acc(bufA);
        sycl::accessor bufB_acc(bufB);
        sycl::accessor bufC_acc(bufC);

        sycl::queue q;

        q.submit([&](sycl::handler& h)
        {
            h.require(bufA_acc);
            h.require(bufB_acc);
            h.require(bufC_acc);

            h.parallel_for(count, [=](sycl::id<1> idx)
            {
                bufC_acc[idx] = bufA_acc[idx] * bufB_acc[idx];
            });
        });
        q.wait();
    }

    for(int i = 0; i < A.size(); i++)
    {
        CHECK(C[i] == Approx(A[i] * B[i]));
    }

}

TEST_CASE( "Abstract basic access pattern", "[accessor abstraction]" )
{
    size_t count = 10;
    class Data
    {
    public:
        Data(std::vector<int>& A, std::vector<float>& B, std::vector<float>& C) :


        bufA(A),
        bufB(B),
        bufC(C),
        access{bufA, bufB, bufC, A.size()}
        {

        }

        sycl::buffer<int> bufA;
        sycl::buffer<float> bufB;
        sycl::buffer<float> bufC;


        struct Access
        {
            void give_access(sycl::handler& h)
            {
                h.require(bufA_acc);
                h.require(bufB_acc);
                h.require(bufC_acc);
            }

            int& A(size_t i) const {return bufA_acc[i];}
            float& B(size_t i) const {return bufB_acc[i];}
            float& C(size_t i) const {return bufC_acc[i];}

            sycl::accessor<int> bufA_acc;
            sycl::accessor<float> bufB_acc;
            sycl::accessor<float> bufC_acc;
            const size_t size;
        };
        Access access;


    };

    std::vector<int> A = {10, 10, 10, 10, 10};
    std::vector<float> B = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<float> C(A.size(), 0);



    {

        Data data(A, B, C);
        sycl::queue q;

        q.submit([&](sycl::handler& h)
        {
            auto data_access = data.access;
            data_access.give_access(h);

            h.parallel_for(count, [=](sycl::id<1> idx)
            {
                size_t i = idx[0];
                data_access.C(i) = data_access.A(i) * data_access.B(i);

                size_t size = data_access.size;
            });
        });

        q.submit([&](sycl::handler& h)
        {
            auto data_access = data.access;
            data_access.give_access(h);

            h.parallel_for(count, [=](sycl::id<1> idx)
            {
                size_t i = idx[0];
                data_access.C(i) = 2 * data_access.C(i);
            });
        });
        q.wait();
    }

    for(int i = 0; i < A.size(); i++)
    {
        CHECK(C[i] == Approx(2.0 * A[i] * B[i]));
    }

}

class DataTest2
{
public:
    DataTest2(std::vector<int>& A, std::vector<float>& B, std::vector<float>& C) :

    size{A.size()},
    bufA(A),
    bufB(B),
    bufC(C),
    device_access{bufA, bufB, bufC, A.size()}

    {

    }

    size_t size;
    sycl::buffer<int> bufA;
    sycl::buffer<float> bufB;
    sycl::buffer<float> bufC;


    template<bool TIsDevice>
    struct Access
    {
        template<class TData>
        using Accessor_t = std::conditional_t<TIsDevice, sycl::accessor<TData>, sycl::host_accessor<TData>>;


        void give_access(sycl::handler& h)
        {
            h.require(bufA_acc);
            h.require(bufB_acc);
            h.require(bufC_acc);
        }

        int& A(size_t i) const {return bufA_acc[i];}
        float& B(size_t i) const {return bufB_acc[i];}
        float& C(size_t i) const {return bufC_acc[i];}

        Accessor_t<int> bufA_acc;
        Accessor_t<float> bufB_acc;
        Accessor_t<float> bufC_acc;
        const size_t size;
    };
    Access<true> device_access;
    Access<false> get_host_access()
    { return Access<false> {bufA, bufB, bufC, size}; }


};

TEST_CASE( "Abstract basic access pattern with host side access", "[accessor abstraction]" )
{
    size_t count = 10;


    std::vector<int> A = {10, 10, 10, 10, 10};
    std::vector<float> B = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<float> C(A.size(), 0);



    {

        DataTest2 data(A, B, C);
        sycl::queue q;

        q.submit([&](sycl::handler& h)
        {
            auto data_access = data.device_access;
            data_access.give_access(h);

            h.parallel_for(count, [=](sycl::id<1> idx)
            {
                size_t i = idx[0];
                data_access.C(i) = data_access.A(i) * data_access.B(i);

                size_t size = data_access.size;
            });
        });

        q.submit([&](sycl::handler& h)
        {
            auto data_access = data.device_access;
            data_access.give_access(h);

            h.parallel_for(count, [=](sycl::id<1> idx)
            {
                size_t i = idx[0];
                data_access.C(i) = 2 * data_access.C(i);
            });
        });
        q.wait();

        auto host_access = data.get_host_access();
        for(int i = 0; i < A.size(); i++)
        {
            CHECK(host_access.bufC_acc[i] == Approx(2.0 * host_access.bufA_acc[i] * host_access.bufB_acc[i]));
        }
    }

    for(int i = 0; i < A.size(); i++)
    {
        CHECK(C[i] == Approx(2.0 * A[i] * B[i]));
    }

}