#include <vector>
#define START 1
#define END 1024
// #include "VectorNormalizer_AVX512FP64.h"
#include "../Normalizers/SelectedNormalizer.h"
#include <cassert>
#include <iostream>
#include "../Utils/VectorUtils.h"
template<unsigned char residual_start = 15,typename vectType = float,size_t min_dim=START,size_t max_dim = END, typename Algorithm = Naive>
class CorrectnessTester
{
    public:
    vectType min_value;
    vectType max_value;
    vectType value_step;
    std::vector<std::vector<vectType>> vectors;
    CorrectnessTester<residual_start-1,vectType,min_dim,max_dim> *next_residual;
    CorrectnessTester(vectType min_value = 0,vectType max_value = 1024,vectType value_step = 1): min_value(min_value),max_value(max_value),value_step(value_step), next_residual(NULL)
    {
        unsigned char dim_start = min_dim + (16-min_dim%16) +residual_start;
        for(; dim_start<=max_dim;dim_start+=16)
        {
            for(vectType _start = min_value; _start <= max_value;_start+=value_step )
            {
                vectors.emplace_back(dim_start,_start);
                vectors.emplace_back(dim_start,_start);
                auto& vector  = vectors.back();
                for(int i=0;i<vector.size();i+=2)
                    vector[i]/=2;
            }

        } 
    }
    CorrectnessTester<residual_start-1,vectType,min_dim,max_dim>* get_next_residual()
    {
        next_residual = new  CorrectnessTester<residual_start-1,vectType,min_dim,max_dim>(min_value,max_value,value_step); 
        return next_residual;
   }
    ~CorrectnessTester()
    {
        if(next_residual)
        delete next_residual;
    }
    static void test_vectors()
    {
        CorrectnessTester tv(1,10,1);
        // std::cout<<"Testing vector in residual: "<<int(residual_start)<<std::endl;
        for(auto& vector : tv.vectors)
        {
            auto vector_cpy(vector);
            Naive::Normalize<residual_start>(vector.data(),vector.size());
            Algorithm::template Normalize<residual_start>(vector_cpy.data(),vector_cpy.size());
            assert(Algorithm::compareVectors(vector,vector_cpy));
        }
        CorrectnessTester<residual_start-1,vectType,min_dim,max_dim>::test_vectors();
    }
};

template<typename vectType,size_t min_dim,size_t max_dim>
class CorrectnessTester<1,vectType,min_dim,max_dim>
{
    public:
    vectType min_value;
    vectType max_value;
    vectType value_step;
    std::vector<std::vector<vectType>> vectors;
    CorrectnessTester(vectType min_value = 0,vectType max_value = 1024,vectType value_step = 1): min_value(min_value),max_value(max_value),value_step(value_step)
    {
        for(size_t dim_start = min_dim; dim_start<=max_dim;dim_start+=16)
        {
            for(vectType _start = min_value; _start <= max_value;_start+=value_step )
            {
                vectors.emplace_back(dim_start,_start);
            }

        } 
    }
    void* get_next_residual()
    {
        return NULL;
    }
    static void test_vectors()
    {
        CorrectnessTester tv(1,10,1);
        for(auto& vector : tv.vectors)
        {
            auto vector_cpy(vector);
            Naive::Normalize<1>(vector.data(),vector.size());
            Naive::Normalize<1>(vector_cpy.data(),vector_cpy.size());
            assert(vector_cpy==vector);
        }
        return;
    }
    ~CorrectnessTester()
    {
    }
};