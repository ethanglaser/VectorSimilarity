#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <numeric>
#include <algorithm>

#include "../Utils/ReadDatFile.h"
#include "../Normalizers/SelectedNormalizer.h"
#include "../Utils/VectorUtils.h"
template< typename vectType = float>
struct DataBaseErrorCheck
{
    std::vector<std::vector<vectType>> embeddings;
    DataBaseErrorCheck(const std::string& filename){
        readDatFile<vectType>(filename,embeddings);
    }

    template<typename Algorithm = Naive>
    std::map<std::string,vectType> getErrorStatistics()
    {
        std::vector<vectType> errors;
        for(auto& vect : embeddings)
        {
            auto naiveCpy = vect;
            auto algCpy = vect;
            Naive:: template Normalize<0>(naiveCpy.data(),naiveCpy.size());
            Algorithm::template Normalize<0>(algCpy.data(),algCpy.size());
            errors.push_back(calulcate_L2(naiveCpy,algCpy));
        }
        vectType sum = std::accumulate(errors.begin(), errors.end(), 0.0);
        vectType len = errors.size();
        vectType mean = sum / len;

        std::vector<vectType> diff(errors.size());
        std::transform(errors.begin(), errors.end(), diff.begin(),
                    std::bind2nd(std::minus<vectType>(), mean));
        vectType sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        vectType stdev = std::sqrt(sq_sum / errors.size());

        std::map<std::string,vectType> results;
        results["mean"] = mean;
        results["std"] = stdev;
        return results;
    }


    void print_vectors(size_t head = 5)
    {
        size_t counter = 1;
        std::cout<<"There are "<<embeddings.size()<<" vectors in dataset"<<std::endl;
        for(auto& vect : embeddings)
        {
            if(counter > head && head>0)
                break;
            print_vect(vect.data(),vect.size());
            counter+=1;
        }
    }


};