#pragma once
#include <cmath>

template <unsigned char residual, typename vect_type>
void Naive_Normalizer( vect_type* const v,size_t dim)
{
            vect_type sumOfPower = 0;
            for(size_t i = 0; i < dim ; i++)
            {
                sumOfPower+= (v[i]*v[i]);
            }
            sumOfPower = sqrt((double)sumOfPower);
            if(sumOfPower==0) 
                return;
            for(size_t i = 0; i < dim ; i++)
            {
                v[i]/=sumOfPower;
            }
        
}