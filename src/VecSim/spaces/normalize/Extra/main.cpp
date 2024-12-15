#include <iostream>
#include <vector>
#include <cmath>

// #include "Normalizers/VectorNormalizer_AVXFP32.h"
// #include "VectorNormalizer_AVXFP64.h"
// #include "VectorNormalizer_AVX512FP64.h"
#include "Testers/CorrectnessTester.h"
#include "Testers/DataBaseErrorCheck.h"
#include "Testers/KNNtester.h"
#include "Utils/ReadDatFile.h"


int main(int argc, char* argv[])
{
    // DataBaseErrorCheck<float> dbec("data_set_train.dat");
    KNNtester<float> knnTester("dpedia.raw");
    std::vector<std::vector<float>> test_set = knnTester.splitTestTrain();
    test_set = std::vector<std::vector<float>>(test_set.begin(),test_set.begin()+10);
    // knnTester.print_vectors();
    auto results  = knnTester.getRecallStatistics<SingleSqrt>(test_set,5);
    // readDatFile("data_set_train.dat",test_set);
    std::cout<<"recall: "<<results["recall"]<<", precision: "<<results["precision"]<<std::endl;
    std::cout<<"tp: "<<results["tp"]<<", fp: "<<results["fp"]<<std::endl;

    // auto results = dbec.getErrorStatistics<SingleRsqrt>();
    // std::cout<<"mean error: "<<results["mean"]<<std::endl;
    // std::cout<<"error std: "<<results["std"]<<std::endl;
    // float ones[] = {1,1,1,1}; 
    // CorrectnessTester<7,float,32,128,SingleSqrt> ::test_vectors();
    // CorrectnessTester<15,float,16,128,SingleRsqrt> ::test_vectors();
    // CorrectnessTester<15,double,16,128,SingleScalar> ::test_vectors();

}


