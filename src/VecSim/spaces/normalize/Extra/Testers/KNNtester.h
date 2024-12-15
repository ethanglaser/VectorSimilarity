#include <vector>
#include <algorithm>
#include <numeric>

#include "../Normalizers/SelectedNormalizer.h"
#include "../Utils/ReadDatFile.h"
#include "../Utils/VectorUtils.h"

template< typename vectType>
struct KNNtester
{

    std::vector<std::vector<vectType>> embeddings;
    KNNtester(const std::string& filename){
        
        std::string extention(filename.begin()+filename.find('.'),filename.end());
        if (extention== std::string(".dat"))
        {    
            readDatFile<vectType>(filename,embeddings);
            return;
        }
        if (extention == std::string(".raw"))
        {
            readRawFile<vectType>(filename,embeddings);
        }
            
    }

    template <typename Algorithm = Naive>
    std::map<std::string,float> getRecallStatistics(const std::vector<std::vector<vectType>>& test_set,size_t k =10)
    {
        size_t tn=0,fp=0,tp=0,fn=0;
        int counter=0;
        for(const auto &queryVec : test_set)
        {
            if(counter%100==0)
            {
                printf("%d\n",counter);
            }
            counter++;
            std::vector<vectType> naiveQueryCpy = queryVec;
            std::vector<vectType> algQueryCpy = queryVec;
            Naive:: template Normalize<0>(naiveQueryCpy.data(),naiveQueryCpy.size());
            Algorithm::template Normalize<0>(algQueryCpy.data(),algQueryCpy.size());
            std::vector<vectType> naiveDistances;
            std::vector<vectType> algDistances;

            for(auto& embVect : embeddings)
            {
                std::vector<vectType> naiveEmbCpy = embVect;
                std::vector<vectType> algEmbCpy = embVect;
                Naive:: template Normalize<0>(naiveEmbCpy.data(),naiveEmbCpy.size());
                Algorithm::template Normalize<0>(algEmbCpy.data(),algEmbCpy.size());
                vectType naiveCosDist = dot_product(naiveEmbCpy,naiveQueryCpy);
                vectType algCosDist = dot_product(algEmbCpy,algQueryCpy);

                naiveDistances.push_back(naiveCosDist);
                algDistances.push_back(algCosDist);
            }
            auto naiveK = argpartition<vectType>(naiveDistances,k);
            auto algK = argpartition<vectType>(algDistances,k);

            // Sort the vectors (required by set_intersection)
            std::sort(naiveK.begin(), naiveK.end());
            std::sort(algK.begin(), algK.end());

            // Vector to store the intersection
            std::vector<size_t> intersection_result;

            // Compute intersection
            std::set_intersection(naiveK.begin(), naiveK.end(), algK.begin(), algK.end(), 
                                std::back_inserter(intersection_result));
            tp+= intersection_result.size();
            fp+= k - intersection_result.size();
            tn+= embeddings.size() - (2*k - intersection_result.size());
            fn+= (k - intersection_result.size());
        }
        std::map<std::string,float> results;
        results["tp"] = tp;     
        results["fp"] = fp;     
        results["tn"] = tn;     
        results["fn"] = fn;  
        results["recall"] = ((float)tp)/((float)fn+(float)tp);
        results["precision"] = ((float)tp)/((float)fp+(float)tp); 
        return results;
    }
    void print_vectors(size_t head = 5)
    {
        size_t counter = 1;
        for(auto& vect : embeddings)
        {
            if(counter > head && head>0)
                break;
            print_vect(vect.data(),vect.size());
            counter+=1;
        }
        std::cout<<"There are "<<embeddings.size()<<" vectors in dataset"<<std::endl;
        std::cout<<"Each size of  "<<embeddings[0].size()<<" dims"<<std::endl;
    }

    std::vector<std::vector<vectType>> splitTestTrain(float train = 0.7)
    {
        std::vector<std::vector<vectType>> test_sample(embeddings.begin()+embeddings.size()*0.7,embeddings.end());
        embeddings = std::vector<std::vector<vectType>> (embeddings.begin(),embeddings.begin()+embeddings.size()*0.7);
        return test_sample;
    }

};