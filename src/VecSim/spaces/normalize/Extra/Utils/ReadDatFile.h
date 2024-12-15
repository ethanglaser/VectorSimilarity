
#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

template<typename vectType = float>
void readDatFile(const std::string& filename,std::vector<std::vector<vectType>> &data) {
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::vector<vectType> row;
        std::istringstream iss(line);
        vectType value;

        while (iss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    infile.close();
}


template<typename vectType = float>
void readRawFile(const std::string& filename,std::vector<std::vector<vectType>> &data) {
    std::ifstream input(filename, std::ios::binary);
    
    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);
    size_t dim = 768;
    for (size_t i = 0; i < 10000; i++) {
        std::vector<vectType> query(dim);
        input.read((char *)query.data(), dim * sizeof(vectType));
        data.push_back(query);
    }

}


/*

    std::ifstream input(test_file, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);

    InsertToQueries(input);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::InsertToQueries(std::ifstream &input) {
    for (size_t i = 0; i < n_queries; i++) {
        std::vector<data_t> query(dim);
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}


*/