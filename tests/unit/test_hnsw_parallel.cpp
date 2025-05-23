/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "unit_test_utils.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/vec_sim_debug.h"
#include <unistd.h>
#include <random>
#include <thread>
#include <atomic>

// Helper macro to get the closest even number which is equal or lower than x.
#define FLOOR_EVEN(x) ((x) - ((x) & 1))

template <typename index_type_t>
class HNSWTestParallel : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    VecSimIndex *index = nullptr;
    VecSimIndex *CreateNewIndex(HNSWParams &params, bool is_multi = false) {
        index = test_utils::CreateNewIndex(params, index_type_t::get_index_type(), is_multi);
        return index;
    }
    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) const {
        return static_cast<HNSWIndex<data_t, dist_t> *>(index);
    }
    HNSWIndex_Single<data_t, dist_t> *CastToHNSW_Single(VecSimIndex *index) const {
        return static_cast<HNSWIndex_Single<data_t, dist_t> *>(index);
    }

    void TearDown() override {
        if (index)
            VecSimIndex_Free(index);
    }

    /* Helper methods for testing repair jobs:
     * Collect all the nodes that require repair due to the deletions, from top level down, and
     * insert them into a queue.
     */
    void CollectRepairJobs(HNSWIndex<data_t, dist_t> *hnsw_index,
                           std::vector<pair<idType, size_t>> &jobQ) {
        size_t n = hnsw_index->indexSize();
        for (labelType element_id = 0; element_id < n; element_id++) {
            if (!hnsw_index->isMarkedDeleted(element_id)) {
                continue;
            }
            ElementGraphData *element_data = hnsw_index->getGraphDataByInternalId(element_id);
            for (size_t level = 0; level <= element_data->toplevel; level++) {
                ElementLevelData &cur_level_data =
                    hnsw_index->getElementLevelData(element_data, level);

                // Go over the neighbours of the element in a specific level.
                for (size_t i = 0; i < cur_level_data.numLinks; i++) {
                    idType cur_neighbor = cur_level_data.links[i];
                    ElementLevelData &neighbor_level_data =
                        hnsw_index->getElementLevelData(cur_neighbor, level);
                    for (size_t j = 0; j < neighbor_level_data.numLinks; j++) {
                        // If the edge is bidirectional, do repair for this neighbor
                        if (neighbor_level_data.links[j] == element_id) {
                            jobQ.emplace_back(cur_neighbor, level);
                            break;
                        }
                    }
                }
                // Next, go over the rest of incoming edges (the ones that are not bidirectional)
                // and make repairs.
                for (auto incoming_edge : *cur_level_data.incomingUnidirectionalEdges) {
                    jobQ.emplace_back(incoming_edge, level);
                }
            }
        }
    }

    /**
     * Serialize the connections of the index for debugging purposes.
     * Example output:
     *  index connections: {
     *      Entry Point Label: 3
     *
     *      Node 0:
     *          Level 0 neighbors:
     *              1, 2, 3, 4, 5,
     *      Node 1:
     *          Level 0 neighbors:
     *              0, 2, 3,
     *      Node 2:
     *          Level 0 neighbors:
     *              0, 1, 3,
     *      Node 3:
     *          Level 0 neighbors:
     *              0, 1, 2, 4, 5,
     *          Level 1 neighbors:
     *              4,
     *      Node 4:
     *          Level 0 neighbors:
     *              0, 3, 5,
     *          Level 1 neighbors:
     *              3,
     *      Node 5:
     *          Level 0 neighbors:
     *              0, 3, 4,
     *  }
     */
    std::string serializeIndexConnections(VecSimIndex *index) const {
        std::string res("index connections: {");
        auto *hnsw_index = CastToHNSW(index);

        if (index->indexSize() > 0) {
            res += "\nEntry Point Label: ";
            res += std::to_string(hnsw_index->getEntryPointLabel()) + "\n";
        }
        for (idType id = 0; id < index->indexSize(); id++) {
            labelType label = hnsw_index->getExternalLabel(id);
            if (label == SIZE_MAX)
                continue; // The ID is not in the index
            int **neighbors_output;
            VecSimDebug_GetElementNeighborsInHNSWGraph(index, label, &neighbors_output);
            res += "\nNode " + std::to_string(label) + ":";
            for (size_t l = 0; neighbors_output[l]; l++) {
                res += "\n\tLevel " + std::to_string(l) + " neighbors:\n\t\t";
                auto &neighbours = neighbors_output[l];
                auto neighbours_count = neighbours[0];
                for (size_t j = 1; j <= neighbours_count; j++) {
                    res += std::to_string(neighbours[j]) + ", ";
                }
            }
            VecSimDebug_ReleaseElementNeighborsInHNSWGraph(neighbors_output);
        }
        return res + "}";
    }

    void insertVectorParallelSafe(VecSimIndex *parallel_index, size_t dim, labelType label,
                                  data_t val, std::shared_mutex &indexGuard,
                                  std::atomic<size_t> &counter, std::mutex &barrier,
                                  size_t block_size = DEFAULT_BLOCK_SIZE) {
        // The decision as to when to allocate a new block is made by the index internally in the
        // "addVector" function, where there is an internal counter that is incremented for each
        // vector. To ensure that the thread which is taking the write lock is the one that performs
        // the resizing, we make sure that no other thread is allowed to bypass the thread for which
        // the global counter is a multiple of the block size. Hence, we use the barrier lock and
        // lock in every iteration to ensure we acquire the right lock (read/write) based on the
        // global counter, so threads won't call "addVector" with the inappropriate lock.
        bool exclusive = true;
        barrier.lock();
        if (counter++ % block_size != 0) {
            indexGuard.lock_shared();
            exclusive = false;
        } else {
            // Lock exclusively if we are performing resizing due to a new block.
            indexGuard.lock();
        }
        barrier.unlock();
        GenerateAndAddVector<data_t>(parallel_index, dim, label, val);
        exclusive ? indexGuard.unlock() : indexGuard.unlock_shared();
    }

    void parallelInsertSearch(bool is_multi);
};

// DataTypeSet, TEST_DATA_T and TEST_DIST_T are defined in unit_test_utils.h

TYPED_TEST_SUITE(HNSWTestParallel, DataTypeSet);

TYPED_TEST(HNSWTestParallel, parallelSearchKnn) {
    size_t n = 20000;
    size_t k = 11;
    size_t dim = 45;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .M = 64, .efConstruction = 200, .efRuntime = n};
    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    size_t n_threads = std::min(8U, std::thread::hardware_concurrency());
    std::atomic_int successful_searches(0);
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Run parallel searches where every searching thread expects to get different labels as results
    // (determined by the thread id), which are labels in the range [50+myID-5, 50+myID+5].
    auto parallel_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the res
            // label and the query val (query_val, query_val-1, query_val+1, query_val-2,
            // query_val+2, ...) The score is the L2 distance between the vectors that correspond
            // the ids.
            int sign = (res_index % 2 == 0) ? 1 : -1;
            size_t expected_id = query_val + (sign * int((res_index + 1) / 2));
            double expected_score = dim * ((res_index + 1) / 2) * ((res_index + 1) / 2);
            ASSERT_EQ(id, expected_id);
            ASSERT_DOUBLE_EQ(score, expected_score);
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    size_t memory_before = index->statisticInfo().memory;
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_search, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_FALSE(testing::Test::HasFatalFailure()) << this->serializeIndexConnections(index);
    ASSERT_EQ(successful_searches, n_threads);

    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), 1);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()), 1);
    // Make sure that we properly update the allocator atomically during the searches. The expected
    // Memory delta should only be the visited nodes handler added to the pool.
    size_t max_elements = this->CastToHNSW(index)->maxElements;
    size_t expected_memory =
        memory_before + (index->debugInfo().hnswInfo.visitedNodesPoolSize - 1) *
                            (sizeof(VisitedNodesHandler) + sizeof(tag_t) * max_elements +
                             2 * sizeof(size_t) + sizeof(void *));
    ASSERT_EQ(expected_memory, index->statisticInfo().memory);
}

TYPED_TEST(HNSWTestParallel, parallelSearchKNNMulti) {
    size_t dim = 45;
    size_t n = 20000;
    size_t n_labels = 1000;
    size_t k = 11;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .M = 64, .efRuntime = n};
    VecSimIndex *index = this->CreateNewIndex(params, true);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i % n_labels, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    size_t n_threads = std::min(8U, std::thread::hardware_concurrency());
    std::atomic_int successful_searches(0);
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Run parallel searches where every searching thread expects to get different label as results
    // (determined by the thread id), which are labels in the range [50+myID-5, 50+myID+5].
    auto parallel_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [=](size_t id, double score, size_t res_index) {
            int sign = (res_index % 2 == 0) ? 1 : -1;
            size_t expected_id = query_val + (sign * int((res_index + 1) / 2));
            double expected_score = dim * ((res_index + 1) / 2) * ((res_index + 1) / 2);
            ASSERT_EQ(id, expected_id);
            ASSERT_DOUBLE_EQ(score, expected_score);
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_search, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_FALSE(testing::Test::HasFatalFailure()) << this->serializeIndexConnections(index);
    ASSERT_EQ(successful_searches, n_threads);
    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), 1);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()), 1);
}

TYPED_TEST(HNSWTestParallel, parallelSearchCombined) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 64;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .M = 64, .efConstruction = 200, .efRuntime = n};
    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    size_t n_threads = std::min(15U, std::thread::hardware_concurrency());
    std::atomic_int successful_searches(0);
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    /* Run parallel searches of three kinds: KNN, range, and batched search. */

    // In knn, we expect to get different labels as results (determined by the thread id), which are
    // labels in the range [50+myID-5, 50+myID+5].
    auto parallel_knn_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the res
            // label and the query val (query_val, query_val-1, query_val+1, query_val-2,
            // query_val+2, ...) The score is the L2 distance between the vectors that correspond
            // the ids.
            int sign = (res_index % 2 == 0) ? 1 : -1;
            size_t expected_id = query_val + (sign * int((res_index + 1) / 2));
            double expected_score = dim * ((res_index + 1) / 2) * ((res_index + 1) / 2);
            ASSERT_EQ(id, expected_id);
            ASSERT_DOUBLE_EQ(score, expected_score);
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    auto parallel_range_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T pivot_id = 100.01 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, pivot_id);
        auto verify_res_by_score = [&](size_t id, double score, size_t res_index) {
            int sign = (res_index % 2 == 0) ? -1 : 1;
            size_t expected_id = pivot_id + (sign * int((res_index + 1) / 2));
            double factor = ((res_index + 1) / 2) - sign * 0.01;
            double expected_score = dim * factor * factor;
            ASSERT_EQ(id, expected_id);
            ASSERT_NEAR(score, expected_score, 0.01);
        };
        uint expected_num_results = 11;
        // To get 11 results in the range [pivot_id-5, pivot_id+5], set the radius as the L2 score
        // in the boundaries.
        double radius = dim * expected_num_results * expected_num_results / 4.0;
        runRangeQueryTest(index, query, radius, verify_res_by_score, expected_num_results,
                          BY_SCORE);
        successful_searches++;
    };

    auto parallel_batched_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet
        // in every iteration. The results order should be sorted by their score (distance from the
        // query vector), which means sorted from the largest id to the lowest.
        // Run different number of iterations for every thread id.
        size_t total_iterations = myID;
        size_t n_res = 5;
        while (VecSimBatchIterator_HasNext(batchIterator) && iteration_num < total_iterations) {
            std::vector<size_t> expected_ids(n_res);
            for (size_t i = 0; i < n_res; i++) {
                expected_ids[i] = (n - iteration_num * n_res - i - 1);
            }
            auto verify_res = [&](size_t id, double score, size_t res_index) {
                ASSERT_EQ(expected_ids[res_index], id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, total_iterations);
        VecSimBatchIterator_Free(batchIterator);
        successful_searches++;
    };

    std::thread thread_objs[n_threads];
    size_t memory_before = index->statisticInfo().memory;
    for (size_t i = 0; i < n_threads; i++) {
        if (i % 3 == 0) {
            thread_objs[i] = std::thread(parallel_knn_search, i);
        } else if (i % 3 == 1) {
            thread_objs[i] = std::thread(parallel_range_search, i);
        } else {
            thread_objs[i] = std::thread(parallel_batched_search, i);
        }
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_FALSE(testing::Test::HasFatalFailure()) << this->serializeIndexConnections(index);

    ASSERT_EQ(successful_searches, n_threads);
    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), 1);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()), 1);

    // Make sure that we properly update the allocator atomically during the searches.
    // Memory delta should only be the visited nodes handler added to the pool.
    size_t max_elements = this->CastToHNSW(index)->maxElements;
    size_t expected_memory =
        memory_before + (index->debugInfo().hnswInfo.visitedNodesPoolSize - 1) *
                            (sizeof(VisitedNodesHandler) + sizeof(tag_t) * max_elements +
                             2 * sizeof(size_t) + sizeof(void *));
    ASSERT_EQ(expected_memory, index->statisticInfo().memory);
}

TYPED_TEST(HNSWTestParallel, parallelInsert) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 32;
    // r/w lock to ensure that index is locked (stop the world) upon adding a new block to the
    // global data structures, which is non read safe for parallel insertions.
    std::shared_mutex indexGuard;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .M = 16, .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);
    size_t n_threads = 10;

    // Save the number of tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);
    std::atomic<size_t> counter{0};
    std::mutex barrier;

    auto parallel_insert = [&](int myID) {
        for (labelType label = myID; label < n; label += n_threads) {
            completed_tasks[myID]++;
            // Insert vector while acquire the guard lock exclusively if we are performing resizing.
            this->insertVectorParallelSafe(index, dim, label, label, indexGuard, counter, barrier);
        }
    };
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_insert, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    // Validate that every thread executed n/n_threads jobs.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), n / n_threads);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()),
              ceil((double)n / n_threads));

    TEST_DATA_T query[dim];
    TEST_DATA_T query_val = (TEST_DATA_T)n / 2;
    GenerateVector<TEST_DATA_T>(query, dim, query_val);
    auto verify_res = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the res
        // label and the query val (n/2, n/2-1, n/2+1, n/2-2, n/2+2, ...) The score is the L2
        // distance between the vectors that correspond the ids.
        int sign = (res_index % 2 == 0) ? 1 : -1;
        size_t expected_id = query_val + (sign * int((res_index + 1) / 2));
        double expected_score = dim * ((res_index + 1) / 2) * ((res_index + 1) / 2);
        ASSERT_EQ(id, expected_id);
        ASSERT_DOUBLE_EQ(score, expected_score);
    };
    runTopKSearchTest(index, query, k, verify_res);
    ASSERT_FALSE(testing::Test::HasFatalFailure()) << this->serializeIndexConnections(index);
}

TYPED_TEST(HNSWTestParallel, parallelInsertMulti) {
    size_t n = 10000;
    size_t n_labels = 1000;
    size_t per_label = n / n_labels;
    size_t k = 11;
    size_t dim = 32;

    // r/w lock to ensure that index is locked (stop the world) upon adding a new block to the
    // global data structures, which is non read safe for parallel insertions.
    std::shared_mutex indexGuard;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .M = 16, .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params, true);
    size_t n_threads = 10;

    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);
    std::atomic<size_t> counter{0};
    std::mutex barrier;

    auto parallel_insert = [&](int myID) {
        for (size_t i = myID; i < n; i += n_threads) {
            completed_tasks[myID]++;
            // Insert vector while acquire the guard lock exclusively if we are performing resizing.
            this->insertVectorParallelSafe(index, dim, i % n_labels, i, indexGuard, counter,
                                           barrier);
        }
    };
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_insert, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    // Validate that every thread executed n/n_threads jobs.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), n / n_threads);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()),
              ceil((double)n / n_threads));

    TEST_DATA_T query[dim];
    size_t query_val = n / 2 + 10;
    size_t pivot_val = (size_t)query_val % n_labels;
    GenerateVector<TEST_DATA_T>(query, dim, (TEST_DATA_T)query_val);
    auto verify_res = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the res
        // label and query_val%n_labels (that is ids 10, 9, 11, ... for the current arguments).
        // The score is the L2 distance between the vectors that correspond the ids.
        int sign = (res_index % 2 == 0) ? 1 : -1;
        size_t expected_id = pivot_val + (sign * int((res_index + 1) / 2));
        double expected_score = dim * ((res_index + 1) / 2) * ((res_index + 1) / 2);
        ASSERT_EQ(id, expected_id);
        ASSERT_DOUBLE_EQ(score, expected_score);
    };
    runTopKSearchTest(index, query, k, verify_res);
    ASSERT_FALSE(testing::Test::HasFatalFailure()) << this->serializeIndexConnections(index);
}

template <class index_type_t>
void HNSWTestParallel<index_type_t>::parallelInsertSearch(bool is_multi) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 32;
    // r/w lock to ensure that index is locked (stop the world) upon adding a new block to the
    // global data structures, which is non read safe for parallel insertions.
    std::shared_mutex indexGuard;
    data_t query_val = (data_t)n / 4;
    labelType first_res_label = query_val - k / 2;
    labelType last_res_label = query_val + k / 2;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .M = 64, .efConstruction = 200, .efRuntime = n};

    VecSimIndex *parallel_index = this->CreateNewIndex(params, is_multi);

    std::atomic<size_t> indexed_vectors(0);
    // Insert the vectors we expect to search for.
    for (labelType res_label = first_res_label; res_label <= last_res_label; res_label++) {
        GenerateAndAddVector<data_t>(parallel_index, dim, res_label, res_label);
        indexed_vectors++;
    }
    size_t n_threads = std::min(10U, FLOOR_EVEN(std::thread::hardware_concurrency()));
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);
    std::mutex barrier;

    auto parallel_insert = [&](int myID) {
        for (labelType label = myID; label < n; label += n_threads / 2) {
            completed_tasks[myID]++;
            if (label >= first_res_label && label <= last_res_label) {
                continue; // Skip the vectors we already indexed.
            }
            // Insert vector while acquire the guard lock exclusively if we are performing resizing.
            this->insertVectorParallelSafe(parallel_index, dim, label, label, indexGuard,
                                           indexed_vectors, barrier);
        }
    };
    std::atomic_int successful_searches(0);
    size_t batch_size = n / 20;
    auto parallel_knn_search = [&](int myID) {
        size_t local_search_count = 0;
        while (indexed_vectors < 0.95 * n) {
            if (indexed_vectors < (local_search_count * batch_size)) {
                usleep(100); // Wait for another batch of vectors to be indexed.
                continue;
            }
            completed_tasks[myID]++;
            data_t query[dim];
            GenerateVector<data_t>(query, dim, query_val);
            auto verify_res = [&](size_t id, double score, size_t res_index) {
                // We expect to get the results with increasing order of the distance between the
                // res label and the query val (n/4, n/4-1, n/4+1, n/4-2, n/4+2, ...) The score is
                // the L2 distance between the vectors that correspond the ids.
                int sign = (res_index % 2 == 0) ? 1 : -1;
                size_t expected_id = query_val + (sign * int((res_index + 1) / 2));
                double expected_score = dim * ((res_index + 1) / 2) * ((res_index + 1) / 2);
                ASSERT_EQ(id, expected_id);
                ASSERT_DOUBLE_EQ(score, expected_score);
            };
            local_search_count++;
            indexGuard.lock_shared();
            runTopKSearchTest(parallel_index, query, k, verify_res);
            indexGuard.unlock_shared();
            successful_searches++;
        }
    };

    auto hnsw_index = this->CastToHNSW(parallel_index);
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        if (i < n_threads / 2) {
            thread_objs[i] = std::thread(parallel_insert, i);
        } else {
            thread_objs[i] = std::thread(parallel_knn_search, i);
        }
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }

    ASSERT_FALSE(testing::Test::HasFatalFailure()) << this->serializeIndexConnections(hnsw_index);
    ASSERT_EQ(VecSimIndex_IndexSize(parallel_index), n);
    // Validate that every insertion thread executed n/(n_threads/2) jobs.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
              n / (n_threads / 2));
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
              ceil((double)n / (n_threads / 2)));
}
TYPED_TEST(HNSWTestParallel, parallelInsertSearchSingle) { this->parallelInsertSearch(false); }
TYPED_TEST(HNSWTestParallel, parallelInsertSearchMulti) { this->parallelInsertSearch(true); }

TYPED_TEST(HNSWTestParallel, parallelRepairs) {
    size_t n = 1000;
    size_t dim = 32;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2};

    auto *hnsw_index = this->CastToHNSW(this->CreateNewIndex(params));
    size_t n_threads = std::min(10U, std::thread::hardware_concurrency());
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Create some random vectors and insert them to the index.
    std::srand(10); // create pseudo random generator with ana arbitrary seed.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        VecSimIndex_AddVector(hnsw_index, vector, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(hnsw_index), n);

    // Queue of repair jobs, each job is represented as {id, level}
    auto jobQ = std::vector<pair<idType, size_t>>();

    // Collect all the nodes that require repairment due to the deletions, from top level down.
    for (size_t element_id = 0; element_id < n; element_id += 2) {
        hnsw_index->markDelete(element_id);
    }
    ASSERT_EQ(hnsw_index->getNumMarkedDeleted(), n / 2);
    // Every that every deleted node should have at least 2 connections to repair.
    auto report = hnsw_index->checkIntegrity();
    ASSERT_GE(report.connections_to_repair, n);

    this->CollectRepairJobs(hnsw_index, jobQ);
    size_t n_jobs = jobQ.size();
    ASSERT_EQ(report.connections_to_repair, n_jobs);

    auto executeRepairJobs = [&](int myID) {
        for (size_t i = myID; i < n_jobs; i += n_threads) {
            auto job = jobQ[i];
            hnsw_index->repairNodeConnections(job.first, job.second); // {element_id, level}
            completed_tasks[myID]++;
        }
    };

    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(executeRepairJobs, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    // Check index integrity, also make sure that no node is pointing to a deleted node.
    report = hnsw_index->checkIntegrity();
    ASSERT_TRUE(report.valid_state);
    ASSERT_EQ(report.connections_to_repair, 0);

    // Validate that the tasks are spread among the threads uniformly.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()),
              floorf((float)n_jobs / n_threads));
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()),
              ceilf((float)n_jobs / n_threads));
}

TYPED_TEST(HNSWTestParallel, parallelRepairSearch) {
    size_t n = 10000;
    size_t k = 10;
    size_t dim = 32;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .efRuntime = n};

    auto *hnsw_index = this->CastToHNSW(this->CreateNewIndex(params));
    size_t n_threads = std::min(10U, FLOOR_EVEN(std::thread::hardware_concurrency()));
    // Save the number of tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(hnsw_index), n);

    // Queue of repair jobs, each job is represented as {id, level}
    auto jobQ = std::vector<pair<idType, size_t>>();

    for (size_t element_id = 0; element_id < n; element_id += 2) {
        hnsw_index->markDelete(element_id);
    }
    ASSERT_EQ(hnsw_index->getNumMarkedDeleted(), n / 2);
    // Every deleted node i should have at least 2 connection to repair (to i-1 and i+1), except for
    // 0 and n-1 that has at least one connection to repair.
    ASSERT_GE(hnsw_index->checkIntegrity().connections_to_repair, n - 2);

    // Collect all the nodes that require repairment due to the deletions, from top level down.
    this->CollectRepairJobs(hnsw_index, jobQ);
    size_t n_jobs = jobQ.size();

    auto executeRepairJobs = [&](int myID) {
        for (size_t i = myID; i < n_jobs; i += n_threads / 2) {
            auto job = jobQ[i];
            hnsw_index->repairNodeConnections(job.first, job.second); // {element_id, level}
            completed_tasks[myID]++;
        }
    };

    bool run_queries = true;
    auto parallel_knn_search = [&](int myID) {
        TEST_DATA_T query_val = (TEST_DATA_T)n / 4 + 2 * myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the
            // res label and the query val and only odd labels (query_val-1, query_val+1,
            // query_val-3, query_val+3, ...) The score is the L2 distance between the vectors that
            // correspond the ids.
            int sign = (res_index % 2 == 0) ? -1 : 1;
            int next_odd = res_index | 1;
            size_t expected_id = query_val + (sign * next_odd);
            double expected_score = dim * next_odd * next_odd;
            ASSERT_EQ(id, expected_id);
            ASSERT_DOUBLE_EQ(score, expected_score);
        };
        do {
            runTopKSearchTest(hnsw_index, query, k, verify_res);
            completed_tasks[myID]++;
        } while (run_queries);
    };

    std::thread thread_objs[n_threads];
    // Run queries, expect to get only non-deleted vector as results.
    for (size_t i = n_threads / 2; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_knn_search, i);
    }

    // Run the repair jobs.
    for (size_t i = 0; i < n_threads / 2; i++) {
        thread_objs[i] = std::thread(executeRepairJobs, i);
    }
    for (size_t i = 0; i < n_threads / 2; i++) {
        thread_objs[i].join();
    }
    // Once all the repair jobs are done, signal the query threads to finish.
    run_queries = false;
    for (size_t i = n_threads / 2; i < n_threads; i++) {
        thread_objs[i].join();
    }

    ASSERT_FALSE(testing::Test::HasFatalFailure()) << this->serializeIndexConnections(hnsw_index);
    // Check index integrity, also make sure that no node is pointing to a deleted node.
    auto report = hnsw_index->checkIntegrity();
    ASSERT_TRUE(report.valid_state);
    ASSERT_EQ(report.connections_to_repair, 0);

    // Validate that every search thread ran at least one job.
    ASSERT_GE(*std::min_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()), 1);
    // Validate that the repair tasks are spread among the threads uniformly.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
              floorf((float)n_jobs / (n_threads / 2.0)));
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
              ceilf((float)n_jobs / (n_threads / 2.0)));
}

TYPED_TEST(HNSWTestParallel, parallelRepairInsert) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 4;
    size_t block_size = 10;

    // r/w lock to ensure that index is locked (stop the world) upon adding a new block to the
    // global data structures, which is non read safe for parallel insertions.
    std::shared_mutex indexGuard;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .blockSize = block_size, .efRuntime = n};

    auto *hnsw_index = this->CastToHNSW(this->CreateNewIndex(params));
    size_t n_threads = std::min(8U, FLOOR_EVEN(std::thread::hardware_concurrency()));
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Insert n/2 vectors to the index.
    for (size_t i = 0; i < n / 2; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(hnsw_index), n / 2);

    // Queue of repair jobs, each job is represented as {id, level}
    auto jobQ = std::vector<pair<idType, size_t>>();
    for (size_t element_id = 0; element_id < n / 2; element_id += 2) {
        hnsw_index->markDelete(element_id);
    }
    ASSERT_EQ(hnsw_index->getNumMarkedDeleted(), n / 4);
    // Every deleted node i should have at least 2 connection to repair (to i-1 and i-1), except for
    // 0 that has at least one connection to repair.
    ASSERT_GE(hnsw_index->checkIntegrity().connections_to_repair, n / 2 - 1);

    // Collect all the nodes that require repairment due to the deletions, from top level down.
    this->CollectRepairJobs(hnsw_index, jobQ);
    size_t n_jobs = jobQ.size();

    auto executeRepairJobs = [&](int myID) {
        for (size_t i = myID - n_threads / 2; i < n_jobs; i += n_threads / 2) {
            auto job = jobQ[i];
            indexGuard.lock_shared();
            hnsw_index->repairNodeConnections(job.first, job.second); // {element_id, level}
            indexGuard.unlock_shared();
            completed_tasks[myID]++;
        }
    };

    std::atomic<size_t> counter{hnsw_index->indexSize()};
    std::mutex barrier;

    auto parallel_insert = [&](int myID) {
        // Reinsert the even ids that were deleted, and n/4 more even ids.
        for (labelType label = 2 * myID; label < n; label += n_threads) {
            completed_tasks[myID]++;
            // Insert vector while acquire the guard lock exclusively if we are performing resizing.
            this->insertVectorParallelSafe(hnsw_index, dim, label, label, indexGuard, counter,
                                           barrier, block_size);
        }
    };

    std::thread thread_objs[n_threads];

    // Insert n/2 new vectors while we repair connections.
    for (size_t i = 0; i < n_threads / 2; i++) {
        thread_objs[i] = std::thread(parallel_insert, i);
    }
    for (size_t i = n_threads / 2; i < n_threads; i++) {
        thread_objs[i] = std::thread(executeRepairJobs, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    // Check index integrity, also make sure that no node is pointing to a deleted node.
    ASSERT_EQ(hnsw_index->indexSize(), n);
    auto report = hnsw_index->checkIntegrity();
    ASSERT_TRUE(report.valid_state);
    ASSERT_EQ(report.connections_to_repair, 0);

    // Validate that the repair tasks are spread among the threads uniformly.
    ASSERT_EQ(*std::min_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()),
              floorf((float)n_jobs / (n_threads / 2.0)));
    ASSERT_EQ(*std::max_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()),
              ceilf((float)n_jobs / (n_threads / 2.0)));

    // Run queries to validate the index new state.
    TEST_DATA_T query[dim];
    // Around 3n/4 we only have even numbers vectors.
    size_t query_val = 3 * n / 4;
    GenerateVector<TEST_DATA_T>(query, dim, query_val);
    auto verify_res_even = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the
        // res label and the query val (3n/4, 3n/4 - 2, 3n/4 + 2, 3n/4 - 4 3n/4 + 4, ...) The score
        // is the L2 distance between the vectors that correspond the ids.
        int sign = (res_index % 2 == 0) ? 1 : -1;
        int next_even = 2 * ((res_index + 1) / 2); // (res_index % 2 ? res_index+1 : res_index;
        size_t expected_id = query_val + (sign * next_even);
        double expected_score = dim * next_even * next_even;
        ASSERT_EQ(id, expected_id);
        ASSERT_DOUBLE_EQ(score, expected_score);
    };
    runTopKSearchTest(hnsw_index, query, k, verify_res_even);
    ASSERT_FALSE(testing::Test::HasFatalFailure()) << this->serializeIndexConnections(hnsw_index);

    // Around n/4 we should have all vectors (even and odd).
    query_val = n / 4;
    GenerateVector<TEST_DATA_T>(query, dim, query_val);
    auto verify_res = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the
        // res label and the query val (n/4, n/4 - 1, n/4 + 1, n/4 - 2 n/4 + 2, ...) The score
        // is the L2 distance between the vectors that correspond the ids.
        int sign = (res_index % 2 == 0) ? 1 : -1;
        size_t expected_id = query_val + sign * int((res_index + 1) / 2);
        double expected_score = dim * ((res_index + 1) / 2) * ((res_index + 1) / 2);
        ASSERT_EQ(id, expected_id);
        ASSERT_DOUBLE_EQ(score, expected_score);
    };
    runTopKSearchTest(hnsw_index, query, k, verify_res);
    ASSERT_FALSE(testing::Test::HasFatalFailure()) << this->serializeIndexConnections(hnsw_index);
}
