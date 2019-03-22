/*
 * Realistic classification example.
 */
#include <yannl.hpp>
#include "print_matrix.hpp"
#include <sys/time.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>

using namespace std;
using namespace yannl;
#define FEATURES 2
void generate_artificial_training_set(Matrix &data, Matrix &target, size_t set_size, size_t set_features);

int main() {
    /*
     * The goal of this network is to take in "FEATURES" numbers and determine if their
     * sum is positive or negative. Positive is represented by the matrix:
     * {1, 0}, while negative is represented by the matrix {0, 1}. The two
     * numbers should be in the domain: [-500, 500].
     */
    Matrix data, target;
    generate_artificial_training_set(data, target, 2000, FEATURES);

    // Set a unique seed for this training session.
    srand(time(NULL));

    /*
     * Create network with "FEATURES" input neurons, 4 hidden layers each with 8 neurons
     * and tanh activation function, and 2 output neurons with softmax
     * activation function.
     */
    Network network(CLASSIFICATION, FEATURES, {8, 8, 8, 8}, {tanh, tanh, tanh, tanh}, 2, softmax);

    // Convenience structure for easy parameter filling.
    TrainingParams params;
    params.data = data;
    params.target = target;
    params.batch_size = data.rows();
    params.max_iterations = 80;
    params.epoch_analysis_interval = 10;
    params.min_accuracy = 1;
    params.learning_rate = 0.1;
    params.annealing_factor = 50;
    params.regularization_factor = 0.0015;
    params.momentum_factor = 0.95;
    params.shuffle = true;
    //Timing data
    struct timeval end, start;
    // Train network. Request it to print out loss every epoch_analysis_interval.
    gettimeofday(&start, NULL);
    network.train(params, true);
    gettimeofday(&end, NULL);
    cout << "Learning Runtime in seconds: " << ((end.tv_sec - start.tv_sec) * 
    1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl << endl;	// calculate time elapsed
    printf("Accuracy: %.3f\n", network.accuracy(data, target));

    // Save and load it (for demonstration purposes if you want).
    network.save("my_network.txt");
    network = Network("my_network.txt");
    return 0;
}

void generate_artificial_training_set(Matrix &data, Matrix &target, size_t set_size, size_t set_features) {
    // Generate data first.
    data = Matrix(set_size, set_features);
    double rand_val;
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); j++){
            rand_val = (rand() % 1000) - (rand() % 1000);
            data[i][j] = rand_val;
        }
    }

    // Generate target next.
    target = Matrix(set_size, 2);
    for (size_t i = 0; i < target.rows(); ++i) {
        double sum = 0;
        for (size_t j = 0; j < data.cols(); ++j) {
            sum += data[i][j];
        }

        if (sum > 0) {
            target[i][0] = 1;
        } else {
            target[i][1] = 1;
        }
    }
}
