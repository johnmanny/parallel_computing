#include "src/Yannl.hpp"
#include "examples/print_matrix.hpp"
#include <iostream>
#include <string.h>
#include <math.h>
#include <fstream>
#include <stdlib.h>	// for atoi, strtol
#include <time.h>	// for time calculations in determing days between
#include <iomanip>
#include <sys/time.h>	// for timing data
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

#define EXAMPLECOUNT 44628
#define TRAININGSET 40000
#define TESTSET EXAMPLECOUNT - TRAININGSET


using namespace std;
using namespace yannl;
//----------------------------------------------------------------//
/* function prototypes */
//----------------------------------------------------------------//

void initExamples(Matrix& exampleSet, Matrix& outputsSet,  Matrix& testSet, Matrix& targetSet);         // initializes examples array 
//void initNN(NeuralNet *);			// initializes neural net
//void deAlloc(NeuralNet *);
//void backPropLearning(example *, neuralNet *);	// performs neural net training, returns on 'good' training
//void printNN(neuralNet *);
//void neuronsPrint(neuralNet *);
time_t getEpochTime(char *);			// used for time comparisons

void initExamples(Matrix& exampleSet, Matrix& outputsSet, Matrix& testSet, Matrix& targetSet){
    char filename[256] = "../data/reducedApptData.csv";
    ifstream input;
    input.open(filename);
    if (!input) {
        cerr << "ERROR! " << filename << " not found" << endl;
    }

    char attr1[257];			// attribute from file
    char attr2[257];			// second attribute used only for time calculation
    int i = -1, j = i - TRAININGSET, index;			// iterators for examples (CHECK OUT my black magic!)
    int age;				// used to store string->int change for age attribute
    time_t schTime, apptTime;		// for calculating days between scheduled and appointment
    double elapsed;

    // clear first line
    input.getline(attr1, 1024);
    //Setup boolean style Matrix pointers
    Matrix* in;
    Matrix* out;
    // executes loop per valid line
    while(i < EXAMPLECOUNT){
        i++;
        j++;
        if(i < TRAININGSET){
            in = &exampleSet;
            out = &outputsSet;
            index = i;
        } 
        else {
            in = &testSet;
            out = &targetSet;
            index = j;
        }
        // 0. sex
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "M") == 0)
            (*in)[index][0] = 0;
	else if ((strcmp(attr1, "F") == 0))
            (*in)[index][0] = 1;
	else
            std::cout << "\tERROR! sex field is: " << attr1 << "for i: " << i << endl;

        ////////////////////////
	// 1. days between section (scheduled day, appointment day) - NEED TO FINISH
        input.getline(attr1, 256, ',');
        schTime = getEpochTime(attr1);
        input.getline(attr2, 256, ',');
        apptTime = getEpochTime(attr2);

        // find time between days
        elapsed = difftime(apptTime, schTime);
        if (elapsed > 1210000.0)					// <14 days
            (*in)[index][1] = 0.6;
        else if (elapsed > 431999.0)				// 5-14 days
            (*in)[index][1] = 0.5;
        else if (elapsed >= 0.0) 					// 0-5 days
            (*in)[index][1] = 0.4;
        else
            std::cout << "ERROR - ELAPSED TIME IS NEGATIVE (" << elapsed << ") ON EXAMPLE: " << i << endl;

        ////////////////////////

        // 2. age
        input.getline(attr1, 256, ',');
        age = atoi(attr1);

        if (age < 25)
            (*in)[index][2] = 0.7;
        else if (age > 49)
            (*in)[index][2] = 0.85;
        else if (age > 24)
            (*in)[index][2] = 0.8;
        else {
            (*in)[index][2] = 0.65;
            std::cout << "\ti " << i << " AGE: " << age << endl;
        }

        // 3. scholarship
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            (*in)[index][3] = 0;
        else
            (*in)[index][3] = 1;
            
        // hypertension
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            (*in)[index][4] = 0;
        else
            (*in)[index][4] = 1;

        // diabetes
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            (*in)[index][5] = 0;
        else
            (*in)[index][5] = 1;

        // alchoholic
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            (*in)[index][6] =0;
        else
            (*in)[index][6] = 1;

        // handicap
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            (*in)[index][7] = 0;
        else
            (*in)[index][7]= 1;

        // reminder
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            (*in)[index][8] = 0;
        else
            (*in)[index][8] = 1;
        // appointment no-show
        input.getline(attr1, 256);				// getline includes newline
	attr1[strlen(attr1) - 1] = '\0';			// replace newline with null terminating char
        if (strcmp(attr1, "No") == 0) {
            (*out)[index][0] = 1;
            (*out)[index][1] = 0;
        }
        else {
            (*out)[index][0] = 0;
            (*out)[index][1] = 1;
        }
    }

    std::cout << "\n---Total Examples Counted: " << i << endl;

    input.close();
}

time_t getEpochTime(char * dateTime) {
    struct tm tm;
    time_t t;

    // "%Y-%m-%dT%H:%M:%SZ"
    if (strptime(dateTime, "%Y-%m-%dT%H:%M:%SZ", &tm) == NULL) {
        std::cout << "---ERROR, TIME NOT CONVERTED SUCCESSFULLY" << endl;
    }

    // daylight savings time check
    tm.tm_isdst = -1;

    // set times to zero
    tm.tm_hour = 0;
    tm.tm_min = 0;
    tm.tm_sec = 0;

    t = mktime(&tm);
    if (t == -1)
        std::cout << "---ERROR, TIME VALUE IS MESSED UP, YO" << endl;

    return t;
}



int main(int argc, char * argv[]) {
    //Classification network with 9 input neurons, one hidden layer with 5 neurons and sigmoid
    //1 output neuron with sigmoid. (Could be 2 with softmax?)
    Network my_net(CLASSIFICATION, 9, {5}, {sigmoid}, 2, softmax);
    //Setup matrices that hold all the data
    Matrix inputs(TRAININGSET+1, 9);
    Matrix outputs(TRAININGSET+1, 2);
    Matrix tests(TESTSET+1, 9);
    Matrix targets(TESTSET+1, 2);

    struct timeval end, start;
    gettimeofday(&start, NULL);
    initExamples(inputs, outputs, tests, targets);
    gettimeofday(&end, NULL);
    cout << "last example output: ";
    cout << outputs.get(TRAININGSET, 0);
    cout << endl;  
    cout << "Runtime for initializing examples in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl << endl;
    //Setup PARAMS
    TrainingParams params;
    params.data = inputs;
    params.target = outputs;
    //TODO: do I have to supress this entire functionality for parrallelization, or just use it?
    params.batch_size = inputs.rows();
    params.max_iterations = 30;
    //Every ten cycles through the whole data set, print out info
    params.epoch_analysis_interval= 5;
    //Stop at this accuracy
    params.min_accuracy = .8;
    params.learning_rate = 0.1;
    //Annealing factor. TODO: How are they incorporating annealing?
    params.annealing_factor = 10;
    params.regularization_factor = 0.0015;
    params.momentum_factor = .5;
    //Data is shuffled
    params.shuffle = true;
    //TRAIN the net with these params
    std::cout << "---Beginning Neural Network Training on Training set of examples (" << TRAININGSET << ")" << endl;
    gettimeofday(&start, NULL);
    my_net.train(params, true);
    gettimeofday(&end, NULL);
    cout << "Learning Runtime in seconds: " << ((end.tv_sec - start.tv_sec) * 
    1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl << endl;	// calculate time elapsed
    //Get predictions for new data and calculate accuracy.
    cout << "---Predicting Testing set of examples (" << TESTSET << ")" << endl;
    gettimeofday(&start, NULL);
    my_net.feed_forward(tests);
    gettimeofday(&end, NULL);
    cout << "Prediction Runtime in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl;		// calculate time elapsed
    //std::cout  << "\n network type: " << my_net._type;
    //print_matrix(my_net.predict(true));
    printf("accuracy   on  tests: %.3f\n", my_net.accuracy(tests, targets));
    printf("accuracy on training: %.3f\n", my_net.accuracy(inputs, outputs));
    return 0;
}