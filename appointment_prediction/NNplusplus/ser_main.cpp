#include "Matrix.hpp"
#include "NeuralNet.hpp"
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

#define EXAMPLECOUNT 110523
#define TRAININGSET 100000
#define TESTSET EXAMPLECOUNT - TRAININGSET


using namespace std;
//----------------------------------------------------------------//
/* function prototypes */
//----------------------------------------------------------------//

void initExamples(vector<Matrix>& exampleSet, vector<Matrix>& outputsSet, vector<int>& indexSet);         // initializes examples array 
//void initNN(NeuralNet *);			// initializes neural net
//void deAlloc(NeuralNet *);
//void backPropLearning(example *, neuralNet *);	// performs neural net training, returns on 'good' training
//void printNN(neuralNet *);
//void neuronsPrint(neuralNet *);
time_t getEpochTime(char *);			// used for time comparisons

void initExamples(vector<Matrix>& exampleSet, vector<Matrix>& outputsSet, vector<int>& indexSet){
    char filename[256] = "../data/trimmedApptData.csv";
    ifstream input;
    input.open(filename);
    if (!input) {
        cerr << "ERROR! " << filename << " not found" << endl;
    }

    char attr1[257];			// attribute from file
    char attr2[257];			// second attribute used only for time calculation
    int exampleNum;			// iterator for examples
    int age;				// used to store string->int change for age attribute
    time_t schTime, apptTime;		// for calculating days between scheduled and appointment
    double elapsed;

    // clear first line
    input.getline(attr1, 1024);
    // Init Matrix
    Matrix mtrx;
    Matrix out;
    // executes loop per valid line
    for (exampleNum = 0; exampleNum < EXAMPLECOUNT; exampleNum++) {
    //for (exampleNum = 0; exampleNum < 10; exampleNum++) {
        mtrx = Matrix(1,9);
        out = Matrix(1,1);
        // 0. sex
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "M") == 0)
            mtrx(0,0) = 0.2;
	else if ((strcmp(attr1, "F") == 0))
            mtrx(0,0) = 0.3;
	else
            cout << "\tERROR! sex field is: " << attr1 << "for exampleNum: " << exampleNum << endl;

        ////////////////////////
	// 1. days between section (scheduled day, appointment day) - NEED TO FINISH
        input.getline(attr1, 256, ',');
        schTime = getEpochTime(attr1);
        input.getline(attr2, 256, ',');
        apptTime = getEpochTime(attr2);

        // find time between days
        elapsed = difftime(apptTime, schTime);
        if (elapsed > 1210000.0)					// <14 days
            mtrx(0,1) = 0.6;
        else if (elapsed > 431999.0)				// 5-14 days
            mtrx(0,1) = 0.5;
        else if (elapsed >= 0.0) 					// 0-5 days
            mtrx(0,1) = 0.4;
        else
            cout << "ERROR - ELAPSED TIME IS NEGATIVE (" << elapsed << ") ON EXAMPLE: " << exampleNum << endl;

        ////////////////////////

        // 2. age
        input.getline(attr1, 256, ',');
        age = atoi(attr1);

        if (age < 25)
            mtrx(0,2) = 0.7;
        else if (age > 49)
            mtrx(0,2) = 0.85;
        else if (age > 24)
            mtrx(0,2) = 0.8;
        else {
            mtrx(0,2) = 0.65;
            cout << "\tEXAMPLENUM " << exampleNum << " AGE: " << age << endl;
        }

        // 3. scholarship
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            mtrx(0,3) = 0.05;
        else
            mtrx(0,3) = 0.95;
            
        // hypertension
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            mtrx(0,4) = 0.05;
        else
            mtrx(0,4) = 0.95;

        // diabetes
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            mtrx(0,5) = 0.05;
        else
            mtrx(0,5) = 0.95;

        // alchoholic
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            mtrx(0,6) = 0.05;
        else
            mtrx(0,6) = 0.95;

        // handicap
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            mtrx(0,7) = 0.05;
        else
            mtrx(0,7) = 0.95;

        // reminder
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            mtrx(0,8) = 0.05;
        else
            mtrx(0,8) = 0.95;

        // appointment no-show
        input.getline(attr1, 256);				// getline includes newline
	attr1[strlen(attr1) - 1] = '\0';			// replace newline with null terminating char
        if (strcmp(attr1, "No") == 0)
            out(0,0) = 0.05;
        else {
            out(0,0) = 0.95;
        }
    exampleSet.push_back(mtrx);
    outputsSet.push_back(out);
    indexSet.push_back(exampleNum);
    }

    cout << "\n---Total Examples Counted: " << exampleNum << endl;

    input.close();
}

time_t getEpochTime(char * dateTime) {
    struct tm tm;
    time_t t;

    // "%Y-%m-%dT%H:%M:%SZ"
    if (strptime(dateTime, "%Y-%m-%dT%H:%M:%SZ", &tm) == NULL) {
        cout << "---ERROR, TIME NOT CONVERTED SUCCESSFULLY" << endl;
    }

    // daylight savings time check
    tm.tm_isdst = -1;

    // set times to zero
    tm.tm_hour = 0;
    tm.tm_min = 0;
    tm.tm_sec = 0;

    t = mktime(&tm);
    if (t == -1)
        cout << "---ERROR, TIME VALUE IS MESSED UP, YO" << endl;

    return t;
}

int train(NeuralNet& NN, vector<Matrix>inputs, vector<Matrix>outputs, vector<int> indexSet){
    for (int i = 0; i < TRAININGSET; i++){
        int index = indexSet[i];
        NN.trainingCycle(inputs[index], outputs[index]);
    }
    return 0;
}

void predictExamples(NeuralNet& NN, vector<Matrix>inputs, vector<Matrix>outputs, vector<int> indexSet) {

    const double correctThreshold = 0.5;        // a simple threshold for yes/no predictions
    int correct = 0;                            // number of correct predictions
    int index;
    for (int j = TRAININGSET; j < EXAMPLECOUNT; j++) {
        index = indexSet[j];
        double prediction = NN.queryNet(inputs[index])(0, 0);

        //// Record if prediction was correct (simple halfway threshold used)
        // correct if: example is .95 - activated is above .5
	    //		example is .05 - activated below .5
        if ((outputs[index](0,0) != 0.05) && (prediction >= correctThreshold)) {
            // In this case, example is yes and network is yes (above simple threshold)
            correct++;
        }
        if ((outputs[index](0,0) != 0.95) && (prediction < correctThreshold)) {
            // In this case example is no, output is below threshold (correct output)
            correct++;
        }

    }

    cout << setprecision(2) << "\tNumber of Examples predicted correctly: " <<  correct << "\tOut of " << TESTSET << " examples" << endl;
    cout << "\tPercent of correct predictions: " << setprecision(2) << correct / ((double)TESTSET) * 100.0 << endl;
}

int main(int argc, char * argv[]) {
    NeuralNet my_net(9, *argv[1], 1, *argv[2], 0.1);
    vector<Matrix> inputs;
    vector<Matrix> outputs;
    vector<int> indexes;

    struct timeval end, start;
    gettimeofday(&start, NULL);
    initExamples(inputs, outputs, indexes);
    gettimeofday(&end, NULL);
    cout << "last example output: ";
    cout << outputs.size();
    cout << EXAMPLECOUNT-2;
    outputs[EXAMPLECOUNT-2].printMtrx();
    cout << endl;
    //Need random indexing but identical shuffling is unreliable, so using indexes as a middle man

    auto rng = default_random_engine{};
    shuffle(indexes.begin(), indexes.end(), rng);
    
    cout << "Runtime for initializing examples in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl << endl;


    cout << "---Beginning Neural Network Training on Training set of examples (" << TRAININGSET << ")" << endl;
    gettimeofday(&start, NULL);
    train(my_net, inputs, outputs, indexes);
    gettimeofday(&end, NULL);

    cout << "Learning Runtime in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl << endl;	// calculate time elapsed


    cout << "---Predicting Testing set of examples (" << TESTSET << ")" << endl;
    gettimeofday(&start, NULL);
    predictExamples(my_net, inputs, outputs, indexes);
    gettimeofday(&end, NULL);
    cout << "Prediction Runtime in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl;		// calculate time elapsed

    /*
    //deAlloc(&backPropNN);					// deallocate neural network dynamic memory
    //delete [] exampleSet;
    */
    return 0;
}