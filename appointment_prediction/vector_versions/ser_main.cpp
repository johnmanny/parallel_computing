/*
	Author(s): John Nemeth, Isaac Lance
	Description: implementation file for no show appointment prediction
			project
        Sources: back prop algorithm taken from 'artificial intelligence
                modern approach' and other various info cited in comments
*/

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
#include "ser_nn.h"

using namespace std;

//----------------------------------------------------------------//
/* function prototypes */
//----------------------------------------------------------------//

void initExamples(example *);			// initializes examples array 
void initNN(neuralNet *);			// initializes neural net
void deAlloc(neuralNet *);
void backPropLearning(example *, neuralNet *);	// performs neural net training, returns on 'good' training
void printNN(neuralNet *);
void neuronsPrint(neuralNet *);
time_t getEpochTime(char *);			// used for time comparisons

////////////////////////////////////////////////////////////////////
/*  initExamples reads in partial data from apptdata.csv in the data folder
	and assigns values to the attributes in order to use the sigmoid activation
	function. Called in main. 

 Attributes used and their values between 0.0 to 1.0:
	1. sex 			- male = 0.2, female = 0.3
	2. days between		- 0-5 days = 0.4, 5-14 days = 0.5, 14+ days = 0.6
	3. age 			- unknown = 0.65, under 25 = 0.7, 25-50 = 0.8, 50+ = 0.85
	4. if had scholarship,	- yes = 0.95 , no = 0.05
	5. has hypertension	- yes , no
	6. if has diabetes	- yes , no
	7. if alcholic		- yes , no
	8. if handicap		- yes , no
	9. if received reminder	- yes , no
	10. apptment no-show	- yes , no
*/
void initExamples(example * exampleSet) {

    char filename[256] = "data/trimmedApptData.csv";

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

    // executes loop per valid line
    for (exampleNum = 0; exampleNum < EXAMPLECOUNT; exampleNum++) {
        // 0. sex
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "M") == 0)
            exampleSet[exampleNum].inputsByOrder[0] = 0.2;
	else if ((strcmp(attr1, "F") == 0))
            exampleSet[exampleNum].inputsByOrder[0] = 0.3;
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
            exampleSet[exampleNum].inputsByOrder[1] = 0.6;
        else if (elapsed > 431999.0)				// 5-14 days
            exampleSet[exampleNum].inputsByOrder[1] = 0.5;
        else if (elapsed >= 0.0) 					// 0-5 days
            exampleSet[exampleNum].inputsByOrder[1] = 0.4;
        else
            cout << "ERROR - ELAPSED TIME IS NEGATIVE (" << elapsed << ") ON EXAMPLE: " << exampleNum << endl;

        ////////////////////////

        // 2. age
        input.getline(attr1, 256, ',');
        age = atoi(attr1);

        if (age < 25)
            exampleSet[exampleNum].inputsByOrder[2] = 0.7;
        else if (age > 49)
            exampleSet[exampleNum].inputsByOrder[2] = 0.85;
        else if (age > 24)
            exampleSet[exampleNum].inputsByOrder[2] = 0.8;
        else {
            exampleSet[exampleNum].inputsByOrder[2] = 0.65;
            cout << "\tEXAMPLENUM " << exampleNum << " AGE: " << age << endl;
        }

        // 3. scholarship
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            exampleSet[exampleNum].inputsByOrder[3] = 0.05;
        else
            exampleSet[exampleNum].inputsByOrder[3] = 0.95;
            
        // hypertension
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            exampleSet[exampleNum].inputsByOrder[4] = 0.05;
        else
            exampleSet[exampleNum].inputsByOrder[4] = 0.95;

        // diabetes
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            exampleSet[exampleNum].inputsByOrder[5] = 0.05;
        else
            exampleSet[exampleNum].inputsByOrder[5] = 0.95;

        // alchoholic
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            exampleSet[exampleNum].inputsByOrder[6] = 0.05;
        else
            exampleSet[exampleNum].inputsByOrder[6] = 0.95;

        // handicap
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            exampleSet[exampleNum].inputsByOrder[7] = 0.05;
        else
            exampleSet[exampleNum].inputsByOrder[7] = 0.95;

        // reminder
        input.getline(attr1, 256, ',');
        if (strcmp(attr1, "0") == 0)
            exampleSet[exampleNum].inputsByOrder[8] = 0.05;
        else
            exampleSet[exampleNum].inputsByOrder[8] = 0.95;

        // appointment no-show
        input.getline(attr1, 256);				// getline includes newline
	attr1[strlen(attr1) - 1] = '\0';			// replace newline with null terminating char
        if (strcmp(attr1, "No") == 0)
            exampleSet[exampleNum].output = 0.05;
        else {
            exampleSet[exampleNum].output = 0.95;
        }
    }

    cout << "\n---Total Examples Counted: " << exampleNum << endl;

    input.close();
}

////////////////////////////////////////////////////////////////////
/* Return Epoch time from passed string
	source for function: http://pubs.opengroup.org/onlinepubs/009695399/functions/strptime.html
*/
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

////////////////////////////////////////////////////////////////////
/* init some NN values 
*/
void initNN(neuralNet * nn, int numLayers, char * neuronsPerLayer[]) {

    nn->layerCount = numLayers + 2;			// add input and output layers to provided hidden layer counts
    nn->layers = new layer[numLayers + 2];		// declare number of layers

    cout << "\n---Initialzing Neural Network" << endl;    
    // get command line arguments representing number of hidden neurons per layer
    for (int i = 1; i <= numLayers; i++) {
        cout << "\tHidden layer " << i << " neuron count: " << neuronsPerLayer[i] << endl;
        long int numOfNeurons = strtol(neuronsPerLayer[i], NULL, 10);
        if (numOfNeurons == 0L) {
            cout << "ERROR READING PASSED ARGUMENT VALUES, ENDING PROGRAM" << endl;
            deAlloc(nn);						// release allocated memory
            exit(1);
        }
        nn->layers[i].neurons = new neuron[numOfNeurons];		// declare neurons for this particular hidden layer
        nn->layers[i].neuronCount = numOfNeurons;
    }

    // allocate special values for input and output layers
    nn->layers[0].neuronCount = INPUTNEURONS;
    nn->layers[0].neurons = new neuron[INPUTNEURONS];			// declare & init input layer

    nn->layers[numLayers+1].neuronCount = 1;
    nn->layers[numLayers+1].neurons = new neuron[1];			// declare & init output layer
    nn->layers[numLayers+1].neurons[0].weights = NULL;			// init weights pointer to null
    nn->layers[numLayers+1].neurons[0].curWError = 0.0;
    nn->layers[numLayers+1].neurons[0].val = 0.0; 			// init for use in backprop
    nn->layers[numLayers+1].neurons[0].activatedVal = 0.0; 		// init for use in backprop
   
    /* Connect neurons in each layer to next neurons in sequence
    	- Each neuron in a layer has a number of associated weights that is equal to the
		number of neurons which are present in the next layer
    */
    for (int i = 0; i < nn->layerCount - 1; i++) {			// for all layers except last
        for (int x = 0; x < nn->layers[i].neuronCount; x++) {		// for each neuron in the layer
            nn->layers[i].neurons[x].weights = new double[nn->layers[i+1].neuronCount];
            for (int j = 0; j < nn->layers[i+1].neuronCount; j++) {
                nn->layers[i].neurons[x].weights[j] = 0.0;		// init weights just in case
            }
            nn->layers[i].neurons[x].val = 0.0;
            nn->layers[i].neurons[x].activatedVal = 0.0;
            nn->layers[i].neurons[x].curWError = 0.0;
        }
    }
}

////////////////////////////////////////////////////////////////////
/* print neural net info */
void neuronsPrint(neuralNet * nn) {
    cout << "\n---Printing Neural Network\n\tTotal layers: " << nn->layerCount << endl;
    for (int i = 0; i < nn->layerCount; i++) {
        cout << "--Layer " << i << ":" << endl;
        for (int j = 0; j < nn->layers[i].neuronCount; j++) {
            cout << "\tneuron " << j << ":" << endl;
            cout << "\t\tval: " << nn->layers[i].neurons[j].val << endl;
            cout << "\t\tactivatedVal: " << nn->layers[i].neurons[j].activatedVal << endl;
            cout << "\t\tcurrent error of the weight: " << nn->layers[i].neurons[j].curWError << endl;
            cout << "\t\tderivedVal: " << nn->layers[i].neurons[j].derivedVal << endl;
            if (nn->layers[i].neurons[j].weights != NULL) {
                cout << "\t\tweight connections: " << endl;
                for (int k = 0; k < nn->layers[i+1].neuronCount; k++) {
                    cout << "\t\t\tweight " << k << ": " << nn->layers[i].neurons[j].weights[k] << endl;
                }
            }
        }
    }
    cout << "---Ending Printing Neural Network" << endl;
}
////////////////////////////////////////////////////////////////////
/* remove all allocated memory */
void deAlloc(neuralNet * nn) {

    // remove weights
    for (int i = 0; i < nn->layerCount; i++) {				// for all layers
        if (nn->layers[i].neuronCount != 0) {				// initialized to zero in header file
            for (int x = 0; x < nn->layers[i].neuronCount; x++) { 	// for each neuron in a layer
                if (nn->layers[i].neurons[x].weights != NULL)
                    delete [] nn->layers[i].neurons[x].weights;
            }
        }
    }

    // remove neurons
    for (int i = 0; i < nn->layerCount; i++) {
         if (nn->layers[i].neurons != NULL)
             delete [] nn->layers[i].neurons;
    }

    if (nn->biasByLayer != NULL)
        delete [] nn->biasByLayer;

    if (nn->layers != NULL)
        delete [] nn->layers;
}

///////////////////////////////////////////////////////////////////
/* Performs back prop algorithm for nn learning 
	Pseudocode for similar algorithm found on page 734 of Russell's
	Artificial Intelligence - A modern approach, Figure 18.24.
	Called in main, Only returns a trained neural network. 
	Constantly randomizes weights every iteration. 

	Since this implementation is based on the math in Russell's textbook,
	the error of the output node -can- be negative as well as the weights.
*/
//void backPropLearning(example * examplesArr, neuralNet * nn) {
void backPropLearning(vector<example> examplesArr, neuralNet * nn) {

    int i, j, k, v;
    int exampleIterations = 0;
    int outputLayer = nn->layerCount - 1;
    double learnRate;
    double bestOError = 500.0;			// Used to record which example number provides the best outputerror
    int bestExampleNum = 0;			// ^same

    // seed pseudo-random generator for same sequence every time
    srand(999);
    // create array of biases for each layer
    nn->biasByLayer = new double[outputLayer];                                          // init bias used in each layer
    for (j = 0; j < outputLayer; j++) {
        nn->biasByLayer[j] = (rand() % 1000) / 1000.0;
    }

    while (1) {

        nn->layers[outputLayer].neurons[0].activatedVal = 500.0;			// reset activated val of output neuron for convergence tests

        // assign random initial weights for each neuron connection other than for last later
        for (i = 0; i < outputLayer; i++) {						// for each layer except output layer
            for (k = 0; k < nn->layers[i].neuronCount; k++) {				// for each neuron in the layer
                for (j = 0; j < nn->layers[i+1].neuronCount; j++) {			// for each connection this neuron has, randomize weight
                    nn->layers[i].neurons[k].weights[j] = (rand() % 1000) / 1000.0 ;
                }
            }
        }

        double outputError = 1.0, oldError, oldActivated; 	// variables for convergence tests
        // loop through examples 
	for (j = 0; j < TRAININGSET; j++) {
            /* -------------------------
             * Propogate inputs forward to the output neuron
             *	- Assign input nodes their example values
             *	- At each neuron, sum prior layer's connected neurons activated vals * their weights
             *		before activating current neuron.
             *		- Previously initialized, layer unique bias is used for each neuron
             * ------------------------- */
            //// record previous activated value for output neuron (used in convergence test)
            oldActivated = nn->layers[outputLayer].neurons[0].activatedVal;

            //// assigns input values used in the example
            for (v = 0; v < INPUTNEURONS; v++) {
                nn->layers[0].neurons[v].val = examplesArr[j].inputsByOrder[v];
                nn->layers[0].neurons[v].activatedVal = examplesArr[j].inputsByOrder[v];
            }

            //// Find values for hidden and output neurons
            // for all layers other than the input
            for (i = 1; i < nn->layerCount; i++) {

		// for each neuron in this layer
                for (v = 0; v < nn->layers[i].neuronCount; v++) {

                    // assign layer bias as part of sum
                    nn->layers[i].neurons[v].val = nn->biasByLayer[i-1];

                    // for each connection this neuron has
                    for (k = 0; k < nn->layers[i-1].neuronCount; k++) {
                        nn->layers[i].neurons[v].val += (nn->layers[i-1].neurons[k].activatedVal
    						 * nn->layers[i-1].neurons[k].weights[v]);
                    }
                    nn->layers[i].neurons[v].activate();
                }
            }

            /* -------------------------
             * Propogating error backwards
             *		- Error of all nodes must be found first
             * ------------------------- 
             */
            //// Find error of output node
            oldError = nn->layers[outputLayer].neurons[0].curWError;		// used in convergence test (minimality test)
            nn->layers[outputLayer].neurons[0].derive();
            nn->layers[outputLayer].neurons[0].curWError = (nn->layers[outputLayer].neurons[0].derivedVal
								 * (examplesArr[j].output - nn->layers[outputLayer].neurons[0].activatedVal));
            outputError = nn->layers[outputLayer].neurons[0].curWError;		// outputError used in convergence test readability

            //// This check is to record the example number where the lowest outputerror is achieved
            if (fabs(outputError) < fabs(bestOError)) {
                bestOError = fabs(outputError);
                bestExampleNum = j;
            }
                
            //// Calculate responsibility of error for neurons in earlier layers
            // for each layer other than the output (already calculated)
            for (i = outputLayer - 1; i >= 0; i--) {

                // for each neuron in the layer
                for (v = 0; v < nn->layers[i].neuronCount; v++) {		// for each neuron in this layer
                    neuron * curNeuron = &nn->layers[i].neurons[v];
                    curNeuron->curWError = 0.0;					// reset from last example

                    // sums total responsibility of neuron's weight error considering connection's error value
                    for (k = 0; k < nn->layers[i+1].neuronCount; k++) {
                        curNeuron->curWError += (curNeuron->weights[k] * nn->layers[i+1].neurons[k].curWError);
                    }
                    curNeuron->derive();
                    curNeuron->curWError = curNeuron->curWError * curNeuron->derivedVal;
                }
            }

            // learning rate - decays as failed example iterations increase
            learnRate = 1000.0/(1000.0 + exampleIterations);

            //// Update all weights using calculated errors
            // for each layer except last
            for (i = 0; i < outputLayer; i++) {

                // for each neuron in each layer
                for (v = 0; v < nn->layers[i].neuronCount; v++) {

                    // for each connection the current neuron has
                    neuron * curNeuron = &nn->layers[i].neurons[v];
                    for (k = 0; k < nn->layers[i+1].neuronCount; k++) {
                        curNeuron->weights[k] += (learnRate * curNeuron->activatedVal * nn->layers[i+1].neurons[k].curWError);
                    }
                }
            }
        }

        /* ------------------------
         * Test for convergence:
	 * 	This section tests for satisfiable training of the network on the training set of examples. 
	 * 	- First checks whether the error in the output node on the final example is within an acceptable
	 * 		and defined range
	 * 	- Second, it checks whether the old activated and error values have been getting smaller, implying
         *              convergence towards a minimum.
         * -----------------------
	 */
        if ( fabs(outputError) < 0.02) {			// outputError threshold is arbitrary

            cout.precision(6);
            cout << "\n\t---PASSED OUTPUTERROR TEST - outputError:\t" << fixed << outputError << endl;

            // variables to test for final example evaluation (convergence)
            double actDiff = fabs(oldActivated - nn->layers[outputLayer].neurons[0].activatedVal);
            double errorDiff = fabs(oldError - outputError);
            if ((actDiff > 0.15) && (errorDiff > 0.04)) {	// difference arbitrarily unacceptable
                cout << "\t---FAILED CONVERGENCE TEST\n\toutput difference upon final example:\t"  << actDiff
                 << "\n\terror difference upon final example:\t" << errorDiff << endl;
                cout << "\tloops through example set:\t" << exampleIterations << endl;
                cout << "\tlearnRate:\t\t\t" << learnRate << endl;
                cout << "\toutput target:\t\t\t" << examplesArr[TRAININGSET-1].output << endl;
                cout << "\toutput result:\t\t\t" << nn->layers[outputLayer].neurons[0].activatedVal << endl;
                cout << setprecision(15) << "\tbest outputError achieved:\t\t" << bestOError << "\n\t- for example num:\t\t" << bestExampleNum << endl;
                cout << "\tCONTINUING BACK PROP LEARNING..." << endl;
                exampleIterations++;
                continue;
            }
            cout << "\t---PASSED OUTPUT ERROR & CONVERGENCE TEST\n\toutput difference upon final example:\t"  << actDiff
                 << "\n\terror difference upon final example:\t" << errorDiff << endl;
            cout << "\tloops through example set:\t" << exampleIterations << endl;
            cout << "\tlearnRate:\t\t\t" << learnRate << endl;
            cout << "\toutput target:\t\t\t" << examplesArr[TRAININGSET-1].output << endl;
            cout << "\toutput result:\t\t\t" << nn->layers[outputLayer].neurons[0].activatedVal << endl;
            cout << setprecision(15) << "\tbest outputError achieved:\t\t" << bestOError << "\n\t- for example num:\t\t" << bestExampleNum << endl;
            return;
        }
        exampleIterations++;

        if ((exampleIterations % 100) == 0) {
            //neuronsPrint(nn);
            cout << "\n---Convergence not achieved\n\tIterations through examples:\t\t" << exampleIterations << endl;
            cout << "\tlearnRate:\t\t\t\t" << learnRate << "\n\tlast example outputError:\t\t" << outputError << endl;
            cout << "\tlast example output target:\t\t" << examplesArr[TRAININGSET-1].output << endl;
            cout << "\tlast example output result:\t\t" << nn->layers[outputLayer].neurons[0].activatedVal << endl;
            cout << setprecision(15) << "\tbest outputError achieved:\t\t" << bestOError << "\n\t- for example num:\t\t" << bestExampleNum << endl;
	}
    }
}

//////////////////////////////////////////////////////
//void predictExamples(neuralNet * nn, example * examplesArr) {
void predictExamples(neuralNet * nn, vector<example> examplesArr) {

    const double correctThreshold = 0.5;        // a simple threshold for yes/no predictions
    int correct = 0;                            // number of correct predictions
    int testIndex = TRAININGSET;

    // define output neuron pointer for readability (lol)
    neuron * outputNeuron = &nn->layers[nn->layerCount-1].neurons[0];

    for (int j = 0; j < TESTSET; j++) {
        /* -------------------------
         * Propogate inputs forward to the output neuron
         * ------------------------- */

        //// assigns input values used in the example
        for (int v = 0; v < INPUTNEURONS; v++) {
            nn->layers[0].neurons[v].val = examplesArr[testIndex].inputsByOrder[v];
            nn->layers[0].neurons[v].activatedVal = examplesArr[testIndex].inputsByOrder[v];
        }

        //// find values for hidden and output neurons
        // for all layers other than the input
        for (int i = 1; i < nn->layerCount; i++) {

            // for each neuron in this layer
            for (int v = 0; v < nn->layers[i].neuronCount; v++) {

                // assign layer bias as part of sum
                nn->layers[i].neurons[v].val = nn->biasByLayer[i-1];

                // for each connection this neuron has
                for (int k = 0; k < nn->layers[i-1].neuronCount; k++) {
                    nn->layers[i].neurons[v].val += (nn->layers[i-1].neurons[k].activatedVal
                                             * nn->layers[i-1].neurons[k].weights[v]);
                }
                nn->layers[i].neurons[v].activate();
            }
        }

        //// Record if prediction was correct (simple halfway threshold used)
        // correct if: example is .95 - activated is above .5
	//		example is .05 - activated below .5
        if ((examplesArr[testIndex].output != 0.05) && (outputNeuron->activatedVal >= correctThreshold)) {
            // In this case, example is yes and network is yes (above simple threshold)
            correct++;
        }
        if ((examplesArr[testIndex].output != 0.95) && (outputNeuron->activatedVal < correctThreshold)) {
            // In this case example is no, output is below threshold (correct output)
            correct++;
        }

        testIndex++;
    }

    cout << setprecision(2) << "\tNumber of Examples predicted correctly: " <<  correct << "\tOut of " << TESTSET << " examples" << endl;
    cout << "\tPercent of correct predictions: " << setprecision(2) << correct / ((double)TESTSET) * 100.0 << endl;
}

////////////////////////////////////////////////////////////////////
/* Declares and initializes examples and neural net values. Then trains
	Neural Network in backPropLearning.
*/
int main(int argc, char * argv[]) {

    // initialize neuralnetwork with passed arguments (exits if not)
    neuralNet backPropNN;
    initNN(&backPropNN, argc - 1, argv);

    // declare example array on heap
    example * exampleSet = new example[EXAMPLECOUNT];
    vector <example> exVec;

    struct timeval end, start;
    gettimeofday(&start, NULL);				// record time at start
    initExamples(exampleSet);				// initialize all examples using EXAMPLECOUNT (nn.h)
    gettimeofday(&end, NULL);				// record time at end
   
    for (int i = 0; i < EXAMPLECOUNT; i++) {
        exVec.push_back(exampleSet[i]);
    }
    cout << "last example output: " << exVec[EXAMPLECOUNT].output << endl;
    auto rng = default_random_engine{};
    shuffle(exVec.begin(), exVec.end(), rng);
    //random_shuffle(exVec.begin(), exVec.end());
    
    cout << "Runtime for initializing examples in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl << endl;


    cout << "---Beginning Neural Network Training on Training set of examples (" << TRAININGSET << ")" << endl;
    gettimeofday(&start, NULL);
    //backPropLearning(exampleSet, &backPropNN);
    backPropLearning(exVec, &backPropNN);
    gettimeofday(&end, NULL);

    cout << "Learning Runtime in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl << endl;	// calculate time elapsed


    cout << "---Predicting Testing set of examples (" << TESTSET << ")" << endl;
    gettimeofday(&start, NULL);
    predictExamples(&backPropNN, exVec);
    gettimeofday(&end, NULL);
    cout << "Prediction Runtime in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl;		// calculate time elapsed


    deAlloc(&backPropNN);					// deallocate neural network dynamic memory
    delete [] exampleSet;

    return 0;
}
