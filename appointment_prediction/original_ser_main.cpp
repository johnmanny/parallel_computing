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
#include <random>
#include <limits>
#include <stdlib.h>	// for atoi
#include <time.h>
#include <iomanip>
#include <sstream>
#include <sys/time.h>	// for timing data
#include "nn.h"

using namespace std;

//----------------------------------------------------------------//
/* function prototypes */
//----------------------------------------------------------------//

void initExamples(example *);			// initializes examples array 
void initNN(neuralNet *);			// initializes neural net
void backPropLearning(example *, neuralNet *);	// performs neural net training, returns on 'good' training
void printNN(neuralNet *);
time_t getEpochTime(char *);		// used for time comparisons

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
	10. went to appointment	- yes , no

Tasks to do:
	- skip first line
	For every line:
		- skip first comma
		- skip second comma
		- define sex
		- define schedule and appointment day(two commas)
			- determine days between and set classification
		- define age
		- skip neighborhood
		- grab scholarship, hypertension, diabetes, alchomlism,
			handicap, reminder received, and output (whether showed up)
*/
void initExamples(example * exampleSet) {

    char filename[256] = "data/trimmedApptData.csv";
    //char filename[256] = "data/backup.csv";

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
        if (elapsed > 1210000)					// <14 days
            exampleSet[exampleNum].inputsByOrder[1] = 0.6;
        else if (elapsed > 431999)				// 5-14 days
            exampleSet[exampleNum].inputsByOrder[1] = 0.5;
        else if (elapsed >= 0) 					// 0-5 days
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

        // went to appt
        input.getline(attr1, 256);				// getline includes newline
	attr1[strlen(attr1) - 1] = '\0';			// replace newline with null terminating char
        if (strcmp(attr1, "No") == 0)
            exampleSet[exampleNum].inputsByOrder[9] = 0.05;
        else {
            exampleSet[exampleNum].inputsByOrder[9] = 0.95;
        }
    }

    cout << "---Total Examples Counted: " << exampleNum << endl;

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
    //wistringstream ss{ dateTime };
    //tm dt;
    //ss >> get_time(&dt, dateTimeFormat.c_str());
    return t;
}

////////////////////////////////////////////////////////////////////
/* init some NN values 
*/
void initNN(neuralNet * nn) {

    nn->outputNeuron.activatedVal = 500.0;
    nn->layerCount = 3;
}

////////////////////////////////////////////////////////////////////
/* prints NN info
 - NEEDS UPDATING
 */
void printNN(neuralNet * nn) {

    cout << "---INPUT BREAKDOWN:" << endl;
    char inputName[256];
    for (int i = 0; i < INPUTNEURONS; i++) {
        switch (i) {
            case 0:
                strcpy(inputName, "gender");
                break;
            case 1:
                strcpy(inputName, "time between appointments");
                break;
            case 2:
                strcpy(inputName, "age bracket");
                break;
            case 3:
                strcpy(inputName, "has scholarship");
                break;
            case 4:
                strcpy(inputName, "has hypertension");
                break;
            case 5:
                strcpy(inputName, "has diabetes");
                break;
            case 6:
                strcpy(inputName, "has alcoholism");
                break;
            case 7:
                strcpy(inputName, "has handicap");
                break;
            case 8:
                strcpy(inputName, "recieved reminder");
                break;
        }
        for (int y = 0; y < HIDDENNEURONS; y++) {
            cout << inputName << "\t->hidden" << y << " weight: " << nn->inputWeights[(i * HIDDENNEURONS) + y] << endl;
        }
        cout << inputName << " val: " << nn->inputNeurons[i].val << "\n\tending activatedVal: " << nn->inputNeurons[i].activatedVal << endl;
    }
    for (int i = 0; i < HIDDENNEURONS; i++) {
        cout << "hidden" << i << "->output weight: " << nn->hiddenWeights[i] << endl;
        cout << "\tval: " << nn->hiddenNeurons[i].val << " activated Val: " << nn->hiddenNeurons[i].activatedVal
             << " derived val: " << nn->hiddenNeurons[i].derivedVal << endl;
    }
    cout << "output's val: " << nn->outputNeuron.val << " activated Val: " << nn->outputNeuron.activatedVal
         << " derived val: " << nn->outputNeuron.derivedVal << endl;
    cout << "yes: 0.95 no: 0.05" << endl;
}

////////////////////////////////////////////////////////////////////
/* Performs back prop algorithm for nn learning 
	Pseudocode for similar algorithm found on page 734 of Russell's
	Artificial Intelligence - A modern approach, Figure 18.24.
	Called in main, Only returns a trained neural network. 
	Constantly randomizes weights every iteration. 

	Since this implementation is based on the math in Russell's textbook,
	the error of the output node -can- be negative as well as the weights.
*/
void backPropLearning(example * examplesArr, neuralNet * nn) {

    int inputWCount = INPUTNEURONS * HIDDENNEURONS;
    int exampleIterations = 0;
    double learnRate;
    // used for generating random weights
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    while (1) {

        nn->outputNeuron.activatedVal = 500.0;		// reset activated val for convergence tests
        // assign random initial weights for each neuron connection
        for (int i = 0; i < inputWCount; i++) {
            nn->inputWeights[i] = dis(gen);
        }
        for (int i = 0; i < HIDDENNEURONS; i++) {
            nn->hiddenWeights[i] = dis(gen);
        }

        int j;
        double outputError = 1.0, oldError, oldActivated; 	// variables for convergence tests
        // loop through examples 

	for (j = 0; j < TRAININGSET; j++) {
            /* ---propogate inputs forward to output---*/
            for (int v = 0; v < INPUTNEURONS; v++) {	// assign initial vals
                nn->inputNeurons[v].val = examplesArr[j].inputsByOrder[v];
                nn->inputNeurons[v].activatedVal = examplesArr[j].inputsByOrder[v];
            }
            for (int x = 0; x < HIDDENNEURONS; x++) {		// produce vals for hidden neurons
                nn->hiddenNeurons[x].val = 0.35;			// bias
                for (int z = 0; z < INPUTNEURONS; z++) {	// sum vals of each input and weight
                    nn->hiddenNeurons[x].val += (nn->inputNeurons[z].activatedVal
						 * nn->inputWeights[x + (HIDDENNEURONS * z)]);
                }
                nn->hiddenNeurons[x].activate();	// executes activation func tied to neurons
            }

            nn->outputNeuron.val = 0.6;		// for bias
            for (int x = 0; x < HIDDENNEURONS; x++) {	// propagate input to output neuron
                nn->outputNeuron.val += (nn->hiddenNeurons[x].activatedVal * nn->hiddenWeights[x]);
            }
            oldActivated = nn->outputNeuron.activatedVal;
            nn->outputNeuron.activate();

            /*---propogate error backwards---*/
            // find error of output
            nn->outputNeuron.derive();
            oldError = outputError;			// used for convergence test to ensure covergence (minimality)
	    // ---------------------NEED TO FIX
            outputError = (nn->outputNeuron.derivedVal * (examplesArr[j].output - nn->outputNeuron.activatedVal));
            //outputError = 0.5 * ((examplesArr[j].output - nn->outputNeuron.activatedVal) * (examplesArr[j].output - nn->outputNeuron.activatedVal));
            
	    // ---------------------^TOFIX?
            //cout << "\n\tneural network activated val * : " << nn->outputNeuron.activatedVal << endl;

            //// find error of hidden neuron weights (the delta rule, based on the chain rule)
            //		good explanation: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
            double hiddenError[HIDDENNEURONS];
            for (int x = 0; x < HIDDENNEURONS; x++) {
                nn->hiddenNeurons[x].derive();
                
                hiddenError[x] = nn->hiddenNeurons[x].derivedVal * (nn->hiddenWeights[x] * outputError);
                //other sol: hiddenError[x] = (nn->outputNeuron.activatedVal - examplesArr[j].output)
		//		* (nn->outputNeuron.activatedVal * (1.0 - nn->outputNeuron.activatedVal)
		//		* nn->hiddenNeurons[x].activatedVal);
                //other better sol: hiddenError[x] = (nn->outputNeuron.activatedVal - examplesArr[j].output)
		//		* nn->outputNeuron.derivedVal * nn->hiddenNeurons[x].activatedVal;
            }


            int neuWeightStartIndex;					// for when compilation excludes automatic optimizations
            learnRate = 5000.0/(1000.0 + exampleIterations);		// learning rate - decays as failed example iterations increase (increased precision)

            /* ---update each weight using errors--- */
            // update input weights
            for (int x = 0; x < INPUTNEURONS; x++) {
                neuWeightStartIndex = x * HIDDENNEURONS;
                for (int y = 0; y < HIDDENNEURONS; y++) {
                    nn->inputWeights[neuWeightStartIndex + y] += (learnRate * nn->inputNeurons[x].activatedVal * hiddenError[y]);
                    //double WeightIndexError = (nn->outputNeuron.activatedVal - examplesArr[j].output) * nn->hiddenNeurons[y].derivedVal
		    //				* nn->inputNeurons[x].activatedVal;
                    //nn->inputWeights[neuWeightStartIndex + y] -= (learnRate * WeightIndexError);
                }
            }
            // update hidden neuron weights
            for (int x = 0; x < HIDDENNEURONS; x++) {
                nn->hiddenWeights[x] += (learnRate * nn->hiddenNeurons[x].activatedVal * outputError);
                //nn->hiddenWeights[x] += (learnRate * hiddenError[x]);
            }
        }

        //printNN(nn);
        /* Test for convergence:
	 * 	This section tests for satisfiable training of the network on the training set of examples. 
	 * 	- First checks whether the error in the output node on the final example is within an acceptable
	 * 		and defined range
	 * 	- Second, it checks whether the old activated value of the output neuron and the error of the last example
	 * 		has been trending downward, implying a local minimum of maximum 
	 * NOT WORKING CORRECTLY - REANALYZE ERROROUTPUT METHODS*/
        if ( fabs(outputError) < 0.002) {

            cout.precision(15);
            cout << "\t---PASSED OUTPUTERROR TEST THROUGH ITERATION OF EXAMPLES AND NEW WEIGHTS - VALUE: " << fixed << outputError << endl;

            // variables to test for final example evaluation (convergence)
            double actDiff = fabs(oldActivated - nn->outputNeuron.activatedVal);
            double errorDiff = fabs(oldError - outputError);
            if ((actDiff > 0.01) && (errorDiff > 0.01)) {	// difference arbitrarily unacceptable
                cout << "\tFAILED CONVERGENCE TEST - CONTINUING BACK PROP LEARNING..." << endl;
                //exampleIterations = 0;
                continue;
            }
            cout << "\tPASSED CONVERGENCE TEST\n\toutput difference upon final example: " << actDiff
                 << "\n\terror difference upon final example: " << errorDiff << endl;
            cout << "\tloops through example set: " << exampleIterations << endl;
            cout << "\tlearnRate: " << learnRate << endl;
            cout << "\toutput target " << examplesArr[TRAININGSET-1].output << endl;
            cout << "\toutput result: " << nn->outputNeuron.activatedVal << endl;
            //printNN(nn);
            return;
        }
        exampleIterations++;

        if ((exampleIterations % 100) == 0) {
            cout << "\n---Convergence not achieved\n\tIterations through examples: " << exampleIterations << endl;
            cout << "\tlearnRate: " << learnRate << "\n\toutputError: " << outputError << endl;
            cout << "\toutput target " << examplesArr[TRAININGSET-1].output << endl;
            cout << "\toutput result: " << nn->outputNeuron.activatedVal << endl;
            //printNN(nn);
	}

    }
}

//////////////////////////////////////////////////////
void predictExamples(neuralNet * nn, example * examplesArr) {

    int correct = 0;		// number of correct predictions
    int testIndex = TRAININGSET + 1;

    for (int j = 0; j < TESTSET; j++) {
    // fix indents
    /* ---propogate inputs forward to output---*/
    for (int v = 0; v < INPUTNEURONS; v++) {	// assign initial vals
        nn->inputNeurons[v].val = examplesArr[testIndex].inputsByOrder[v];
        nn->inputNeurons[v].activatedVal = examplesArr[testIndex].inputsByOrder[v];
    }
    for (int x = 0; x < HIDDENNEURONS; x++) {		// produce vals for hidden neurons
        nn->hiddenNeurons[x].val = 0.35;			// bias
        for (int z = 0; z < INPUTNEURONS; z++) {	// sum vals of each input and weight
            nn->hiddenNeurons[x].val += (nn->inputNeurons[z].activatedVal
		 * nn->inputWeights[x + (HIDDENNEURONS * z)]);
        }
        nn->hiddenNeurons[x].activate();	// executes activation func tied to neurons
    }

    nn->outputNeuron.val = 0.6;		// for bias
    for (int x = 0; x < HIDDENNEURONS; x++) {	// propagate input to output neuron
        nn->outputNeuron.val += (nn->hiddenNeurons[x].activatedVal * nn->hiddenWeights[x]);
    }
    nn->outputNeuron.activate();
    if ((examplesArr[testIndex].output != 0.05) && (nn->outputNeuron.activatedVal >= 0.5)) {	// ouput yes, nn is yes
        correct++;
    } else {

        if (nn->outputNeuron.activatedVal < 0.5) { 						// output no, nn no
            correct++;
        }

    }
    testIndex++;
    // fix indents
    }
    cout << "Number of Examples predicted correctly: " <<  correct << "\tOut of " << TESTSET << " examples" << endl;
    cout << "Percent of correct predictions: " << setprecision(2) << ((double)correct) / ((double)TESTSET) * 100.0 << endl;
        

}
////////////////////////////////////////////////////////////////////
/* Declares and initializes examples and neural net values. Then trains
	Neural Network in backPropLearning.
*/
int main() {

    struct timeval end, start;

    example * exampleSet = new example[EXAMPLECOUNT];	// declare example array on heap
    neuralNet backPropNN;				// declare neural net

    gettimeofday(&start, NULL);				// record time at start
    initExamples(exampleSet);				// initialize all examples
    gettimeofday(&end, NULL);				// record time at end

    cout << "Runtime for initializing exmaples in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl << endl;		// calculate time elapsed

    // initialize neuralnetwork
    initNN(&backPropNN);

    cout << "---Beginning Neural Network Training on Training set of examples (" << TRAININGSET << ")" << endl;
    gettimeofday(&start, NULL);				// record time at start
    backPropLearning(exampleSet, &backPropNN);
    gettimeofday(&end, NULL);				// record time at end

    cout << "Learning Runtime in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl << endl;	// calculate time elapsed


    cout << "---Predicting Testing set of examples (" << TESTSET << ")" << endl;
    gettimeofday(&start, NULL);				// record time at start
    predictExamples(&backPropNN, exampleSet);
    gettimeofday(&end, NULL);				// record time at end
    cout << "Prediction Runtime in seconds: " << ((end.tv_sec - start.tv_sec) * 
		1000000LL + (end.tv_usec - start.tv_usec))/1000000.0 << endl;		// calculate time elapsed

    delete [] exampleSet;

    return 0;
}
