/*
	Author(s): John Nemeth, Isaac Lance
	Description: header file for neural net 
        Sources: 
*/

#ifndef NN_H
#define NN_H

#define INPUTNEURONS 9
#define OUTPUTNEURONS 1
#define EXAMPLECOUNT 110523
#define TRAININGSET 44632
#define TESTSET (EXAMPLECOUNT - TRAININGSET)
#define MAXLAYERS 5
#define MAXNEURONS 9

#include <vector>

//----------------------------------------------------------------//
/* functions for neuron activation and derivation */
//----------------------------------------------------------------//
inline double activate(double val) { return 1.0 / (1.0 + exp(-val)); }
inline double derive(double activatedVal) { return activatedVal * (1.0 - activatedVal); }

//----------------------------------------------------------------//
/* structs for NN */
//----------------------------------------------------------------//

///////////////////////////////////////
/* struct for Neuron
    - Uses sigmoid activation function
*/
typedef struct Neuron {
    double val;			// sum of input weights
    double activatedVal;	// activated by sigmoid function to range it between 0-1
    double derivedVal;		// used for error calculation

    /* Weights:
        - Each neuron holds the strength of connections to neurons in the next layer in its weight array
        - The index in the array is the index of the neuron in the next layer. I.E:
            Neuron 2->Neuron 4 in next layer is at: weights[3]
    */
    double weights[MAXNEURONS];
    double curWError;		// current error of weight value

} neuron;

///////////////////////////////////////
/* struct for example of patient 
	-input vector attributes ordered as: 
	1. sex
	2. # of days between schedule date and appointment date
	3. age 
	4. if had scholarship,
	5. has hypertension
	6. if has diabetes
	7. if alcholic 
	8. if handicap
	9. if received reminder
	- output value is whether they went to appointment
*/
typedef struct Example {

    double inputsByOrder[INPUTNEURONS];
    double output;
} example;

///////////////////////////////////////
// struct for the layers in neural network
typedef struct Layer {
     int neuronCount = 0;
     neuron neurons[MAXNEURONS];
} layer;

///////////////////////////////////////
// struct for neural network
typedef struct NeuralNetwork {

    int layerCount;
    layer layers[MAXLAYERS];
    double biasByLayer[MAXLAYERS];

} neuralNet;

#endif
