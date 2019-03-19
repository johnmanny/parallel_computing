/*
	Author(s): John Nemeth, Isaac Lance
	Description: header file for neural net 
        Sources: 
*/

#ifndef NN_H
#define NN_H

#define INPUTNEURONS 9
#define HIDDENNEURONS 5
#define OUTPUTNEURONS 1
//#define TOTALEXAMPLES 110522
#define EXAMPLECOUNT 110522

#define TRAININGSET 100007
#define TESTSET EXAMPLECOUNT - TRAININGSET

//----------------------------------------------------------------//
/* structs for NN */
//----------------------------------------------------------------//

///////////////////////////////////////
/* struct for Neuron
reference structure from https://www.youtube.com/watch?v=PQo78WNGiow
*/
typedef struct Neuron {

    void activate() { this->activatedVal = (1.0 / (1.0 + exp((-this->val)) ) ); };
    void derive() {this->derivedVal = (this->activatedVal * (1.0 - this->activatedVal)); };
    double val;
    double activatedVal;
    double derivedVal;
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
// struct for neural network
typedef struct NeuralNetwork {

    int layerCount;
    /* inputWeights[0] = weight from input node 1 to hidden node 1
	inputWeights[1] = weight from input node 1 to hidden node 2
	inputWeights[2] = weight from input node 1 to hidden node 3
	inputWeights[3] = weight from input node 2 to hidden node 1
	inputWeights[17] = weight from input nod 9 to hidden node 3
    */
    double inputWeights[INPUTNEURONS * HIDDENNEURONS];
    double hiddenWeights[HIDDENNEURONS * OUTPUTNEURONS];
    neuron inputNeurons[INPUTNEURONS];
    neuron hiddenNeurons[HIDDENNEURONS];
    neuron outputNeuron;

} neuralNet;

#endif
