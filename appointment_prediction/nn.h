/*
	Author(s): John Nemeth, Isaac Lance
	Description: header file for neural net 
        Sources: 
*/

#ifndef NN_H
#define NN_H

#define INPUTNEURONS 9
#define HIDDENNEURONS 5
#define HIDDENLAYERS 2
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
    - Uses sigmoid activation function
*/
typedef struct Neuron {
    void activate() { this->activatedVal = (1.0 / (1.0 + exp((-this->val)) ) ); };
    void derive()   { this->derivedVal = (this->activatedVal * (1.0 - this->activatedVal)); };
    double val;			// sum of input weights
    double activatedVal;	// activated by sigmoid function to range it between 0-1
    double derivedVal;		// used for error calculation

    double * weights;		// holds weights for connecting neurons in next layer
    double curWError;		// current error of weight value
    //nConn * conn;		// list of which neurons this neuron connects to

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

/*
///////////////////////////////////////
// struct for weight connection between two neurons
typedef struct NeuronConnection {

    neuron * from;
    neuron * to;
    double weight;    
} nConn;
*/

///////////////////////////////////////
// struct for the layers in neural network
typedef struct Layer {
     int neuronCount = 0;
     neuron * neurons;
} layer;

///////////////////////////////////////
// struct for neural network
typedef struct NeuralNetwork {

    int layerCount;
    /* assuming 3 hidden nodes:
        inputWeights[0] = weight from input node 1 to hidden node 1
	inputWeights[1] = weight from input node 1 to hidden node 2
	inputWeights[2] = weight from input node 1 to hidden node 3
	inputWeights[3] = weight from input node 2 to hidden node 1
	inputWeights[17] = weight from input nod 9 to hidden node 3
    */
    //// hidden neuron x at hidden layer Y starts at index:
    // (HIDDENNEURONS*HIDDENNEURONS) + HIDDENNEURONS*Y + x
    //		I.E: with HIDDENNEURONS = 5, LAYERS = 3, 1 output neuron (last set of weights point to 1 output)
    //          Hidden neuron 2->output weight at hidden layer 3 is:
    //		(5*5)*5*Y
    //original: double hiddenWeights[((HIDDENNEURONS * HIDDENNEURONS) * (HIDDENLAYERS-1)) + HIDDENNEURONS];
    //original: double inputWeights[INPUTNEURONS * HIDDENNEURONS];
    //original: double hiddenWeights[HIDDENNEURONS * OUTPUTNEURONS];
    //double allWeights[(INPUTNEURONS * HIDDENNEURONS) + 
    //		(HIDDENNEURONS * (HIDDENLAYERS - 1) + HIDDENNEURONS];
    //original: neuron inputNeurons[INPUTNEURONS];
    //original: neuron hiddenNeurons[HIDDENNEURONS];
    //secondary: neuron hiddenNeurons[HIDDENNEURONS * HIDDENLAYERS];
    
    //neuron inputNeurons[INPUTNEURONS];		// number of input neurons won't change
    layer * layers;
    //neuron outputNeuron;

} neuralNet;

#endif
