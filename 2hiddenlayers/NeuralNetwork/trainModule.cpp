#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>
#include "trainModule.h"
#include "testModule.h"
#include "const.h"

using namespace std;


void initializeTrainModule() {
	// Layer 1 - Layer 2 = Input layer - Hidden layer
	for (int i = 1; i <= inputNeuronNumber; ++i) {
		weightsInputHidden[i] = new double[hiddenNeuronNumber + 1];
		delta1[i] = new double[hiddenNeuronNumber + 1];
	}

	inputLayerOUT = new double[inputNeuronNumber + 1];
	/////

	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		weightsInputHidden2nd[i] = new double[hiddenNeuronNumber + 1];
		delta2[i] = new double[hiddenNeuronNumber + 1];
	}

	hiddenLayerIn = new double[hiddenNeuronNumber + 1];
	hiddenLayerOUT = new double[hiddenNeuronNumber + 1];
	deltaHiddenSum = new double[hiddenNeuronNumber + 1];


	// Layer 2 - Layer 3 = Hidden layer - Output layer
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		weightstHiddenOutput[i] = new double[outputNeuronNumber + 1];
		delta3[i] = new double[outputNeuronNumber + 1];
	}

	hiddenLayerIn2nd = new double[hiddenNeuronNumber + 1];
	hiddenLayerOUT2nd = new double[hiddenNeuronNumber + 1];
	deltaHiddenSum2nd = new double[hiddenNeuronNumber + 1];



	// Layer 3 - Output layer
	outputLayerIN = new double[outputNeuronNumber + 1];
	outputLayerOUT = new double[outputNeuronNumber + 1];
	deltaOutputSum = new double[outputNeuronNumber + 1];

	initializeInputHiddenLayer();

	initializeHiddenOutputLayer();
}

void initializeInputHiddenLayer()
{
	for (int i = 1; i <= inputNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			int sign = rand() % 2;
			weightsInputHidden[i][j] = (double)(rand() % 6) / 10.0;
			if (sign == 1) {
				weightsInputHidden[i][j] = -weightsInputHidden[i][j];
			}
		}
	}

	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			int sign = rand() % 2;
			weightsInputHidden2nd[i][j] = (double)(rand() % 6) / 10.0;
			if (sign == 1) {
				weightsInputHidden2nd[i][j] = -weightsInputHidden2nd[i][j];
			}
		}
	}
}

void initializeHiddenOutputLayer()
{
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= outputNeuronNumber; ++j) {
			int sign = rand() % 2;
			weightstHiddenOutput[i][j] = (double)(rand() % 10 + 1) / (10.0 * outputNeuronNumber);
			if (sign == 1) {
				weightstHiddenOutput[i][j] = -weightstHiddenOutput[i][j];
			}
		}
	}
}

char aboutTraining() {

	cout << "Train Neural Network " << folderName << endl;
	cout << "This neural Networ has " << inputNeuronNumber << " input neurons and " << hiddenNeuronNumber << " hidden neurons and " << outputNeuronNumber << " output neurons" << endl;
	cout << "We've got " << trainingDataSamplesCounter << " samples from :" << endl;
	cout << training_image_fn << endl;
	cout << training_label_fn << endl;
	cout << "Click 'q' to quit or anything else to quit" << endl;

	char enter = getchar();
	getchar();
	return enter;
}


void forwardProcedureTraining() {
	
	clearLayerInArray();

	for (int i = 1; i <= inputNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			hiddenLayerIn[j] += inputLayerOUT[i] * weightsInputHidden[i][j];
		}
	}

	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		hiddenLayerOUT[i] = sigmoid(hiddenLayerIn[i]);
	}
	//
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			hiddenLayerIn2nd[j] += hiddenLayerOUT[i] * weightsInputHidden2nd[i][j];
		}
	}

	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		hiddenLayerOUT2nd[i] = sigmoid(hiddenLayerIn2nd[i]);
	}
	//

	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= outputNeuronNumber; ++j) {
			outputLayerIN[j] += hiddenLayerOUT2nd[i] * weightstHiddenOutput[i][j];
		}
	}

	for (int i = 1; i <= outputNeuronNumber; ++i) {
		outputLayerOUT[i] = sigmoid(outputLayerIN[i]);
	}
}

void clearLayerInArray()
{
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		hiddenLayerIn[i] = 0.0;
	}
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		hiddenLayerIn2nd[i] = 0.0;
	}
	for (int i = 1; i <= outputNeuronNumber; ++i) {
		outputLayerIN[i] = 0.0;
	}
}

double squareErrorOfTraining() {
	double res = 0.0;
	for (int i = 1; i <= outputNeuronNumber; ++i) {
		res += (outputLayerOUT[i] - outputLayerResults[i]) * (outputLayerOUT[i] - outputLayerResults[i]);
	}
	res *= 0.5;
	return res;
}

void calculateDeltaSum()
{
	double sum;

	for (int i = 1; i <= outputNeuronNumber; ++i) {
		//double output_sum_margin_of_error = (outputLayerResults[i] - outputLayerOUT[i]);
		// derivative of sigmoid function = outputLayerOUT[i] * (1 - outputLayerOUT[i])
		// delta output sum = derivative of sigmoid function *  output_sum_margin_of_error
		deltaOutputSum[i] = outputLayerOUT[i] * (1 - outputLayerOUT[i]) * (outputLayerResults[i] - outputLayerOUT[i]);
	}

	//
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		//sum is hidden_sum_margin_of_error
		sum = 0.0;
		for (int j = 1; j <= outputNeuronNumber; ++j) {
			sum += weightstHiddenOutput[i][j] * deltaOutputSum[j];
		}
		deltaHiddenSum2nd[i] = hiddenLayerOUT2nd[i] * (1 - hiddenLayerOUT2nd[i]) * sum;
	}
	//

	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		//sum is hidden_sum_margin_of_error
		sum = 0.0;
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			sum += weightsInputHidden2nd[i][j] * deltaHiddenSum2nd[j];
		}
		deltaHiddenSum[i] = hiddenLayerOUT[i] * (1 - hiddenLayerOUT[i]) * sum;
	}

}

void backPropagation() {

	calculateDeltaSum();

	updateWeights();

}

void updateWeights()
{
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= outputNeuronNumber; ++j) {
			delta3[i][j] = (learning_rate * deltaOutputSum[j] * hiddenLayerOUT2nd[i]) + (momentum * delta3[i][j]);
			weightstHiddenOutput[i][j] += delta3[i][j];
		}
	}
	//

	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; j++) {
			delta2[i][j] = (learning_rate * deltaHiddenSum2nd[j] * hiddenLayerOUT[i]) + (momentum * delta2[i][j]);
			weightsInputHidden2nd[i][j] += delta2[i][j];
		}
	}
	//
	for (int i = 1; i <= inputNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; j++) {
			delta1[i][j] = (learning_rate * deltaHiddenSum[j] * inputLayerOUT[i]) + (momentum * delta1[i][j]);
			weightsInputHidden[i][j] += delta1[i][j];
		}
	}
}

int trainingProcess() {

	clearDeltaArrays();

	for (int i = 1; i <= epochs; ++i) {


		forwardProcedureTraining();


		backPropagation();


		if (squareErrorOfTraining() < epsilon) {
			return i;
		}
	}
	return epochs;
}

void clearDeltaArrays()
{
	for (int i = 1; i <= inputNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			delta1[i][j] = 0.0;
		}
	}

	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			delta2[i][j] = 0.0;
		}
	}

	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= outputNeuronNumber; ++j) {
			delta3[i][j] = 0.0;
		}
	}
}

void readImageTraining() {
	// Reading imageStream
	char number;
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			imageStream.read(&number, sizeof(char));
			if (number == 0) {
				imageMatrix[i][j] = 0;
			}
			else {
				imageMatrix[i][j] = 1;
			}
		}
	}

	cout << "Image:" << endl;
	//displaing
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			cout << imageMatrix[i][j];
		}
		cout << endl;
	}
	//saving
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			int pos = i + (j - 1) * width;
			inputLayerOUT[pos] = imageMatrix[i][j];
		}
	}

	// Reading label
	labelsStream.read(&number, sizeof(char));

	//write proper results 
	for (int i = 1; i <= outputNeuronNumber; ++i) {
		outputLayerResults[i] = 0.0;
	}
	outputLayerResults[number + 1] = 1.0;

	cout << "Label: " << (int)(number) << endl;
}


void saveNeuralNetworkModel(string file_name) {
	ofstream file(file_name.c_str(), ios::out);

	// Input layer - Hidden layer
	for (int i = 1; i <= inputNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			file << weightsInputHidden[i][j] << " ";
		}
		file << endl;
	}

	// Hidden layer - Hidden layer
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			file << weightsInputHidden2nd[i][j] << " ";
		}
		file << endl;
	}

	// Hidden layer - Output layer
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= outputNeuronNumber; ++j) {
			file << weightstHiddenOutput[i][j] << " ";
		}
		file << endl;
	}

	file.close();
}

//MAIN FUNCTION																										START THERE
int learnNeuralNetwork() {

	if (aboutTraining() == 'q') return 0;

	report.open(report_fn_train.c_str(), ios::out);
	imageStream.open(training_image_fn.c_str(), ios::in | ios::binary);
	labelsStream.open(training_label_fn.c_str(), ios::in | ios::binary);

	readUnimportantTrainHeaders();

	initializeTrainModule();

	training();

	saveNeuralNetworkModel(model_fn_train); //last saving

	report.close();
	imageStream.close();
	labelsStream.close();

	return 0;
}

void training()
{
	for (int sample = 1; sample <= trainingDataSamplesCounter; ++sample) {
		cout << "Sample number" << sample << endl;

		readImageTraining();

		// Learning process: Perceptron (Forward procedure) - Back propagation
		int iterationCounter = trainingProcess();

		cout << "Iteration number: " << iterationCounter << endl;
		printf("Error: %0.6lf\n\n", squareErrorOfTraining());
		//save raport with new training sample
		report << "(Sample number" << sample << ")  iteration number: " << iterationCounter << ", Error = " << squareErrorOfTraining() << endl;

		// Save the current network (weights)
		if (sample % 1000 == 0) {
			//cout << "Saving network is done" << endl;
			saveNeuralNetworkModel(model_fn_train);
			//cout << endl << endl<< "Click 'q' to quit training or anything else to contiune" << endl<<endl;

			//char enter = getchar();
			//getchar();
			//if (enter == 'q') {
			//	return;
			//}
		}
	}
}

void readUnimportantTrainHeaders()
{
	// Reading file headers
	char number;
	for (int i = 1; i <= 16; ++i) {
		imageStream.read(&number, sizeof(char));
	}
	for (int i = 1; i <= 8; ++i) {
		labelsStream.read(&number, sizeof(char));
	}
}

