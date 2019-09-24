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
#include "const.h"
#include "testModule.h"

using namespace std;

int labels[10];
int positive_labels[10];

double square_error() {
	double res = 0.0;
	for (int i = 1; i <= outputNeuronNumber; ++i) {
		res += (outputLayerOUT[i] - outputLayerResults[i]) * (outputLayerOUT[i] - outputLayerResults[i]);
	}
	res *= 0.5;
	return res;
}


void testFailInfo(int label, int predict, int sample)
{
	cout << "TEST FAILED!" << endl;
	cout << "Test result = " << predict << " , correct: " << predict << endl << endl;
	cout << "Image:" << endl;
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			cout << imageMatrix[i][j];
		}
		cout << endl;
	}
	cout << "Image that failed in test presented above " << endl;
	cout << endl << endl;

	report << "(Sample number" << sample << ")	FALSE		, answer: " << predict << ", correct answer: " << label << ". Error was " << square_error() << endl;
}

void testPassInfo(int& correctResultCounter, int label, int predict, int sample)
{
	++correctResultCounter;
	cout << "TEST PASSED !" << endl;
	cout << "Test result = " << predict << " , correct: " << label << endl << endl;
	report << "(Sample number" << sample << ")	TRUE		, answer: " << predict << ", correct answer: " << label << ". Error was " << square_error() << endl;
}

void readFileUnusedHeaders()
{
	char number;
	for (int i = 1; i <= 16; ++i) {
		imageStream.read(&number, sizeof(char));
	}
	for (int i = 1; i <= 8; ++i) {
		labelsStream.read(&number, sizeof(char));
	}
}

char about() {

	cout << "Testing Neural Network" << endl;
	cout << "This neural Networ has " << inputNeuronNumber << " input neurons and " << hiddenNeuronNumber << " hidden neurons and " << outputNeuronNumber << "output neurons" << endl;
	cout << "We've got " << testingSamplesCounter << " samples from :" << endl;
	cout << testing_image_fn << endl;
	cout << testing_label_fn << endl;
	cout << "Click 'q' to quit or anything else to quit" << endl;

	char enter = getchar();
	getchar();
	return enter;
}


void memoryAllocation() {
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
}


void load_model(string file_name) {
	ifstream file(file_name.c_str(), ios::in);

	// Input layer - Hidden layer
	for (int i = 1; i <= inputNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			file >> weightsInputHidden[i][j];
		}
	}

	// Hidden layer - Hidden layer
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= hiddenNeuronNumber; ++j) {
			file >> weightsInputHidden2nd[i][j];
		}
	}

	// Hidden layer - Output layer
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		for (int j = 1; j <= outputNeuronNumber; ++j) {
			file >> weightstHiddenOutput[i][j];
		}
	}

	file.close();
}


void forwardProcedureTesting() {
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		hiddenLayerIn[i] = 0.0;
	}

	for (int i = 1; i <= outputNeuronNumber; ++i) {
		outputLayerIN[i] = 0.0;
	}
	for (int i = 1; i <= hiddenNeuronNumber; ++i) {
		hiddenLayerIn2nd[i] = 0.0;
	}

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


int readImageTest() {
	// Reading image
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

	//writing
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			int pos = i + (j - 1) * width;
			inputLayerOUT[pos] = imageMatrix[i][j];
		}
	}

	// Reading label
	labelsStream.read(&number, sizeof(char));
	for (int i = 1; i <= outputNeuronNumber; ++i) {
		outputLayerResults[i] = 0.0;
	}
	outputLayerResults[number + 1] = 1.0;

	return (int)(number);
}


void summary(int nCorrect)
{
	double accuracy = (double)(nCorrect) / testingSamplesCounter * 100.0;
	cout << "Number of correct samples: " << nCorrect << " / " << testingSamplesCounter << endl;
	printf("Accuracy: %0.2lf\n", accuracy);

	report << "Number of correct samples: " << nCorrect << " / " << testingSamplesCounter << endl;
	report << "Accuracy: " << accuracy << endl;

	double result_percent;
	for (int i = 0; i < 10; i++) {
		report << "Label: " << i << " Occurance" << labels[i] << " ";
		report << " Postive predictions: " << positive_labels[i] << "	";
		if (positive_labels[i] > 0) {
			result_percent = (double(positive_labels[i]) / double(labels[i])) * 100.0;
			report << result_percent << "% " << endl;
		}
		else
		{
			report << "0%" << endl;
		}
	}
}

//Main function																	START HERE

int testNeuralNetwork( ) {
	if (about() == 'q') return 0;

	for (int i = 0; i < 10; i++) {
		labels[i] = 0;
		positive_labels[i] = 0;
	}

	report.open(report_fn_test.c_str(), ios::out);
	imageStream.open(testing_image_fn.c_str(), ios::in | ios::binary);
	labelsStream.open(testing_label_fn.c_str(), ios::in | ios::binary);

	readFileUnusedHeaders();

	memoryAllocation();

	load_model(model_fn_train);

	int correctResultCounter = 0;

	testing(correctResultCounter);

	summary(correctResultCounter);

	report.close();
	imageStream.close();
	labelsStream.close();

	return 0;
}

void testing(int& correctResultCounter)
{
	for (int sample = 1; sample <= testingSamplesCounter; ++sample) {
		cout << "Sample " << sample << endl;

		// Get label
		int label = readImageTest();
		labels[label] = labels[label] + 1;

		// Classification - Perceptron procedure
		forwardProcedureTesting();

		// Prediction
		int predict = 1;
		for (int i = 2; i <= outputNeuronNumber; ++i) {
			//check predicition
			if (outputLayerOUT[i] > outputLayerOUT[predict]) {
				predict = i;
			}
		}
		--predict;

		printf("Error: %0.6lf\n", square_error());

		if (label == predict) {
			testPassInfo(correctResultCounter, label, predict, sample);
			positive_labels[label] = positive_labels[label] + 1;
		}
		else {
			testFailInfo(label, predict, sample);
		}
	}
}
