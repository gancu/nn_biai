#pragma once

#include <string>
#include <fstream>

using namespace std;

static string folderName = "mnist2";
//C:\Biai\2lr
const static string relativePath = "E:/2hiddenlayers/";								//CHANGE !

// Weights file name
const string model_fn_train = relativePath + "debug/model-neural-network-" + folderName + ".dat";

// Report file name
const string report_fn_train = relativePath + "debug/training-report-"+ folderName + ".dat";

// Report file name
const string report_fn_test = relativePath + "debug/testing-report-" + folderName + ".dat";

// Training imageStream file name
const string training_image_fn = relativePath + "debug/" + folderName + "/train-images.idx3-ubyte";

// Training labelsStream file name
const string training_label_fn = relativePath + "debug/" + folderName + "/train-labels.idx1-ubyte";

// Testing imageStream file name
const string testing_image_fn = relativePath + "debug/" + folderName + "/t10k-images.idx3-ubyte";

// Testing labelsStream file name
const string testing_label_fn = relativePath + "debug/" + folderName + "/t10k-labels.idx1-ubyte";


// Number of testing samples
const int testingSamplesCounter = 10000;
const int width = 28;
const int height = 28;
const int inputNeuronNumber = width * height;
const int hiddenNeuronNumber = 128;
const int outputNeuronNumber = 10;
const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;
const int trainingDataSamplesCounter = 60000;


static ifstream imageStream;
static ifstream labelsStream;
static ofstream report;


// Image: 28x28 gray scale images.
int static imageMatrix[width + 1][height + 1];

// Input layer - Hidden layer 1st
static double* weightsInputHidden[inputNeuronNumber + 1], * inputLayerOUT;

// Hidden layer 1st - Hidden layer 2nd
static double* weightsInputHidden2nd[hiddenNeuronNumber + 1], * hiddenLayerIn, * hiddenLayerOUT;

//  Hidden layer 2nd - Odput layer
static double* weightstHiddenOutput[hiddenNeuronNumber + 1], * hiddenLayerIn2nd, * hiddenLayerOUT2nd;


// Output layer
static double* outputLayerIN, * outputLayerOUT;

static double* deltaHiddenSum, * deltaOutputSum, * delta1[inputNeuronNumber + 1], * delta2[hiddenNeuronNumber + 1];

static double* deltaHiddenSum2nd,* delta3[hiddenNeuronNumber + 1];


static double outputLayerResults[outputNeuronNumber + 1];

static double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

