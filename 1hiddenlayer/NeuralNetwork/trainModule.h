#pragma once
#include <string>


using namespace std;

int learnNeuralNetwork( );

void clearDeltaArrays();

void calculateDeltaSum();

void clearLayerInArray();

void updateWeights();

void training();

void readUnimportantTrainHeaders();

void initializeHiddenOutputLayer();

void initializeInputHiddenLayer();
