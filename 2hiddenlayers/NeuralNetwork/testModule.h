#pragma once
#include <string>

using namespace std;

int testNeuralNetwork();

void testing(int& correctResultCounter);

void testFailInfo(int label, int predict, int sample);

void testPassInfo(int& correctResultCounter, int label, int predict, int sample);

void readFileUnusedHeaders();

void summary(int nCorrect);
