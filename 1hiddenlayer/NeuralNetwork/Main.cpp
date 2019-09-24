#include <iostream>
#include <filesystem>
#include <string>
#include "testModule.h"
#include "trainModule.h"
#include "const.h"


using namespace std;

void changeModel(char model) {
}


char userChoice() {
	fflush(stdin);
	cout << "Neural Network program - WELCOME" << endl;
	cout << "If you want to: " << endl;
	cout << "1. Learn -> click 'l'" << endl;
	cout << "2. Train -> click 't' " << endl;
	cout << "Otherwise click 'q' and quit " << endl;
	return getchar();

}

int main()
{
	char sign = '1';
	while ((sign = userChoice()) != 'q') {
		getchar();
		if (sign == 'l') {
			learnNeuralNetwork();
		}
		if (sign == 't') {
			testNeuralNetwork();
		}
	}
	return 0;
}

