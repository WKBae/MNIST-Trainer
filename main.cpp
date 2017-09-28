#include "Network.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
	srand(time(NULL));

	nn::Network* network = nn::Network::Builder()
							.input(784)
							.addLayer(256)
							.addLayer(10)
							.build();

	std::vector<double*> labels;
	std::vector<double*> datas;

	int label;

#ifdef STREAM_INPUT
	std::ifstream mnist_train("train.txt");
	while(mnist_train >> label) {
		double* one_hot = new double[10];
		for(int i = 0; i < 10; i++) {
			one_hot[i] = (label == i)? 1 : 0;
		}

		double* input = new double[784];
		for(int i = 0; i < 784; i++) {
			mnist_train >> input[i];
		}

		labels.push_back(one_hot);
		datas.push_back(input);
		std::cout << "Read data #" << datas.size() << std::endl;
	}
#else
	FILE* mnist_train = fopen("train.txt", "r");
	while(!feof(mnist_train)) {
		if(fscanf(mnist_train, "%d", &label) <= 0) break;

		double* one_hot = new double[10];
		for(int i = 0; i < 10; i++) {
			one_hot[i] = (label == i)? 1 : 0;
		}

		double* input = new double[784];
		for(int i = 0; i < 784; i++) {
			fscanf(mnist_train, "%d", &input[i]);
		}

		labels.push_back(one_hot);
		datas.push_back(input);
		//std::cout << "Read data #" << datas.size() << std::endl;
	}
#endif

	std::cout << "Data load complete, total " << datas.size() << " data loaded. "
				<< "Starting training phase..." << std::endl;

	int epoch = 1;
	for(; epoch <= 100; epoch++) {
		network->train(datas.size(), &datas[0], &labels[0]);

		std::cout << "Training epoch #" << epoch << " finished, now testing..." << std::endl;

		std::ifstream mnist_test("test.txt");
		int count = 0, correct = 0;
		while(mnist_test >> label) {
			double* input = new double[784];
			for(int i = 0; i < 784; i++) {
				mnist_test >> input[i];
			}
			
			double* result = network->predict(input);
			double max = 0;
			int maxi = 0;
			for(int i  = 0; i < 10; i++) {
				if(result[i] > max) {
					max = result[i];
					maxi = i;
				}
			}
			if(label == maxi) {
				correct++;
			}
			count++;
		}
		std::cout << "Test data accuracy: " << (double) correct / count
					<< " (" << correct << " / " << count << " correct)" << std::endl;
	}
	return 0;
}
