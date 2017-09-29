#include "Network.h"
#include "DS_MNIST.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define WHOLE_DATA_PER_EPOCH

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

int main() {
	srand(time(NULL));

	nn::Network* network = nn::Network::Builder()
							.input(INPUT_SIZE)
							.addLayer(20)
							.addLayer(30)
							.addLayer(OUTPUT_SIZE)
							.build();

	nn::MNIST dataset("train.txt", "test.txt");

	std::vector<nn::DataEntry> train_set = dataset.get_train_set();
	std::cout << "Train set loaded, total " << train_set.size() << " entries." << std::endl;

	std::vector<nn::DataEntry> test_set = dataset.get_test_set();
	std::cout << "Test set loaded, total " << test_set.size() << " entries." << std::endl;

	std::cout << "Data load complete. Starting training phase..." << std::endl << std::endl;

	const int total_size = train_set.size();

	int epoch = 1;

#ifdef WHOLE_DATA_PER_EPOCH
	for(int start = epoch; epoch <= start + 100; epoch++) {
#else
	const int batch_size = total_size / 50;
	int batch_start = 0;
	for (int start = epoch; epoch <= start + 1000; epoch++) {
#endif

		std::random_shuffle(train_set.begin(), train_set.end());
		std::cout << "Training epoch #" << epoch << "..." << std::endl;
		int begin = time(NULL);

#ifdef WHOLE_DATA_PER_EPOCH
		network->train(total_size, &train_set[0], 0.003);
#else
		network->train(batch_size, &train_set[batch_start]);
		batch_start = (batch_start + batch_size) % total_size;
#endif

		int end = time(NULL);
		std::cout << "Finished in " << end - begin << "s, calculating error..." << std::endl;

		double sq_error = 0;
		int error_count = 0;
		for (auto iter = train_set.begin(); iter != train_set.end(); ++iter) {
			double* result = network->predict(iter->data);
			for (int i = 0; i < OUTPUT_SIZE; i++) {
				double error = result[i] - iter->label[i];
				sq_error += error * error;
				error_count++;
			}
		}
		std::cout << "Done, MSE:" << sq_error / error_count << std::endl << std::endl;

	}

	int count = 0;
	int correct = 0;
	for (auto iter = test_set.begin(); iter != test_set.end(); ++iter) {
		double* data = iter->data;
		double* label = iter->label;

		double* output = network->predict(data);

		double max_ans = 0, max_res = 0;
		int i_ans = 0, i_res = 0;
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			if (label[i] > max_ans) {
				max_ans = label[i];
				i_ans = i;
			}
			if (output[i] > max_res) {
				max_res = output[i];
				i_res = i;
			}
		}
		if (i_ans == i_res) {
			correct++;
		}
		count++;
	}
	std::cout << "Test data accuracy: " << (double)correct / count
		<< " (" << correct << " / " << count << " correct)" << std::endl;
	return 0;
}
