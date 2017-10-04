#include "Network.h"
//#include "THREE.h"
//#include "MNIST.h"
#include "MNIST_bin.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <algorithm>

//#define WHOLE_DATA_PER_EPOCH

int main() {
	srand(time(NULL));

	std::cout << "Loading data set..." << std::endl;

	//nn::THREE dataset("traindata.txt", "testdata.txt");
	//nn::MNIST dataset("train.txt", "test.txt");
	nn::MNIST_bin dataset("train.bin", "test.bin");

	std::vector<nn::DataEntry> train_set, test_set;
	#pragma omp parallel
	{
		#pragma omp single
		{
			train_set = dataset.get_train_set();
			#pragma omp critical
			std::cout << "Train set loaded, total " << train_set.size() << " entries." << std::endl;
		}
		#pragma omp single
		{
			test_set = dataset.get_test_set();
			#pragma omp critical
			std::cout << "Test set loaded, total " << test_set.size() << " entries." << std::endl;
		}
	}
	std::cout << "Data load complete. Starting training phase..." << std::endl << std::endl;


	nn::Network* network = nn::Network::Builder()
		.input(dataset.INPUTS)
		.addLayer(32)
		.addLayer(64)
		.addLayer(dataset.OUTPUTS)
		//.load(std::ifstream("./ckpt/5.ckpt", std::ios::binary))
		.build();


	int epoch = 154000;
	double mse = 1e100;
#ifdef WHOLE_DATA_PER_EPOCH
	const int batch_size = train_set.size();
	for(int start = ++epoch; ; epoch++) {
#else
	const int batch_size = 100;
	for (int start = ++epoch; ; epoch++) {
#endif

		std::random_shuffle(train_set.begin(), train_set.end());

		// start position doesn't matter; train_set is randomly shuffled every epoch.
		network->train(batch_size, &train_set[0]);

#ifndef WHOLE_DATA_PER_EPOCH
		if (epoch % 100 == 0) {
#endif
			std::cout << "Epoch #" << epoch <<  " finished, MSE: ";

			double sq_error = 0;
			int error_count = 0;
			int correct_num = 0;
			for (auto iter = test_set.begin(); iter != test_set.end(); ++iter) {
				auto result = network->predict(iter->data);
				double lmax = 0, rmax = 0;
				double li = -1, ri = -1;
				for (int i = 0; i < dataset.OUTPUTS; i++) {
					if (iter->label[i] > lmax) {
						lmax = iter->label[i];
						li = i;
					}
					if(result[i] > rmax) {
						rmax = result[i];
						ri = i;
					}

					double error = result[i] - iter->label[i];
					sq_error += error * error;
					error_count++;
				}
				if(li == ri) correct_num++;
			}
			mse = sq_error / error_count;
			std::cout << mse << ", Accuracy: " << correct_num * 100.0 / test_set.size() << '%' << std::endl;
#ifndef WHOLE_DATA_PER_EPOCH
		}
#endif

#ifdef WHOLE_DATA_PER_EPOCH
		if (epoch % 5 == 0) {
#else
		if (epoch % 2000 == 0) {
#endif
			char ckptfile[100];
			sprintf(ckptfile, "./ckpt/%d.ckpt", epoch);

			std::cout << std::endl << "[Checkpoint reached] Saving to \"" << ckptfile << "\"..." << std::endl;

			std::ofstream ckpt(ckptfile, std::ios::binary);
			network->dump_network(ckpt);
			ckpt.flush();
			ckpt.close();

			std::cout << "Save complete." << std::endl << std::endl;
		}

		if (mse < 0.001) {
			std::cout << "MSE reached the threshold, run more epoches?(Y/n) ";
			char ans;
			std::cin >> ans;
			if (ans == 'N' || ans == 'n')
				break;

			std::cout << std::endl;
		}
	}

	int count = 0;
	int correct = 0;
	for (auto iter = test_set.begin(); iter != test_set.end(); ++iter) {
		auto data = iter->data;
		auto label = iter->label;

		auto output = network->predict(data);

		double max_ans = 0, max_res = 0;
		int i_ans = 0, i_res = 0;
		for (int i = 0; i < dataset.OUTPUTS; i++) {
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
