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

#define EPOCH_SAMPLES 100
//#define PRINT_TRAIN_ERROR

bool hasOption(char** begin, char** end, const std::string& option) {
	return std::find(begin, end, option) != end;
}

char* getOptionValue(char** begin, char** end, const std::string& option) {
	char** iter = std::find(begin, end, option);
	if (iter != end && ++iter != end && **iter != '-') {
		return *iter;
	} else {
		return NULL;
	}
}

int main(int argc, char* argv[]) {
	if (hasOption(argv, argv + argc, "-h")) {
		std::cout	<< "============================ Neural Network Trainer - Usage ============================" << std::endl
					<< " Train Mode: MNIST_NN [(-c {Checkpoint file} -e {Epoch count} | [-h1 {Neurons in 1st hidden layer}] [-h2 {Neurons in 2nd hidden layer}])] [-t {Threshold value}]" << std::endl
					<< "  > The program reads two files, train.bin and test.bin, and starts training until MSE reaches the threshold" << std::endl
					<< "  > threshold defaults to 0.001, h1 defaults to 128, h2 defaults to 64" << std::endl
					<< " Run Mode: MNIST_NN -r -c {Checkpoint file}" << std::endl
					<< "  > Input 784 integers in range 0~255 through standard input to get the predicted number. Program ends on EOF." << std::endl
					<< "========================================================================================" << std::endl;
		return 0;
	}

	char* checkpoint = getOptionValue(argv, argv + argc, "-c");

	if (hasOption(argv, argv + argc, "-r")) {
		if (!checkpoint) {
			std::cout << "In the run mode, you must specify a weights file(.ckpt) with -c option." << std::endl;
			return -1;
		}
		std::ifstream is(checkpoint, std::ios::binary);
		if (is.fail()) {
			std::cout << "Cannot open checkpoint file, " << checkpoint << std::endl;
			return -2;
		}
		nn::Network* network = nn::Network::Builder().load(is).build();
		is.close();

		double input[784];
		while (true) {
			for (int i = 0; i < 784; i++) {
				if (!(std::cin >> input[i])) {
					return 0;
				}
				input[i] /= 255;
			}

			auto result = network->predict(input);
			double max = 0;
			int maxi = -1;
			for (int i = 0; i < 10; i++) {
				if (result[i] > max) {
					max = result[i];
					maxi = i;
				}
			}
			std::cout << maxi << std::endl;
		}
	} else {
		srand(time(NULL));

		nn::Network* network;
		int epoch;
		if (checkpoint) {
			char* epoch_s = getOptionValue(argv, argv + argc, "-e");
			if (!epoch_s) {
				std::cout << "If you specify checkpoint file, you also have to specify the epoch count with -e option." << std::
					endl;
				return -3;
			}
			epoch = strtoul(epoch_s, NULL, 10);
			if (epoch == 0) {
				std::cout << "Invalid epoch value: " << epoch_s << std::endl;
				return -4;
			}

			std::ifstream is(checkpoint, std::ios::binary);
			if (is.fail()) {
				std::cout << "Cannot open checkpoint file, " << checkpoint << std::endl;
				return -2;
			}
			network = nn::Network::Builder().load(is).build();
			is.close();
		} else {
			char* h_s = getOptionValue(argv, argv + argc, "-h1");
			int h1 = 128;
			if(h_s) {
				h1 = strtoul(h_s, NULL, 10);
				if(h1 == 0) {
					std::cout << "Invalid 1st hidden layer neuron count: " << h1 << std::endl;
					return -5;
				}
			}

			h_s = getOptionValue(argv, argv + argc, "-h2");
			int h2 = 64;
			if (h_s) {
				h2 = strtoul(h_s, NULL, 10);
				if (h2 == 0) {
					std::cout << "Invalid 2nd hidden layer neuron count: " << h2 << std::endl;
					return -6;
				}
			}
			
			epoch = 0;
			network = nn::Network::Builder()
				.input(784)
				.addLayer<nn::activation::Sigmoid>(h1)
				.addLayer<nn::activation::Sigmoid>(h2)
				.addLayer<nn::activation::Sigmoid>(10)
				.build();
		}

		double threshold;
		if (hasOption(argv, argv + argc, "-t")) {
			char* threshold_s = getOptionValue(argv, argv + argc, "-t");
			if (threshold_s) {
				threshold = strtod(threshold_s, NULL);
				if(threshold <= 0.0) {
					std::cout << "Invalid threshold value: " << threshold_s << std::endl;
					return -6;
				}
			} else {
				std::cout << "No threshold value specified with -t parameter!" << std::endl;
				return -7;
			}
		} else {
			threshold = 0.001;
		}

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

		double mse;

		double sq_error = 0;
		int error_count = 0;
		int correct_count = 0;

		for (auto iter = test_set.begin(); iter != test_set.end(); ++iter) {
			auto result = network->predict(iter->data);
			double lmax = 0, rmax = 0;
			double li = -1, ri = -1;
			for (int i = 0; i < dataset.OUTPUTS; i++) {
				if (iter->label[i] > lmax) {
					lmax = iter->label[i];
					li = i;
				}
				if (result[i] > rmax) {
					rmax = result[i];
					ri = i;
				}

				double error = result[i] - iter->label[i];
				sq_error += error * error;
				error_count++;
			}
			if (li == ri) correct_count++;
		}
		mse = sq_error / error_count;
		std::cout << "Before start, Test set MSE: " << mse << ", Accuracy: " << correct_count * 100.0 / test_set.size() << '%' << std::endl;

#ifndef EPOCH_SAMPLES
		const int batch_size = train_set.size();
		for (int start = ++epoch; ; epoch++) {
#else
		const int batch_size = EPOCH_SAMPLES;
		for (int start = ++epoch; ; epoch++) {
#endif

			std::random_shuffle(train_set.begin(), train_set.end());

			// start position doesn't matter; train_set is randomly shuffled every epoch.
			network->train(batch_size, &train_set[0]);

#ifdef EPOCH_SAMPLES
			if (epoch % 100 == 0) {
#endif
				std::cout << "Epoch #" << epoch << " finished,";

#ifdef PRINT_TRAIN_ERROR
				{
					sq_error = 0;
					error_count = 0;
					correct_count = 0;
					for (auto iter = train_set.begin(); iter != train_set.end(); ++iter) {
						auto result = network->predict(iter->data);
						double lmax = 0, rmax = 0;
						double li = -1, ri = -1;
						for (int i = 0; i < dataset.OUTPUTS; i++) {
							if (iter->label[i] > lmax) {
								lmax = iter->label[i];
								li = i;
							}
							if (result[i] > rmax) {
								rmax = result[i];
								ri = i;
							}

							double error = result[i] - iter->label[i];
							sq_error += error * error;
							error_count++;
						}
						if (li == ri) correct_count++;
					}
					mse = sq_error / error_count;
					std::cout << "\tTrain: MSE: " << mse << ",\tAcc: " << correct_count * 100.0 / train_set.size() << "%,";
				}
#endif

				sq_error = 0;
				error_count = 0;
				correct_count = 0;
				for (auto iter = test_set.begin(); iter != test_set.end(); ++iter) {
					auto result = network->predict(iter->data);
					double lmax = 0, rmax = 0;
					double li = -1, ri = -1;
					for (int i = 0; i < dataset.OUTPUTS; i++) {
						if (iter->label[i] > lmax) {
							lmax = iter->label[i];
							li = i;
						}
						if (result[i] > rmax) {
							rmax = result[i];
							ri = i;
						}

						double error = result[i] - iter->label[i];
						sq_error += error * error;
						error_count++;
					}
					if (li == ri) correct_count++;
				}
				mse = sq_error / error_count;
				std::cout  << "\tTest: MSE: " << mse << ",\tAcc: " << correct_count * 100.0 / test_set.size() << '%' << std::endl;
#ifdef EPOCH_SAMPLES
			}
#endif

#ifndef EPOCH_SAMPLES
			if (epoch % 10 == 0) {
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

			if (mse < threshold) {
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
	}
	return 0;
}
