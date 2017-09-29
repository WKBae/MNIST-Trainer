#pragma once

#include "Dataset.h"

namespace nn {
	class MNIST : public Dataset {
	public:
		static const int INPUTS = 784, OUTPUTS = 10;

		MNIST(const char* train_file, const char* test_file)
			: train(train_file), test(test_file)
		{}

		std::vector<DataEntry> get_train_set() {
			int label;
			FILE* mnist_train = fopen(train, "r");
			std::vector<DataEntry> dataset;
			while (fscanf(mnist_train, "%d", &label) > 0) {
				DataEntry entry(INPUTS, OUTPUTS);

				for (int i = 0; i < OUTPUTS; i++) {
					entry.label[i] = (label == i) ? 1 : 0;
				}

				for (int i = 0; i < INPUTS; i++) {
					fscanf(mnist_train, "%f", &entry.data[i]);
					entry.data[i] /= 255.0;
				}

				dataset.push_back(std::move(entry));
			}
			return dataset;
		}

		std::vector<DataEntry> get_test_set() {
			int label;
			FILE* mnist_test = fopen(test, "r");
			std::vector<DataEntry> dataset;
			while (fscanf(mnist_test, "%d", &label) > 0) {
				DataEntry entry(INPUTS, OUTPUTS);

				for (int i = 0; i < OUTPUTS; i++) {
					entry.label[i] = (label == i) ? 1 : 0;
				}

				for (int i = 0; i < INPUTS; i++) {
					fscanf(mnist_test, "%f", &entry.data[i]);
					entry.data[i] /= 255.0;
				}

				dataset.push_back(std::move(entry));
			}
			return dataset;
		}
	private:
		const char *train, *test;
	};
}