#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include "Dataset.h"
#include <cstdio>

namespace nn {
	class MNIST_bin : public Dataset {
	public:
		static const int INPUTS = 784, OUTPUTS = 10;

		MNIST_bin(const char* train_file, const char* test_file)
			: train(train_file), test(test_file)
		{}

		std::vector<DataEntry> get_train_set() override {
			FILE* mnist_train = fopen(train, "rb");
			std::vector<DataEntry> dataset;

			mnist_entry item;
			while (fread(&item, sizeof(item), 1, mnist_train) > 0) {
				DataEntry entry(INPUTS, OUTPUTS);

				for (int i = 0; i < OUTPUTS; i++) {
					entry.label[i] = (item.label == i) ? 1 : 0;
				}

				for (int i = 0; i < INPUTS; i++) {
					entry.data[i] = item.data[i] / 255.0;
				}

				dataset.push_back(std::move(entry));
			}
			return dataset;
		}

		std::vector<DataEntry> get_test_set() override {
			FILE* mnist_test = fopen(test, "rb");
			std::vector<DataEntry> dataset;

			mnist_entry item;
			while (fread(&item, sizeof(item), 1, mnist_test) > 0) {
				DataEntry entry(INPUTS, OUTPUTS);

				for (int i = 0; i < OUTPUTS; i++) {
					entry.label[i] = (item.label == i) ? 1 : 0;
				}

				for (int i = 0; i < INPUTS; i++) {
					entry.data[i] = item.data[i] / 255.0;
				}

				dataset.push_back(std::move(entry));
			}
			return dataset;
		}
	private:
		struct mnist_entry {
			int label;
			unsigned char data[INPUTS];
		};
		const char *train, *test;
	};
}
