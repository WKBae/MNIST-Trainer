#pragma once

#include "Dataset.h"

namespace nn {
	class THREE : public Dataset {
	public:
		static const int INPUTS = 64, OUTPUTS = 3;

		THREE(const char* train_file, const char* test_file)
			: train(train_file), test(test_file)
		{}

		std::vector<DataEntry> get_train_set() {
			int label;
			FILE* three_train = fopen(train, "r");
			std::vector<DataEntry> dataset;
			while (fscanf(three_train, "%d $", &label) > 0) {
				DataEntry entry(INPUTS, OUTPUTS);

				for (int i = 0; i < OUTPUTS; i++) {
					entry.label[i] = (label == i) ? 1 : 0;
				}

				for (int i = 0; i < INPUTS; i++) {
					fscanf(three_train, "%lf", &entry.data[i]);
				}
				fscanf(three_train, "%*d");

				dataset.push_back(std::move(entry));
			}
			return dataset;
		}

		std::vector<DataEntry> get_test_set() {
			int label;
			FILE* three_test = fopen(test, "r");
			std::vector<DataEntry> dataset;
			while (fscanf(three_test, "%d $", &label) > 0) {
				DataEntry entry(INPUTS, OUTPUTS);

				for (int i = 0; i < OUTPUTS; i++) {
					entry.label[i] = (label == i) ? 1 : 0;
				}

				for (int i = 0; i < INPUTS; i++) {
					fscanf(three_test, "%lf", &entry.data[i]);
				}
				fscanf(three_test, "%*d");

				dataset.push_back(std::move(entry));
			}
			return dataset;
		}
	private:
		const char *train, *test;
	};
}