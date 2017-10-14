#pragma once
#include "Config.h"
#include <vector>

namespace nn {
	struct DataEntry {
		NUM_TYPE* data;
		int data_count;

		NUM_TYPE* label;
		int label_count;

		DataEntry() : data(NULL), data_count(0), label(NULL), label_count(0) {}

		DataEntry(int data_size, int label_size)
			: data(new NUM_TYPE[data_size]), data_count(data_size), label(new NUM_TYPE[label_size]), label_count(label_size)
		{}
		DataEntry(int data_size, NUM_TYPE* data, int label_size, NUM_TYPE* label)
			: data(new NUM_TYPE[data_size]), data_count(data_size), label(new NUM_TYPE[label_size]), label_count(label_size)
		{
			for (int i = 0; i < data_count; i++)
				this->data[i] = data[i];
			for (int i = 0; i < label_count; i++)
				this->label[i] = label[i];
		}

		DataEntry(DataEntry& other)
			: data(new NUM_TYPE[other.data_count]), data_count(other.data_count), label(new NUM_TYPE[other.label_count]), label_count(other.label_count)
		{
			for (int i = 0; i < data_count; i++)
				data[i] = other.data[i];
			for (int i = 0; i < label_count; i++)
				label[i] = other.label[i];
		}
		DataEntry(DataEntry&& other)
			: data(other.data), data_count(other.data_count), label(other.label), label_count(other.label_count)
		{
			other.data = NULL;
			other.label = NULL;
		}

		DataEntry& operator=(DataEntry&& other) {
			data = other.data;
			data_count = other.data_count;
			label = other.label;
			label_count = other.label_count;

			other.data = NULL;
			other.label = NULL;

			return *this;
		}

		~DataEntry() {
			delete[] data;
			delete[] label;
		}

	};

	class Dataset {
	public:
		virtual ~Dataset() {}

		virtual std::vector<DataEntry> get_train_set() = 0;
		virtual std::vector<DataEntry> get_test_set() = 0;
	};
}