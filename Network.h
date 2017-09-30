#pragma once

/**
 * Defines layer and network data types for the neural network.
 * Note that the result arrays returned(NUM_TYPE* type return values) must not be modified.
 **/

#include "Common.h"
#include "Layer.h"
#include "Dataset.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace nn {

	/**
	 * The neural network.
	 * Composed of the layers, this class contains the operation for them including train and test(predict).
	 */
	class Network {
	public:
		class Builder {
		public:
			Builder& input(unsigned int input_size) {
				delete_list();
				this->input_size = input_size;

				return *this;
			}
			Builder& addLayer(unsigned int neurons) {
				unsigned int last_size;
				if(tail) {
					last_size = tail->output_size;
				} else {
					last_size = input_size;
				}
				if(last_size == 0 || neurons == 0) {
					throw std::invalid_argument("Neuron count cannot be zero, maybe you missed the call to Builder::input()");
				}

				Layer* layer = new LayerImpl<Sigmoid>(last_size, neurons);
				layer->initialize_weights();

				LayerList* list = new LayerList;
				list->layer = layer;
				list->output_size = neurons;
				list->next = NULL;

				if(tail) {
					tail->next = list;
					tail = list;
				} else {
					head = tail = list;
				}
				count++;

				return *this;
			}
			Network* build() {
				Layer** layers = new Layer*[count];
				LayerList* curr = head;
				for(unsigned int i = 0; i < count && curr != NULL; i++, curr = curr->next) {
					layers[i] = curr->layer;
				}

				Network* net = new Network(count, layers, input_size, tail->output_size);
				//delete this;
				return net;
			}

			Builder& load(std::istream& const input) {
				char magic[6];
				input.read(magic, 5);
				magic[5] = '\0';
				if (input.fail() || strcmp(magic, "NeNet") != 0)
					throw std::invalid_argument("The input is not a network save file");

				unsigned int layers;
				input.read((char*) &layers, sizeof(layers));

				NUM_TYPE* weight_buf = NULL;
				int buf_size = -1;

				for (int i = 0; i < layers; i++) {
					bool fail = false;

					int in, out;
					input.read((char*) &in, sizeof(in));
					fail = fail || input.fail();
					input.read((char*) &out, sizeof(out));
					fail = fail || input.fail();

					int count;
					input.read((char*) &count, sizeof(count));
					fail = fail || input.fail();

					if (fail) {
						throw std::out_of_range("File too short");
					}
					if (count > buf_size) {
						NUM_TYPE* newbuf = new NUM_TYPE[count];
						delete[] weight_buf;
						weight_buf = newbuf;
						buf_size = count;
					}

					input.read((char*) weight_buf, sizeof(NUM_TYPE) * count);
					if (input.fail()) {
						throw std::length_error("File size less than expected");
					}

					Layer* layer = new LayerImpl<Sigmoid>(in, out);
					layer->load_weights(weight_buf, count);

					LayerList* list = new LayerList;
					list->layer = layer;
					list->output_size = out;
					list->next = NULL;

					if (tail) {
						if (tail->output_size != in) {
							throw std::length_error("Layer output-input size mismatch");
						}
						tail->next = list;
						tail = list;
					} else {
						input_size = in;
						head = tail = list;
					}
					count++;
				}

				delete[] weight_buf;

				return *this;
			}

			Builder() : head(NULL), tail(NULL), input_size(0), count(0) {}

			~Builder() {
				delete_list();
			}
		private:
			struct LayerList {
				Layer* layer;
				unsigned int output_size;
				LayerList* next;
			} *head, *tail;

			unsigned int input_size;
			unsigned int count;

			void delete_list() {
				if(!head) return;

				for(LayerList *curr = head; curr != NULL;) {
					LayerList* next = curr->next;
					delete curr;
					curr = next;
				}
				
				head = tail = NULL;
				count = 0;
			}
		};

		void train(unsigned int n, DataEntry* data, NUM_TYPE learning_rate = 0.005) {
			NUM_TYPE** results = new NUM_TYPE*[layer_count + 1];
			NUM_TYPE* orig_delta = new NUM_TYPE[outputs];

			for(int i = 0; i < n; i++) {
				if (data[i].data_count != inputs || data[i].label_count != outputs)
					throw std::length_error("Data Entry size does not match with the network!");

				/* Retrieve the result(f = output) of the layers */
				results[0] = data[i].data;
				for(int l = 0; l < layer_count; l++) {
					results[l + 1] = layers[l]->forward(results[l]);
				}

				/* Restore to original [outputs] size delta. which is changed during backpropagation */
				NUM_TYPE* delta = orig_delta;

				/* Calculate delta for the output layer */
				#pragma omp parallel for
				for(int j = 0; j < outputs; j++) {
					delta[j] = data[i].label[j] - results[layer_count][j];
				}

				/* Backpropagate and get a new delta for the next('backward') layer. */
				for(int l = layer_count - 1; l >= 0; l--) {
					delta = layers[l]->backward(delta);
				}
				
				/* Update weights with learning rate 0.005 */
				#pragma omp parallel for
				for(int l = 0; l < layer_count; l++) {
					layers[l]->update_weights(results[l], 0.005);
				}
			}

			delete[] results;
			delete[] orig_delta;
		}

		NUM_TYPE* predict(NUM_TYPE* data) {
			for(unsigned int i = 0; i < layer_count; i++) {
				data = layers[i]->forward(data);
			}
			return data;
		}

		~Network() {
			for(unsigned int i = 0; i < layer_count; i++) {
				delete layers[i];
			}
			delete[] layers;
		}

		void dump_network(std::ostream& const output) {
			output.write("NeNet", 5);
			output.write((char*) &layer_count, sizeof(layer_count));
			for (int i = 0; i < layer_count; i++) {
				int inputs = layers[i]->inputs;
				int outputs = layers[i]->outputs;
				output.write((char*) &inputs, sizeof(inputs));
				output.write((char*) &outputs, sizeof(outputs));

				std::vector<NUM_TYPE> weights = layers[i]->dump_weights();
				int size = weights.size();
				output.write((char*) &size, sizeof(size));
				output.write((char*) &weights[0], sizeof(NUM_TYPE) * size);
			}
		}
	private:
		Layer** layers;
		const unsigned int layer_count;
		const unsigned int inputs, outputs;

		Network(unsigned int layer_count, Layer** layers, unsigned int inputs, unsigned int outputs)
			: layers(layers), layer_count(layer_count), inputs(inputs), outputs(outputs) {}
	};
	
}
