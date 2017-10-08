#pragma once

/**
 * Defines layer and network data types for the neural network.
 * Note that the result arrays returned(NUM_TYPE* type return values) must not be modified.
 **/

#include "Common.h"
#include "Activation.h"
#include "Layer.h"
#include "Dataset.h"

#include <cstring>
#include <cassert>
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
			/**
			 * Sets input size of the network.
			 * This needs to call first, if not `load()`ing a full network; calling this method clears any layers added before.
			 * @returns this, for chaining.
			 */
			Builder& input(unsigned int input_size) {
				delete_list(true);
				this->input_size = input_size;

				return *this;
			}
			/**
			 * Adds a layer with the output neurons given.
			 * Input size of the layer is set as the previous layer's output or `input()` size.
			 * The activation function is Sigmoid by default, this can be modified by specifying a function in the template.
			 * @throws std::invalid_argument when no `input()` is specified before.
			 * @returns this, for chaining.
			 */
			template<typename A = activation::Sigmoid>
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

				Layer* layer = new LayerImpl<A>(last_size, neurons);
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
			/**
			 * Builds the network.
			 * @throws std::length_error when `build()` is called with no layers added.
			 * @returns The network built.
			 */
			Network* build() {
				if (count <= 0) throw std::length_error("No layers present in the network definition!");
				Layer** layers = new Layer*[count];
				LayerList* curr = head;
				for(unsigned int i = 0; i < count && curr != NULL; i++, curr = curr->next) {
					layers[i] = curr->layer;
				}

				Network* net = new Network(count, layers, input_size, tail->output_size);
				//delete this;
				return net;
			}
			/**
			 * Loads a network from stream.
			 * This method can be called alone, or with other layers before or after the `load()` call.
			 * By combining other layers, the network can be pre-trained per each layers.
			 * @param input The input stream to read from.
			 * @throws std::invalid_argument if the stream is not a valid network.
			 * @throws std::runtime_error if the activation function, read from the stream, is unknown.
			 * @throws std::length_error if the layers cannot be connected due to the output and input size mismatch.
			 * @returns this, for chaining.
			 */
			Builder& load(std::istream& input) {
				char magic[6];
				input.read(magic, 5);
				magic[5] = '\0';
				if (input.fail() || strcmp(magic, "NeNet") != 0)
					throw std::invalid_argument("The input is not a network save file");

				int layers;
				input.read((char*) &layers, sizeof(layers));

				NUM_TYPE* weight_buf = NULL;
				int buf_size = -1;

				for (int i = 0; i < layers; i++) {
					char type;
					input.read(&type, sizeof(type));
					assert(!input.fail());

					int in, out;
					input.read((char*) &in, sizeof(in));
					assert(!input.fail());
					input.read((char*) &out, sizeof(out));
					assert(!input.fail());

					int weight_count;
					input.read((char*) &weight_count, sizeof(weight_count));
					assert(!input.fail());

					if (weight_count > buf_size) {
						NUM_TYPE* newbuf = new NUM_TYPE[weight_count];
						delete[] weight_buf;
						weight_buf = newbuf;
						buf_size = weight_count;
					}

					input.read((char*) weight_buf, sizeof(NUM_TYPE) * weight_count);
					assert(!input.fail());

					Layer* layer;
					switch(type) {
					case activation::types::Sigmoid:
						layer = new LayerImpl<activation::Sigmoid>(in, out);
						break;
					case activation::types::Tanh:
						layer = new LayerImpl<activation::Tanh>(in, out);
						break;
					case activation::types::HardSigmoid:
						layer = new LayerImpl<activation::HardSigmoid>(in, out);
						break;
					case activation::types::ReLU:
						layer = new LayerImpl<activation::ReLU>(in, out);
						break;
					case activation::types::LeakyReLU:
						layer = new LayerImpl<activation::LeakyReLU>(in, out);
						break;
					case activation::types::ELU:
						layer = new LayerImpl<activation::ELU>(in, out);
						break;
					default:
						throw std::runtime_error("Invalid activation function type!");
					}
					layer->load_weights(weight_buf, weight_count);

					LayerList* list = new LayerList;
					list->layer = layer;
					list->output_size = out;
					list->next = NULL;

					if (tail) {
						if (tail->output_size != in)
							throw std::length_error("Last layer's output size doesn't match the new layer's input size!");

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
			Builder& popLayer() {
				LayerList *prev = head;
				assert(prev != NULL);

				if(prev == tail) {
					delete_list(true);
					input_size = 0;
				} else {
					while (prev->next != tail)
						prev = prev->next;

					delete prev->next->layer;
					delete prev->next;
					count--;

					tail = prev;
				}
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

			void delete_list(bool delete_layers = false) {
				if(!head) return;

				for(LayerList *curr = head; curr != NULL;) {
					LayerList* next = curr->next;
					if(delete_layers) delete curr->layer;
					delete curr;
					curr = next;
				}
				
				head = tail = NULL;
				count = 0;
			}
		};

		/**
		 * Trains the network with the given data batch of size `n`.
		 * TODO: implement batch weight update, instead of one update per single data entry.
		 * @param n Number of data to read from the `data` array.
		 * @param data Data array used to train the network.
		 */
		void train(unsigned int n, DataEntry* data) {
			for(unsigned int i = 0; i < n; i++) {
				assert(data[i].data_count == inputs && data[i].label_count == outputs);

				/* Retrieve the result(f = output) of the layers */
				results[0] = data[i].data;
				for(int l = 0; l < layer_count; l++) {
					results[l + 1] = layers[l]->forward(results[l]);
				}

				/* Restore to pre-allocated [outputs] sized array. The pointer is changed during the backpropagation process */
				NUM_TYPE* delta = delta_buf;

				/* Calculate delta for the output layer */
				#pragma loop(hint_parallel(0))
				for(int j = 0; j < outputs; j++) {
					delta[j] = data[i].label[j] - results[layer_count][j];
				}

				/* Backpropagate and get a new delta for the next('backward') layer. */
				for(int l = layer_count - 1; l >= 0; l--) {
					delta = layers[l]->backward(delta);
				}
				
				/* Update weights with their optimizer */
				#pragma loop(hint_parallel(0))
				for(int l = 0; l < layer_count; l++) {
					layers[l]->update_weights(results[l]);
				}
			}
		}

		/**
		 * Predict using the given input, forward-propagated through the network.
		 * @param data Input data. Asserts the length is `Network::inputs`.
		 * @returns Predicted result, the length is same as `Network::outputs`.
		 */
		NUM_TYPE* predict(NUM_TYPE* data) {
			for(int i = 0; i < layer_count; i++) {
				data = layers[i]->forward(data);
			}
			return data;
		}

		~Network() {
			delete[] delta_buf;
			delete[] results;

			for(int i = 0; i < layer_count; i++) {
				delete layers[i];
			}
			delete[] layers;
		}

		/**
		 * Writes the network to stream.
		 * The saved network can be loaded by `Builder::load()`.
		 * @param output Stream to dump this network
		 */
		void dump_network(std::ostream& output) {
			output.write("NeNet", 5);
			output.write((char*) &layer_count, sizeof(layer_count));
			for (int i = 0; i < layer_count; i++) {
				char type = layers[i]->getActivationType();
				int inputs = layers[i]->inputs;
				int outputs = layers[i]->outputs;
				output.write(&type, sizeof(type));
				output.write((char*) &inputs, sizeof(inputs));
				output.write((char*) &outputs, sizeof(outputs));

				std::vector<NUM_TYPE> weights = layers[i]->dump_weights();
				int size = weights.size();
				output.write((char*) &size, sizeof(size));
				output.write((char*) &weights[0], sizeof(NUM_TYPE) * size);
			}
		}

		const int layer_count;
		const int inputs, outputs;
	private:
		Layer** layers;
		NUM_TYPE** results;
		NUM_TYPE* delta_buf;

		Network(unsigned int layer_count, Layer** layers, unsigned int inputs, unsigned int outputs)
			: layers(layers), layer_count(layer_count), inputs(inputs), outputs(outputs), results(new NUM_TYPE*[layer_count + 1]), delta_buf(new NUM_TYPE[outputs]) {}
	};
	
}
