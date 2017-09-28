#pragma once

/**
 * Defines layer and network data types for the neural network.
 * Note that the result arrays returned(NUM_TYPE* type return values) must not be modified.
 **/

#include "Common.h"
#include "Activation.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>

namespace nn {

	/** Abstract interface for a layer of a neural network */
	class Layer {
	public:
		virtual ~Layer() {}
		virtual NUM_TYPE* forward(NUM_TYPE* prev_f) = 0;
		virtual NUM_TYPE* backward(NUM_TYPE* prev_delta) = 0;
		virtual void initialize_weights() = 0;
		virtual void update_weights(NUM_TYPE* prev_f, NUM_TYPE learning_rate) = 0;

		virtual void load_weights(const FILE* fin) {
			// unimplemented
		}
		virtual void load_weights(const std::ifstream& fin) {
			// unimplemented
		}

		virtual void save_weights(const FILE* fout) {
			// unimplemented
		}
		virtual void save_weights(const std::ofstream& fout) {
			// unimplemented
		}
	};

	/** Real implementation of the layer, abstracted due to the requirement of the template argument. */
	template<typename Activation>
	class LayerImpl : public Layer {
	public:
		LayerImpl(unsigned int inputs, unsigned int outputs) : inputs(inputs), outputs(outputs) {
			weights = new NUM_TYPE*[inputs + 1];
			for(unsigned int i = 0; i <= inputs; i++) { // one for dummy input(constant 1)
				weights[i] = new NUM_TYPE[outputs];
			}

			last_f = new NUM_TYPE[outputs];
			last_delta = new NUM_TYPE[outputs];
			last_prop_delta = new NUM_TYPE[inputs];
		}
		~LayerImpl() {
			delete[] last_prop_delta;
			delete[] last_delta;
			delete[] last_f;

			for(unsigned int i = 0; i <= inputs; i++) {
				delete[] weights[i];
			}
			delete[] weights;
		}

		NUM_TYPE* forward(NUM_TYPE* prev_f) {
			memset(last_f, 0, sizeof(NUM_TYPE) * outputs);

			for(unsigned int i = 0; i < inputs; i++) {
				for(unsigned int j = 0; j < outputs; j++) {
					last_f[j] += prev_f[i] * weights[i][j];
				}
				1;
			}
			/* dummy input */
			for(unsigned int j = 0; j < outputs; j++) {
				last_f[j] += 1 * weights[inputs][j];
			}

			for(unsigned int i = 0; i < outputs; i++) {
				last_f[i] = activation.calculate(last_f[i]);
			}

			return last_f;
		}

		NUM_TYPE* backward(NUM_TYPE* prev_delta) {
			for(unsigned int i = 0; i < outputs; i++) {
				last_delta[i] = activation.derivative(last_f[i]) * prev_delta[i];
			}

			memset(last_prop_delta, 0, sizeof(NUM_TYPE) * inputs);
			for(unsigned int i = 0; i < inputs; i++) {
				for(unsigned int j = 0; j < outputs; j++) {
					last_prop_delta[i] += prev_delta[j] * weights[i][j];
				}
			}

			return last_prop_delta;
		}

		void initialize_weights() {
			for(unsigned int i = 0; i <= inputs; i++) {
				for(unsigned int j = 0; j < outputs; j++) {
					weights[i][j] = (rand() % 2 ? +1 : -1) * (rand() / (double) RAND_MAX);
				}
			}
		}
		void update_weights(NUM_TYPE* prev_f, NUM_TYPE learning_rate) {
			for(unsigned int i = 0; i < inputs; i++) {
				for(unsigned int j = 0; j < outputs; j++) {
					weights[i][j] += learning_rate * last_delta[j] * prev_f[i];
				}
			}
		}
	private:
		const unsigned int inputs, outputs;
		NUM_TYPE** weights;
		NUM_TYPE* last_f;
		NUM_TYPE* last_delta;
		NUM_TYPE* last_prop_delta;
		Activation activation;
	};

	/**
	 * The neural network.
	 * Composed of the layers, this class contains the operation for them including train and test(predict).
	 */
	class Network {
	public:
		static class Builder {
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

		void train(unsigned int n, NUM_TYPE** train_data, NUM_TYPE** labels) {
			NUM_TYPE** results = new NUM_TYPE*[layer_count + 1];
			NUM_TYPE* orig_delta = new NUM_TYPE[outputs];

			for(unsigned int i = 0; i < n; i++) {
				/* Retrieve the result(f = output) of the layers */
				results[0] = train_data[i];
				for(unsigned int l = 0; l < layer_count; l++) {
					results[l + 1] = layers[l]->forward(results[l]);
				}

				/* Restore to original [outputs] size delta. which is changed during backpropagation */
				NUM_TYPE* delta = orig_delta;
				/* Calculate delta for the output layer */
				for(unsigned int j = 0; j < outputs; j++) {
					delta[j] = (labels[i][j] - results[layer_count][j]) * results[layer_count][j] * (1 - results[layer_count][j]);
				}
				/* Backpropagate and get a new delta for the next('backward') layer. */
				for(int l = layer_count - 1; l >= 0; l--) {
					delta = layers[l]->backward(delta);
				}
				
				/* Update weights with learning rate 0.005 */
				for(unsigned int l = 0; l < layer_count; l++) {
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
	private:
		Layer** layers;
		const unsigned int layer_count;
		const unsigned int inputs, outputs;

		Network(unsigned int layer_count, Layer** layers, unsigned int inputs, unsigned int outputs)
			: layers(layers), layer_count(layer_count), inputs(inputs), outputs(outputs) {}
	};
	
}
