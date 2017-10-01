#pragma once

#include "Common.h"
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>


namespace nn {

	/** Abstract interface for a layer of a neural network */
	class Layer {
	public:
		Layer(unsigned int inputs, unsigned int outputs) : inputs(inputs), outputs(outputs) {}
		virtual ~Layer() {}

		const int inputs, outputs;

		virtual NUM_TYPE* forward(NUM_TYPE* prev_f) = 0;
		virtual NUM_TYPE* backward(NUM_TYPE* prev_delta) = 0;
		virtual void initialize_weights() = 0;
		virtual void update_weights(NUM_TYPE* prev_f, NUM_TYPE learning_rate) = 0;

		virtual std::vector<NUM_TYPE> dump_weights() { return std::vector<NUM_TYPE>(); }
		virtual int load_weights(NUM_TYPE* begin, int limit = -1) { return 0; }
	};

	/** Real implementation of the layer, abstracted due to the requirement of the template argument. */
	template<typename Activation>
	class LayerImpl : public Layer {
	public:
		LayerImpl(unsigned int inputs, unsigned int outputs) : Layer(inputs, outputs) {
			weights = new NUM_TYPE[(inputs + 1) * outputs];

			last_f = new NUM_TYPE[outputs];
			last_delta = new NUM_TYPE[outputs];
			last_prop_delta = new NUM_TYPE[inputs];
		}
		~LayerImpl() {
			delete[] last_prop_delta;
			delete[] last_delta;
			delete[] last_f;

			delete[] weights;
		}

		NUM_TYPE* forward(NUM_TYPE* prev_f) override {
			#pragma omp parallel for
			for (int j = 0; j < outputs; j++) {
				NUM_TYPE sum = 0;
				for (int i = 0; i < inputs; i++) {
					sum += prev_f[i] * weight(i, j);
				}
				last_f[j] = activation.calculate(sum + weight(inputs, j));
			}

			return last_f;
		}

		NUM_TYPE* backward(NUM_TYPE* prev_delta) override {
			#pragma omp parallel for
			for(int i = 0; i < outputs; i++) {
				last_delta[i] = activation.derivative(last_f[i]) * prev_delta[i];
			}

			#pragma omp parallel for
			for(int i = 0; i < inputs; i++) {
				NUM_TYPE sum = 0;
				for(int j = 0; j < outputs; j++) {
					sum += last_delta[j] * weight(i, j);
				}
				last_prop_delta[i] = sum;
			}

			return last_prop_delta;
		}

		void initialize_weights() override {
			for(int i = 0; i <= inputs; i++) {
				for(int j = 0; j < outputs; j++) {
					weight(i, j) = ((double) rand() / RAND_MAX) - 0.5;
				}
			}
		}
		void update_weights(NUM_TYPE* prev_f, NUM_TYPE learning_rate) {
			#pragma omp parallel for
			for (int j = 0; j < outputs; j++) {
				NUM_TYPE delta = last_delta[j];
				for (int i = 0; i < inputs; i++) {
					weight(i, j) += learning_rate * (delta * prev_f[i]);
				}
				weight(inputs, j) += learning_rate * delta;
			}
		}

		std::vector<NUM_TYPE> dump_weights() {
			std::vector<NUM_TYPE> buf;
			buf.reserve((inputs + 1) * outputs);
			for (int i = 0; i <= inputs; i++) {
				for (int j = 0; j < outputs; j++) {
					buf.push_back(weight(i, j));
				}
			}
			return buf;
		}
		int load_weights(NUM_TYPE* begin, int limit = -1) override {
			int idx = 0;
			for (int i = 0; i <= inputs; i++) {
				for (int j = 0; j < outputs; j++) {
					if (limit >= 0 && idx >= limit) goto fail_too_short;
					weight(i, j) = begin[idx++];
				}
			}
			return idx;
		fail_too_short:
			return -1;
		}

	private:
		NUM_TYPE& weight(unsigned int from, unsigned int to) {
			assert(from <= inputs && to < outputs);
			return weights[to * inputs + from];
		}

		NUM_TYPE* weights;
		NUM_TYPE* last_f;
		NUM_TYPE* last_delta;
		NUM_TYPE* last_prop_delta;
		Activation activation;
	};
}
