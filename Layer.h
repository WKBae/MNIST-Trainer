#pragma once

#include "Common.h"
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>

#define OPTIMIZE_ADAM
//#define OPTIMIZE_RMSPROP
//#define OPTIMIZE_ADAGRAD
//#define OPTIMIZE_NESTEROV
//#define OPTIMIZE_MOMENTUM
// default SG


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
		virtual void update_weights(NUM_TYPE* prev_f) = 0;

		virtual char getActivationType() = 0;
		virtual std::vector<NUM_TYPE> dump_weights() { return std::vector<NUM_TYPE>(); }
		virtual int load_weights(NUM_TYPE* begin, int limit = -1) { return 0; }
	};

	/** Real implementation of the layer, abstracted due to the requirement of the template argument. */
	template<typename Activation>
	class LayerImpl : public Layer {
	public:
		LayerImpl(unsigned int inputs, unsigned int outputs) : Layer(inputs, outputs) {
			weights = new NUM_TYPE[(inputs + 1) * outputs];

#if defined(OPTIMIZE_ADAM)
			last_m = new NUM_TYPE[(inputs + 1) * outputs]();
			last_v = new NUM_TYPE[(inputs + 1) * outputs]();
			t = 0;
			lr_t = 0;
			beta1_sq = beta2_sq = 1;
#elif defined(OPTIMIZE_ADAGRAD) || defined(OPTIMIZE_RMSPROP)
			last_g = new NUM_TYPE[(inputs + 1) * outputs]();
#elif defined(OPTIMIZE_MOMENTUM) || defined(OPTIMIZE_NESTEROV)
			last_v = new NUM_TYPE[(inputs + 1) * outputs]();
#endif

			last_f = new NUM_TYPE[outputs];
			last_delta = new NUM_TYPE[outputs];
			last_prop_delta = new NUM_TYPE[inputs];
		}
		~LayerImpl() {
			delete[] last_prop_delta;
			delete[] last_delta;
			delete[] last_f;

#if defined(OPTIMIZE_RMSPROP)
			delete[] last_g;
#elif defined(OPTIMIZE_MOMENTUM)
			delete[] last_v;
#endif
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
		void update_weights(NUM_TYPE* prev_f) override {
#if defined(OPTIMIZE_ADAM)
			t++;
			beta1_sq *= beta1;
			beta2_sq *= beta2;
			lr_t = learning_rate * sqrt(1.0 - beta2_sq) / (1.0 - beta1_sq);
#endif
			#pragma omp parallel for
			for (int j = 0; j < outputs; j++) {
				NUM_TYPE delta = last_delta[j];
				for (int i = 0; i < inputs; i++) {
					NUM_TYPE loss = delta * prev_f[i];
					weight(i, j) += weight_diff(i, j, loss);
				}
				weight(inputs, j) += weight_diff(inputs, j, delta);
			}
		}

		char getActivationType() override {
			return (char) activation.getId();
		}

		std::vector<NUM_TYPE> dump_weights() override {
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

#if defined(OPTIMIZE_ADAM)
		NUM_TYPE& m(unsigned int from, unsigned int to) {
			assert(from <= inputs && to < outputs);
			return last_m[to * inputs + from];
		}
		NUM_TYPE* last_m;

		NUM_TYPE& v(unsigned int from, unsigned int to) {
			assert(from <= inputs && to < outputs);
			return last_v[to * inputs + from];
		}
		NUM_TYPE* last_v;

		int t;
		NUM_TYPE lr_t;
		const NUM_TYPE learning_rate = 0.0005, beta1 = 0.93, beta2 = 0.9999;
		NUM_TYPE beta1_sq, beta2_sq;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss, NUM_TYPE epsilon = 1e-8) {
			m(i, j) = beta1 * m(i, j) + (1.0 - beta1) * loss;
			v(i, j) = beta2 * v(i, j) + (1.0 - beta2) * (loss * loss);
			return lr_t * m(i, j) / (sqrt(v(i, j)) + epsilon);
		}
#elif defined(OPTIMIZE_RMSPROP)
		NUM_TYPE& g(unsigned int from, unsigned int to) {
			assert(from <= inputs && to < outputs);
			return last_g[to * inputs + from];
		}
		NUM_TYPE* last_g;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss, NUM_TYPE learning_rate = 0.0002, NUM_TYPE rho = 0.999, NUM_TYPE epsilon = 1e-8) {
			g(i, j) = rho * g(i, j) + (1.0 - rho) * (loss * loss);
			return learning_rate * loss / (sqrt(g(i, j)) + epsilon);
		}
#elif defined(OPTIMIZE_ADAGRAD)
		NUM_TYPE& g(unsigned int from, unsigned int to) {
			assert(from <= inputs && to < outputs);
			return last_g[to * inputs + from];
		}
		NUM_TYPE* last_g;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss, NUM_TYPE learning_rate = 0.0005, NUM_TYPE epsilon = 1e-10) {
			g(i, j) += loss * loss;
			return learning_rate * loss / (sqrt(g(i, j)) + epsilon);
		}
#elif defined(OPTIMIZE_NESTEROV)
		NUM_TYPE& velocity(unsigned int from, unsigned int to) {
			assert(from <= inputs && to < outputs);
			return last_v[to * inputs + from];
		}
		NUM_TYPE* last_v;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss, NUM_TYPE learning_rate = 0.0001, NUM_TYPE momentum_factor = 0.95) {
			NUM_TYPE prev_v = velocity(i, j);
			velocity(i, j) = momentum_factor * velocity(i, j) + learning_rate * loss;
			return momentum_factor * prev_v + (1 + momentum_factor) * velocity(i, j);
		}
#elif defined(OPTIMIZE_MOMENTUM)
		NUM_TYPE& velocity(unsigned int from, unsigned int to) {
			assert(from <= inputs && to < outputs);
			return last_v[to * inputs + from];
		}
		NUM_TYPE* last_v;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss, NUM_TYPE learning_rate = 0.0003, NUM_TYPE momentum_factor = 0.98) {
			return velocity(i, j) = momentum_factor * velocity(i, j) + learning_rate * loss;
		}
#else
		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss, NUM_TYPE learning_rate = 0.0001) {
			return learning_rate * loss;
		}
#endif


		NUM_TYPE* last_f;
		NUM_TYPE* last_delta;
		NUM_TYPE* last_prop_delta;
		Activation activation;
	};
}
