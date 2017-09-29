#pragma once

#include "Common.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace nn {
	class ActivationFunction {
	public:
		virtual ~ActivationFunction() {}
		virtual NUM_TYPE calculate(NUM_TYPE x) = 0;
		virtual NUM_TYPE derivative(NUM_TYPE x) = 0;
	};
	class Sigmoid : public ActivationFunction {
	public:
		NUM_TYPE calculate(NUM_TYPE x) {
			return 1.0 / (1.0 + exp(-x));
		}
		NUM_TYPE derivative(NUM_TYPE x) {
			double f = calculate(x);
			return f * (1 - f);
		}
	};
	class ReLU : public ActivationFunction {
	public:
		NUM_TYPE calculate(NUM_TYPE x) {
			return (x >= 0)? x : 0;
		}
		NUM_TYPE derivative(NUM_TYPE x) {
			return (x >= 0)? 1 : 0;
		}
	};


	/** Abstract interface for a layer of a neural network */
	class Layer {
	public:
		virtual ~Layer() {}
		virtual NUM_TYPE* forward(NUM_TYPE* prev_f) = 0;
		virtual NUM_TYPE* backward(NUM_TYPE* prev_delta) = 0;
		virtual void initialize_weights() = 0;
		virtual void update_weights(NUM_TYPE* prev_f, NUM_TYPE learning_rate) = 0;

		virtual std::vector<NUM_TYPE> dump_weights() { return std::vector<NUM_TYPE>(); }
		virtual int load_weights(char* begin) { return 0; }
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
			/*
			#pragma omp single
			memset(last_f, 0, sizeof(NUM_TYPE) * outputs);
			*/
			/*#pragma omp parallel for
			for (int j = 0; j < outputs; j++) {
				NUM_TYPE sum = 0;
				#pragma omp parallel for reduction(+:sum)
				for (int i = 0; i < inputs; i++) {
					sum += prev_f[i] * weights[i][j];
				}
				last_f[j] = sum;
			}*/

			/*NUM_TYPE* sums = new NUM_TYPE[inputs * outputs];
			#pragma omp parallel for
			for (int i = 0; i < inputs; i++) {
				NUM_TYPE f = prev_f[i];
				NUM_TYPE* sumi = sums + (i * outputs);
				for (int j = 0; j < outputs; j++) {
					sumi[j] = f * weights[i][j];
				}
			}*/

			NUM_TYPE* w_transposed = new NUM_TYPE[outputs * inputs];
			for (int i = 0; i < inputs; i++) {
				for (int j = 0; j < outputs; j++) {
					w_transposed[j * inputs + i] = weights[i][j];
				}
			}

			#pragma omp parallel for
			for (int j = 0; j < outputs; j++) {
				NUM_TYPE sum = 0;
				NUM_TYPE* w_t = w_transposed + (j * inputs);
				#pragma omp parallel for reduction(+:sum)
				for (int i = 0; i < inputs; i++) {
					sum += prev_f[i] * w_t[i];
				}
				last_f[j] = activation.calculate(sum + weights[inputs][j]);
			}

			delete[] w_transposed;

			//delete[] sums;
			/* dummy input */
			//#pragma omp for
			//for(int j = 0; j < outputs; j++) {
			//	last_f[j] += /* (1 *) */ weights[inputs][j];
			//}

			/*#pragma omp parallel for
			for (int i = 0; i < outputs; i++) {
				last_f[i] = activation.calculate(last_f[i] + weights[inputs][i]);
			}*/
			return last_f;
		}

		NUM_TYPE* backward(NUM_TYPE* prev_delta) {
			#pragma omp parallel for
			for(int i = 0; i < outputs; i++) {
				last_delta[i] = activation.derivative(last_f[i]) * prev_delta[i];
			}

			/*
			#pragma omp single
			memset(last_prop_delta, 0, sizeof(NUM_TYPE) * inputs);
			*/

			#pragma omp parallel for
			for(int i = 0; i < inputs; i++) {
				NUM_TYPE sum = 0;
				#pragma omp parallel for reduction(+:sum)
				for(int j = 0; j < outputs; j++) {
					sum += last_delta[j] * weights[i][j];
				}
				last_prop_delta[i] = sum;
			}

			return last_prop_delta;
		}

		void initialize_weights() {
			for(int i = 0; i <= inputs; i++) {
				for(int j = 0; j < outputs; j++) {
					weights[i][j] = (rand() / (RAND_MAX / 2.0)) - 1.0;
				}
			}
		}
		void update_weights(NUM_TYPE* prev_f, NUM_TYPE learning_rate) {
			/*#pragma omp parallel for
			for(int i = 0; i < inputs; i++) {
				#pragma omp parallel for
				for(int j = 0; j < outputs; j++) {
					weights[i][j] += learning_rate * last_delta[j] * prev_f[i];
				}
			}*/
			/*#pragma omp parallel for
			for(int n = 0; n < inputs * outputs; n++) {
				int i = n / outputs;
				int j = n % outputs;
				weights[i][j] += learning_rate * (last_delta[j] * prev_f[i]);
			}
			#pragma omp parallel for
			for(int j = 0; j < outputs; j++) {
				weights[inputs][j] += learning_rate * last_delta[j]; /* (* 1) *
			}*/

			#pragma omp parallel for
			for (int i = 0; i <= inputs; i++) {
				NUM_TYPE f = (i < inputs) ? prev_f[i] : 1;
				for (int j = 0; j < outputs; j++) {
					weights[i][j] += learning_rate * (last_delta[j] * f);
				}
			}
		}

		std::vector<NUM_TYPE> dump_weights() {
			std::vector<NUM_TYPE> buf;
			buf.reserve((inputs + 1) * outputs);
			int idx = 0;
			for (int i = 0; i <= inputs; i++) {
				for (int j = 0; j < outputs; j++) {
					buf[idx++] = weights[i][j];
				}
			}
			return buf;
		}

	public:
		const unsigned int inputs, outputs;

	private:
		NUM_TYPE** weights;
		NUM_TYPE* last_f;
		NUM_TYPE* last_delta;
		NUM_TYPE* last_prop_delta;
		Activation activation;
	};
}
