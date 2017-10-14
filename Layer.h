#pragma once

#include "Config.h"
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>

#ifdef XAVIER_INITIALIZATION
#include <limits>
#endif

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

	/** Real implementation of the layer, abstracted to add capability to use activation functions per layer. */
	template<typename Activation>
	class LayerImpl : public Layer {
	public:
		LayerImpl(unsigned int inputs, unsigned int outputs)
		: Layer(inputs, outputs),
			weights(new NUM_TYPE[(inputs + 1) * outputs]),

#if defined(OPTIMIZE_ADAM)
			last_m(new NUM_TYPE[(inputs + 1) * outputs]()),
			last_v(new NUM_TYPE[(inputs + 1) * outputs]()),
			lr_t(0),
			beta1_sq(1), beta2_sq(1),
#elif defined(OPTIMIZE_ADAGRAD) || defined(OPTIMIZE_RMSPROP)
			last_g(new NUM_TYPE[(inputs + 1) * outputs]()),
#elif defined(OPTIMIZE_MOMENTUM) || defined(OPTIMIZE_NESTEROV)
			last_v(new NUM_TYPE[(inputs + 1) * outputs]()),
#endif

			last_f(new NUM_TYPE[outputs]),
			last_delta(new NUM_TYPE[outputs]),
			last_prop_delta(new NUM_TYPE[inputs]),
#endif
			activation()
		{

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

		/**
		 * Forward propagate with given input.
		 * @returns Calculated output of length same as the output of this layer. Should not be deleted or modified.
		 */
		NUM_TYPE* forward(NUM_TYPE* prev_f) override {
			#pragma omp parallel for
			for (int j = 0; j < outputs; j++) {
				NUM_TYPE sum = 0;
				for (int i = 0; i < inputs; i++) {
					sum += prev_f[i] * weight(i, j);
				}
				/* Bias(weight from constant-one) is just added with no multiplication */
				last_f[j] = activation.calculate(sum + weight(inputs, j));
			}

			return last_f;
		}

		/**
		* Backpropagate with error from the top layer.
		* This method calculates and keeps the loss. This will be used on weight update, and is overwritten on future `backward()` call.
		* TODO: To enable batch weight update, the loss have to be sumed up
		* @returns Error to propagate to lower layer, length of this layer's input. This array should not be deleted.
		*/
		NUM_TYPE* backward(NUM_TYPE* prev_delta) override {
			/* Calculate the loss derivative from the backpropagated delta */
			#pragma omp parallel for
			for(int i = 0; i < outputs; i++) {
				last_delta[i] = activation.derivative(last_f[i]) * prev_delta[i];
			}

			/* Calculate delta to propagate, to keep from this layer's weight to be used outside of this instance. */
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
					weight(i, j) =
#ifdef ZERO_BIAS_INITIALIZATION
					(i == inputs) ? /* bias */ 0 :
#endif
#ifdef XAVIER_INITIALIZATION
						//generateGaussianNoise(0, sqrt(3.0 / (inputs + outputs))) // use uniform version instead of normal(gaussian) dist.
						rand() * (1.0 / RAND_MAX) * (2 * 4.0 * sqrt(6.0 / (inputs + outputs))) - (4.0 * sqrt(6.0 / (inputs + outputs)))
#else
						((double)rand() / RAND_MAX) - 0.5
#endif
						;
				}
			}
		}
		void update_weights(NUM_TYPE* prev_f) override {
#ifdef LEARNING_RATE_DECAY
			learning_rate = INITIAL_LEARNING_RATE * decay_factor;
			decay_factor *= LEARNING_RATE_DECAY;
#endif
#if defined(OPTIMIZE_ADAM)
			beta1_sq *= ADAM_BETA1;
			beta2_sq *= ADAM_BETA2;
			lr_t = learning_rate * sqrt(1.0 - beta2_sq) / (1.0 - beta1_sq);
#endif
			#pragma omp parallel for
			for (int j = 0; j < outputs; j++) {
				NUM_TYPE delta = last_delta[j];
				for (int i = 0; i < inputs; i++) {
					NUM_TYPE loss = delta * prev_f[i];
					weight(i, j) +=
						weight_diff(i, j, loss)
#ifdef WEIGHT_DECAY
						- WEIGHT_DECAY * weight(i, j)
#endif
						;
				}
				weight(inputs, j) +=
					weight_diff(inputs, j, delta)
#ifdef WEIGHT_DECAY
					- WEIGHT_DECAY * weight(inputs, j)
#endif
					;
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
		NUM_TYPE& weight(unsigned int from, unsigned int to) const {
			assert(from <= inputs && to < outputs);
			/* Column-order to increase cache hit. Forward propagation and weight update are affected by this optimization. */
			return weights[to * inputs + from];
		}
		NUM_TYPE* weights;

		/* Optimizer implementation */
#if defined(OPTIMIZE_ADAM)
		NUM_TYPE& m(unsigned int from, unsigned int to) const {
			assert(from <= inputs && to < outputs);
			return last_m[to * inputs + from];
		}
		NUM_TYPE* last_m;

		NUM_TYPE& v(unsigned int from, unsigned int to) const {
			assert(from <= inputs && to < outputs);
			return last_v[to * inputs + from];
		}
		NUM_TYPE* last_v;

		NUM_TYPE lr_t;
		NUM_TYPE learning_rate = INITIAL_LEARNING_RATE;
		NUM_TYPE beta1_sq, beta2_sq;
		//const NUM_TYPE beta1 = ADAM_BETA1, beta2 = ADAM_BETA2;
		//const NUM_TYPE epsilon = ADAM_EPSILON;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss) {
			// small performance boost by storing the values temporary
			NUM_TYPE m_ = m(i, j) = ADAM_BETA1 * m(i, j) + (1.0 - ADAM_BETA1) * loss;
			NUM_TYPE v_ = v(i, j) = ADAM_BETA2 * v(i, j) + (1.0 - ADAM_BETA2) * (loss * loss);
			return lr_t * m_ / (sqrt(v_) + ADAM_EPSILON);
		}
#elif defined(OPTIMIZE_RMSPROP)
		NUM_TYPE& g(unsigned int from, unsigned int to) const {
			assert(from <= inputs && to < outputs);
			return last_g[to * inputs + from];
		}
		NUM_TYPE* last_g;

		NUM_TYPE learning_rate = INITIAL_LEARNING_RATE;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss) {
			NUM_TYPE g_ = g(i, j) = RMSPROP_RHO * g(i, j) + (1.0 - RMSPROP_RHO) * (loss * loss);
			return learning_rate * loss / (sqrt(g_) + RMSPROP_EPSILON);
		}
#elif defined(OPTIMIZE_ADAGRAD)
		NUM_TYPE& g(unsigned int from, unsigned int to) const {
			assert(from <= inputs && to < outputs);
			return last_g[to * inputs + from];
		}
		NUM_TYPE* last_g;
		
		NUM_TYPE learning_rate = INITIAL_LEARNING_RATE;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss) {
			NUM_TYPE g_ = g(i, j) += loss * loss;
			return learning_rate * loss / (sqrt(g_) + ADAGRAD_EPSILON);
		}
#elif defined(OPTIMIZE_NESTEROV)
		NUM_TYPE& v(unsigned int from, unsigned int to) const {
			assert(from <= inputs && to < outputs);
			return last_v[to * inputs + from];
		}
		NUM_TYPE* last_v;

		NUM_TYPE learning_rate = INITIAL_LEARNING_RATE;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss) {
			NUM_TYPE prev_v = v(i, j);
			NUM_TYPE v_ = v(i, j) = NESTROV_MOMENTUM_FACTOR * prev_v - learning_rate * loss;
			return NESTROV_MOMENTUM_FACTOR * prev_v - (1 + NESTROV_MOMENTUM_FACTOR) * v_;
		}
#elif defined(OPTIMIZE_MOMENTUM)
		NUM_TYPE& velocity(unsigned int from, unsigned int to) const {
			assert(from <= inputs && to < outputs);
			return last_v[to * inputs + from];
		}
		NUM_TYPE* last_v;

		NUM_TYPE learning_rate = INITIAL_LEARNING_RATE;

		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss) {
			return velocity(i, j) = MOMENTUM_MOMENTUM_FACTOR * velocity(i, j) + learning_rate * loss;
		}
#else
		NUM_TYPE learning_rate = INITIAL_LEARNING_RATE;
		NUM_TYPE weight_diff(int i, int j, NUM_TYPE loss) {
			return learning_rate * loss;
		}
#endif

#ifdef LEARNING_RATE_DECAY
		NUM_TYPE decay_factor = 1;
#endif
		/* End optimizer implementation */

		NUM_TYPE* last_f;
		NUM_TYPE* last_delta;
		NUM_TYPE* last_prop_delta;
		Activation activation;

#ifdef XAVIER_INITIALIZATION
		/* Gaussian distribution generator, from Wikipedia "Box-Muller transform" implementation */
		static double generateGaussianNoise(double mu, double sigma) {
			static const double epsilon = std::numeric_limits<double>::min();
			static const double two_pi = 2.0*3.14159265358979323846;

			thread_local double z1;
			thread_local bool generate;
			generate = !generate;

			if (!generate)
				return z1 * sigma + mu;

			double u1, u2;
			do
			{
				u1 = rand() * (1.0 / RAND_MAX);
				u2 = rand() * (1.0 / RAND_MAX);
			} while (u1 <= epsilon);

			double z0;
			z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
			z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
			return z0 * sigma + mu;
		}
#endif
	};
}
