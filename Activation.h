#pragma once
#include "Common.h"
#include <cmath>

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
			return f * (1.0 - f);
		}
	};

	class Tanh : public ActivationFunction {
	public:
		NUM_TYPE calculate(NUM_TYPE x) {
			return tanh(x);
		}
		NUM_TYPE derivative(NUM_TYPE x) {
			double f = calculate(x);
			return 1.0 - (f * f);
		}
	};

	class ReLU : public ActivationFunction {
	public:
		NUM_TYPE calculate(NUM_TYPE x) {
			return (x >= 0) ? x : 0;
		}
		NUM_TYPE derivative(NUM_TYPE x) {
			return (x >= 0) ? 1 : 0;
		}
	};

}