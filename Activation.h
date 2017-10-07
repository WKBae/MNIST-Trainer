#pragma once
#include "Common.h"
#include <cmath>

namespace nn {

	class ActivationFunction {
	public:
		virtual ~ActivationFunction() {}
		virtual int getId() = 0;
		virtual NUM_TYPE calculate(NUM_TYPE x) = 0;
		virtual NUM_TYPE derivative(NUM_TYPE x) = 0;

		enum types {
			Sigmoid = 0,
			Tanh,
			HardSigmoid,
			ReLU,
			LeakyReLU,
			ELU,
		};
	};

	class Sigmoid : public ActivationFunction {
	public:
		int getId() override {
			return ActivationFunction::types::Sigmoid;
		}

		NUM_TYPE calculate(NUM_TYPE x) override {
			return 1.0 / (1.0 + exp(-x));
		}
		NUM_TYPE derivative(NUM_TYPE x) override {
			double f = calculate(x);
			return f * (1.0 - f);
		}
	};

	class Tanh : public ActivationFunction {
	public:
		int getId() override {
			return ActivationFunction::types::Tanh;
		}

		NUM_TYPE calculate(NUM_TYPE x) override {
			return tanh(x);
		}
		NUM_TYPE derivative(NUM_TYPE x) override {
			double f = calculate(x);
			return 1.0 - (f * f);
		}
	};

	class HardSigmoid : public ActivationFunction {
	public:
		int getId() override {
			return ActivationFunction::types::HardSigmoid;
		}

		NUM_TYPE calculate(NUM_TYPE x) override {
			NUM_TYPE x_ = x * 0.2 + 0.5;
			return (x < 0? 0 : (x > 1? 1 : x));
		}
		NUM_TYPE derivative(NUM_TYPE x) override {
			return (x < 0 || x > 1)? 0 : 0.2;
		}
	};

	class ReLU : public ActivationFunction {
	public:
		int getId() override {
			return ActivationFunction::types::ReLU;
		}

		NUM_TYPE calculate(NUM_TYPE x) override {
			return (x >= 0) ? x : 0;
		}
		NUM_TYPE derivative(NUM_TYPE x) override {
			return (x >= 0) ? 1 : 0;
		}
	};

	class LeakyReLU : public ActivationFunction {
	public:
		int getId() override {
			return ActivationFunction::types::LeakyReLU;
		}

		NUM_TYPE calculate(NUM_TYPE x) override {
			return (x >= 0) ? x : 0.01 * x;
		}
		NUM_TYPE derivative(NUM_TYPE x) override {
			return (x >= 0) ? 1 : 0.01;
		}
	};

	class ELU : public ActivationFunction {
		const NUM_TYPE alpha = 1.0;
	public:
		int getId() override {
			return ActivationFunction::types::ELU;
		}

		NUM_TYPE calculate(NUM_TYPE x) override {
			return (x >= 0) ? x : alpha * (exp(x) - 1.0);
		}
		NUM_TYPE derivative(NUM_TYPE x) override {
			return (x >= 0) ? 1 : calculate(x) + alpha;
		}
	};

}