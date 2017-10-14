#pragma once
#include "Config.h"
#include <cmath>

namespace nn {
	/** Various activation functions implemented */
	namespace activation {
		namespace types {
			enum {
				Sigmoid = 0,
				Tanh,
				HardSigmoid,
				ReLU,
				LeakyReLU,
				ELU,
				Linear,
			};
		}

		class ActivationFunction {
		public:
			virtual ~ActivationFunction() {}
			virtual int getId() = 0;
			virtual NUM_TYPE calculate(NUM_TYPE x) = 0;
			virtual NUM_TYPE derivative(NUM_TYPE x) = 0;
		};

		class Sigmoid : public ActivationFunction {
		public:
			int getId() override {
				return types::Sigmoid;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				return 1.0 / (1.0 + exp(-x));
			}
			NUM_TYPE derivative(NUM_TYPE x) override {
				NUM_TYPE f = calculate(x);
				return f * (1.0 - f);
			}
		};

		class Tanh : public ActivationFunction {
		public:
			int getId() override {
				return types::Tanh;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				return tanh(x);
			}
			NUM_TYPE derivative(NUM_TYPE x) override {
				NUM_TYPE f = calculate(x);
				return 1.0 - (f * f);
			}
		};

		class HardSigmoid : public ActivationFunction {
		public:
			int getId() override {
				return types::HardSigmoid;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				NUM_TYPE x_ = x * 0.2 + 0.5;
				return (x_ < 0 ? 0 : (x_ > 1 ? 1 : x_));
			}
			NUM_TYPE derivative(NUM_TYPE x) override {
				return (x < -2.5 || x > 2.5) ? 0 : 0.2;
			}
		};

		class ReLU : public ActivationFunction {
		public:
			int getId() override {
				return types::ReLU;
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
				return types::LeakyReLU;
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
				return types::ELU;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				return (x >= 0) ? x : alpha * (exp(x) - 1.0);
			}
			NUM_TYPE derivative(NUM_TYPE x) override {
				return (x >= 0) ? 1 : calculate(x) + alpha;
			}
		};

		/** Linear function. Just for testing. */
		class Linear : public ActivationFunction {
			const NUM_TYPE alpha = 1.0;
		public:
			int getId() override {
				return types::Linear;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				return alpha * x;
			}
			NUM_TYPE derivative(NUM_TYPE x) override {
				return alpha;
			}
		};
	}
}
