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
				Absolute,
				HardTanh,
				Sine,
				Cosine,
				Sinc,
			};
		}

		class ActivationFunction {
		public:
			virtual ~ActivationFunction() {
			}

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
				return (x < -2.5)
					       ? 0
					       : (x <= 2.5)
					       ? 0.2 * x + 0.5
					       : 1;
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
			const NUM_TYPE alpha = 0.7;
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

		class Absolute : public ActivationFunction {
		public:
			int getId() override {
				return types::Absolute;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				return abs(x);
			}

			NUM_TYPE derivative(NUM_TYPE x) override {
				return (x < 0) ? -1 : 1;
			}
		};

		class HardTanh : public ActivationFunction {
		public:
			int getId() override {
				return types::HardTanh;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				return (x < -1) ? -1 : (x <= 1) ? x : 1;
			}

			NUM_TYPE derivative(NUM_TYPE x) override {
				return (x < -1 || x > 1) ? 0 : 1;
			}
		};

		class Sine : public ActivationFunction {
		public:
			int getId() override {
				return types::Sine;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				return sin(x);
			}

			NUM_TYPE derivative(NUM_TYPE x) override {
				return cos(x);
			}
		};

		class Cosine : public ActivationFunction {
		public:
			int getId() override {
				return types::Cosine;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				return cos(x);
			}

			NUM_TYPE derivative(NUM_TYPE x) override {
				return -sin(x);
			}
		};

		class Sinc : public ActivationFunction {
		public:
			int getId() override {
				return types::Sinc;
			}

			NUM_TYPE calculate(NUM_TYPE x) override {
				return (x == 0) ? 1 : sin(x) / x;
			}

			NUM_TYPE derivative(NUM_TYPE x) override {
				return (x == 0) ? 0 : (cos(x) - sin(x) / x) / x;
			}
		};
	}
}
