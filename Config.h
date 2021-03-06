#pragma once

namespace nn {
	typedef double NUM_TYPE;

#define DEFAULT_HIDDEN_LAYER_1 200
#define DEFAULT_HIDDEN_LAYER_2 100
#define DEFAULT_MSE_THRESHOLD 0.001

#define DEFAULT_ACTIVATION_LAYER_1 nn::activation::Sigmoid
#define DEFAULT_ACTIVATION_LAYER_2 nn::activation::Sigmoid
#define DEFAULT_ACTIVATION_LAYER_3 nn::activation::Sigmoid

//#define PRINT_TRAIN_ERROR

#define BATCH_TRAIN

#define MINIBATCH_COUNT 2 // if not defined, use whole train set
#define TRAINS_PER_EPOCH 100 //(100 / MINIBATCH_COUNT)

#ifdef MINIBATCH_COUNT
	#define TEST_EPOCHES 100
	#define CHECKPOINT_EPOCHES 2000
#else
	#define TEST_EPOCHES 1
	#define CHECKPOINT_EPOCHES 10
#endif


//#define DROPOUT_RATE 0.2

//#define OPTIMIZE_ADAM
//#define OPTIMIZE_RMSPROP
//#define OPTIMIZE_ADAGRAD
#define OPTIMIZE_NESTEROV
//#define OPTIMIZE_MOMENTUM
// default GD

#if defined(OPTIMIZE_ADAM)
	#define INITIAL_LEARNING_RATE 0.001
	#define ADAM_BETA1 0.9
	#define ADAM_BETA2 0.999
	#define ADAM_EPSILON 1e-8
#elif defined(OPTIMIZE_RMSPROP)
	#define	INITIAL_LEARNING_RATE 0.0003
	#define RMSPROP_RHO 0.99985
	#define RMSPROP_EPSILON 1e-8
	#define LEARNING_RATE_DECAY 0.999992
#elif defined(OPTIMIZE_ADAGRAD)
	#define	INITIAL_LEARNING_RATE 0.0003
	#define RMSPROP_EPSILON 1e-8
#elif defined(OPTIMIZE_NESTEROV)
	#define INITIAL_LEARNING_RATE 0.004
	#define NESTEROV_MOMENTUM_FACTOR 0.95
	#define LEARNING_RATE_DECAY 0.999997
	#define WEIGHT_DECAY 0.00000006
#elif defined(OPTIMIZE_MOMENTUM)
	#define	INITIAL_LEARNING_RATE 0.002
	#define MOMENTUM_MOMENTUM_FACTOR 0.97
	#define LEARNING_RATE_DECAY 0.99997
	#define WEIGHT_DECAY 0.00000006
#else
	#define	INITIAL_LEARNING_RATE 0.05
	#define LEARNING_RATE_DECAY 0.999995
	#define WEIGHT_DECAY 0.0000001
#endif

//#define WEIGHT_DECAY (0.000005)

//#define XAVIER_INITIALIZATION
//#define ZERO_BIAS_INITIALIZATION

}

// int to string
#define STR_(x) #x
#define STR(x) STR_(x)
