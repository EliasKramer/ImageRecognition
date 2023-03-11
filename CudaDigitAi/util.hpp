#pragma once
#include <random>
#include <chrono>

inline float sigmoid(float x)
{
	return 1.0f / (1.0f + exp(-x));
}
//reverse sigmoid
inline float logit(float x)
{
	return log(x / (1.0f - x));
}

inline float cost(float actual, float expected)
{
	return (actual - expected) * (actual - expected);
}

inline float sigmoid_derivative(float x)
{
	float sigmoid_activation = sigmoid(x);
	return sigmoid_activation * (1.0f - sigmoid_activation);
}

inline float cost_derivative(float actual, float expected)
{
	return 2.0f * (actual - expected);
}

inline float rand_between(float min, float max)
{
	// Get the current time as a seed
	auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

	// Create a random engine and distribution
	std::default_random_engine engine(seed);
	std::uniform_real_distribution<float> distribution(min, max);

	// Generate a random number within the given range
	return distribution(engine);
}
