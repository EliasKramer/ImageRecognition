#include "neural_math.h"
#include <math.h>
float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}