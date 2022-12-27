#include "node.h"
#include <memory>
void calculate(node& n, node* input_nodes, int num_input_nodes)
{
	float sum = 0;
	for (int i = 0; i < num_input_nodes; i++)
	{
		sum += input_nodes[i].value * n.node->weights[i];
	}
	n.value = sigmoid(sum + n.node->bias);
}