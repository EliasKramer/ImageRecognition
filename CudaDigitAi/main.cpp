#include <stdio.h>
#include "neural_network.h"
int main()
{
	printf("Hello World!");
	
	n_network& network = create_network(28 * 28, 2, 16, 10);
	delete_network(network);
	
	return 0;
}