#include <iostream>
#include "cuFCPE.h"

int main(int argc, char** argv)
{
	try
	{
		cuFCPE_Test1(argc, argv);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}