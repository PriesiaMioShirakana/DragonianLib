#pragma once
#ifdef _WIN32
#define Py_EXPORTED_SYMBOL __declspec(dllexport)
#else
#define Py_EXPORTED_SYMBOL
#endif

Py_EXPORTED_SYMBOL void cuFCPE_Test1(int argc, char** argv);
Py_EXPORTED_SYMBOL void cuFCPE_Test2(int argc, char** argv);
Py_EXPORTED_SYMBOL void cuFCPE_Test3(int argc, char** argv);