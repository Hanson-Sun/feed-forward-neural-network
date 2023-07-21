#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include "matrix.h"
#include "vector.h"


Math::nVector generateLabel(int i);

std::vector<std::tuple<Math::nVector, Math::nVector>> readData(std::string path, int size);