#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include "matrix.h"

Math::Matrix generateLabel(int i);

std::vector<std::tuple<Math::Matrix, Math::Matrix>> readData(std::string path, int size);