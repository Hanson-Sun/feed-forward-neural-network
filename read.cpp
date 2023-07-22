#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include <cmath>
#include "headers/matrix.h"
#include "headers/read.h"

Math::Matrix generateLabel(int i)
{
    std::vector<double> v(10, 0);
    v[i] = 1;
    return Math::Matrix(v);
}

std::vector<std::tuple<Math::Matrix, Math::Matrix>> readData(std::string path, int size)
{
    std::vector<std::tuple<Math::Matrix, Math::Matrix>> data;

    std::fstream fin;
    // opens an existing csv file or creates a new file.
    fin.open(path, std::ios::in);

    std::string line, word;
    int count = 0;

    if (fin.is_open())
    {
        while (std::getline(fin, line))
        {
            std::vector<double> row;
            std::stringstream s(line);
            
            if (count >= 1)
            {
                while (std::getline(s, word, ','))
                    row.push_back(std::stod(word));
                
                Math::Matrix label = generateLabel((int)std::round(row[0]));
                row.erase(row.begin());

                // for (double &r : row)
                //     r /= 255;
                
                data.push_back(std::tuple<Math::Matrix, Math::Matrix>(Math::Matrix(row), label));
            }

            if (count > size)
            {
                break;
            }

            count++;
        }
    }
    if (count == 0)
        std::cout << "Data not loaded." << std::endl;

    return data;
}
