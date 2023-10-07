#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include <cmath>
#include "Matrix.h"
#include "Read.h"

Math::Matrix generateLabel(int i)
{
    std::vector<double> v(10, -1);
    v[i] = 1;
    return Math::Matrix(v);
}

dataset readData(std::string path, int size)
{
    dataset data;

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
                
                Math::Matrix label = generateLabel((int) std::round(row[0]));
                row.erase(row.begin());

                for (double &r : row)
                    r /= 255;
                
                data.push_back(data_pair(Math::Matrix(row), label));
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
