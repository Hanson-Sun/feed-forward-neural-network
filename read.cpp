#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include "headers/matrix.h"
#include "headers/vector.h"
#include "headers/read.h"

Math::nVector generateLabel(int i)
{
    std::vector<double> v(10, 0);
    v[i] = 1;
    return Math::nVector(v);
}


std::vector<std::tuple<Math::nVector, Math::nVector>> readData(std::string path)
{
    std::vector<std::tuple<Math::nVector, Math::nVector>> data;

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
                {
                    row.push_back(std::stod(word) / 255);
                }
            }

            Math::nVector label = generateLabel((int) row[0]);
            row.erase(row.begin());
            Math::nVector input(row);

            data.push_back(std::tuple<Math::nVector, Math::nVector>(input, label));
            
            count++;
        }
    }
    if (count == 0)
        std::cout << "Data not loaded." << std::endl;

    return data;
}
