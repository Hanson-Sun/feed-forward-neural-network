#include <vector>
#include <cmath>
#include "Cost.h"
#include "Matrix.h"

namespace Cost
{
    double L2Cost::func(Math::Matrix x, Math::Matrix y)
    {
        Math::Matrix diff = x - y;
        double sum = 0;
        for (int i = 0; i < diff.getSize(); i++)
            sum += diff[i] * diff[i];

        return sum;
    }
    Math::Matrix L2Cost::funcDx(Math::Matrix x, Math::Matrix y)
    {
        return (x - y);
    }

    double CrossEntropy::func(Math::Matrix x, Math::Matrix y)
    {
        double sum = 0;
        for (int i = 0; i < x.getSize(); i++)
        {
            double val{};
            if (1 - x[i] <= 0)
                val = y[i] * std::log(x[i]);
            else if (x[i] <= 0)
                val = (1 - y[i]) * std::log(1 - x[i]);
            else
                val = y[i] * std::log(x[i]) + (1 - y[i]) * std::log(1 - x[i]);

            sum += (val > error) ? 0 : val;
        }
        return -sum;
    }

    Math::Matrix CrossEntropy::funcDx(Math::Matrix x, Math::Matrix y)
    {
        int size = x.getSize();

        std::vector<double> v(size);

        for (int i = 0; i < size; i++)
        {
            double val = -(y[i] - x[i]) / (x[i] * (1 - x[i]));
            v[i] = (val > error) ? 0 : val;
        }

        return Math::Matrix(v);
    }
}