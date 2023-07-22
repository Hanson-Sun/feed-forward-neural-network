#include "headers/cost.h"
#include "headers/matrix.h"

namespace Cost
{
    double L2Cost::func(Math::Matrix x, Math::Matrix y)
    {
        Math::Matrix diff = x - y;
        double sum = 0;
        for (int i = 0; i < diff.getSize(); i++)
        {
            sum += diff[i] * diff[i];
        }
        return sum;
    }
    Math::Matrix L2Cost::funcDx(Math::Matrix x, Math::Matrix y)
    {
        return (x - y);
    }
}