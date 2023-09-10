#include <cmath>
#include <vector>
#include "headers/Activation.h"
#include "headers/Matrix.h"

namespace Activation
{
    Math::Matrix Sigmoid::fn(Math::Matrix x)
    {
        x.applyFnHere([](double i)
                      { return 1.0 / (1 + exp(-i)); });

        return x;
    }

    Math::Matrix Sigmoid::fnDerv(Math::Matrix x)
    {
        x.applyFnHere([](double i)
                      {
            double sig = 1.0 / (1 + exp(-i));
            return sig * (1 - sig); });
        return x;
    }

    Math::Matrix Linear::fn(Math::Matrix x)
    {
        return x;
    }

    Math::Matrix Linear::fnDerv(Math::Matrix x)
    {
        return Math::Matrix(x.getRows(), x.getCols(), 1);
    }

    Math::Matrix Relu::fn(Math::Matrix x)
    {

        x.applyFnHere([](double i)
                      { return (i > 0) ? i : 0.0; });

        return x;
    }

    Math::Matrix Relu::fnDerv(Math::Matrix x)
    {

       x.applyFnHere([](double i)
                      { return (i > 0) ? 1.0 : 0.0; });

        return x;
    }

    Math::Matrix LeakyRelu::fn(Math::Matrix x)
    {
        x.applyFnHere([&](double i) -> double
                      {return ((i > 0) ? i : (i * m)); });

        return x;
    }

    Math::Matrix LeakyRelu::fnDerv(Math::Matrix x)
    {
        x.applyFnHere([&](double i) -> double
                      {return (i > 0) ? 1 : m; });

        return x;
    }

}