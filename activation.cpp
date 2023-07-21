#include <cmath>
#include <vector>
#include "headers/activation.h"
#include "headers/vector.h"

namespace Activation
{
    Math::nVector Sigmoid::fn(Math::nVector x)
    {
        std::vector<double> v;
        for (double i : x.getVals()) 
        {
            v.push_back(1.0 / (1 + exp(-i)));
        }
        return Math::nVector(v);
    }

    Math::nVector Sigmoid::fnDerv(Math::nVector x)
    {
        std::vector<double> v;
        for (double i : x.getVals()) 
        {
            double sig = 1.0 / (1 + exp(-i));
            v.push_back(sig * (1 - sig));
        }
        return Math::nVector(v);
    }

    Math::nVector Relu::fn(Math::nVector x)
    {
        std::vector<double> v;
        for (double i : x.getVals()) 
        {
            v.push_back((i > 0) ? i : 0);
        }
        return Math::nVector(v);
    }

    Math::nVector Relu::fnDerv(Math::nVector x)
    {
        std::vector<double> v;
        for (double i : x.getVals()) 
        {
            v.push_back((i > 0) ? 1 : 0);
        }
        return Math::nVector(v);
    }

    Math::nVector LeakyRelu::fn(Math::nVector x)
    {
        std::vector<double> v;
        for (double i : x.getVals()) 
        {
            v.push_back((i > 0) ? i : (i * m));
        }
        return Math::nVector(v);
    }

    Math::nVector LeakyRelu::fnDerv(Math::nVector x)
    {
        std::vector<double> v;
        for (double i : x.getVals()) 
        {
            v.push_back((i > 0) ? 1 : m);
        }
        return Math::nVector(v);
    }

}