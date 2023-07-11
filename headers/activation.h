#pragma once
#include "vector.h"

namespace Activation
{
    class ActivationFn
    {
    public:
        virtual Math::nVector fn(Math::nVector x) = 0;
        virtual Math::nVector fnDerv(Math::nVector x) = 0;
    };

    class Sigmoid : public ActivationFn
    {
    public:
        Math::nVector fn(Math::nVector x) override;
        Math::nVector fnDerv(Math::nVector x) override;
    };

    class Relu : public ActivationFn
    {
    public:
        Math::nVector fn(Math::nVector x) override;
        Math::nVector fnDerv(Math::nVector x) override;
    };

    class LeakyRelu : public ActivationFn
    {
    private:
        double m;
    public:
        LeakyRelu(double slope = 0.1)
        {
            m = slope;
        }
        Math::nVector fn(Math::nVector x) override;
        Math::nVector fnDerv(Math::nVector x) override;
    };
}