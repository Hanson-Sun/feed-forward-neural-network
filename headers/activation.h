#pragma once
#include "matrix.h"

namespace Activation
{
    class ActivationFn
    {
    public:
        virtual Math::Matrix fn(Math::Matrix x) = 0;
        virtual Math::Matrix fnDerv(Math::Matrix x) = 0;
    };

    class Sigmoid : public ActivationFn
    {
    public:
        Math::Matrix fn(Math::Matrix x) override;
        Math::Matrix fnDerv(Math::Matrix x) override;
    };

    class Relu : public ActivationFn
    {
    public:
        Math::Matrix fn(Math::Matrix x) override;
        Math::Matrix fnDerv(Math::Matrix x) override;
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
        Math::Matrix fn(Math::Matrix x) override;
        Math::Matrix fnDerv(Math::Matrix x) override;
    };
}