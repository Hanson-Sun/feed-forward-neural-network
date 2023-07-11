#pragma once
#include "vector.h"

namespace Cost
{
    class CostFn
    {
    public:
        virtual double func(Math::nVector x, Math::nVector y) = 0;
        virtual Math::nVector funcDx(Math::nVector x, Math::nVector y) = 0;
    };

    class L2Cost : public CostFn
    {
    public:
        double func(Math::nVector x, Math::nVector y) override;
        Math::nVector funcDx(Math::nVector x, Math::nVector y) override;
    };

    // class L1Cost : public CostFn
    // {
    // public:
    //     double func(double (*f)(double), Math::nVector x, Math::nVector w, double b) override;
    //     double funcDx(double (*f)(double), Math::nVector x, Math::nVector w, double b) override;
    //     double funcDw(double (*f)(double), Math::nVector x, Math::nVector w, double b) override;
    //     double funcDb(double (*f)(double), Math::nVector x, Math::nVector w, double b) override;
    // };

    // class LogCost : public CostFn
    // {
    // public:
    //     double func(double (*f)(double), Math::nVector x, Math::nVector w, double b) override;
    //     double funcDx(double (*f)(double), Math::nVector x, Math::nVector w, double b) override;
    //     double funcDw(double (*f)(double), Math::nVector x, Math::nVector w, double b) override;
    //     double funcDb(double (*f)(double), Math::nVector x, Math::nVector w, double b) override;
    // };
};