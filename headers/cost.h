#pragma once
#include "matrix.h"

namespace Cost
{
    class CostFn
    {
    public:
        virtual double func(Math::Matrix x, Math::Matrix y) = 0;
        virtual Math::Matrix funcDx(Math::Matrix x, Math::Matrix y) = 0;
    };

    class L2Cost : public CostFn
    {
    public:
        double func(Math::Matrix x, Math::Matrix y) override;
        Math::Matrix funcDx(Math::Matrix x, Math::Matrix y) override;
    };

    // class L1Cost : public CostFn
    // {
    // public:
    //     double func(double (*f)(double), Math::Matrix x, Math::Matrix w, double b) override;
    //     double funcDx(double (*f)(double), Math::Matrix x, Math::Matrix w, double b) override;
    //     double funcDw(double (*f)(double), Math::Matrix x, Math::Matrix w, double b) override;
    //     double funcDb(double (*f)(double), Math::Matrix x, Math::Matrix w, double b) override;
    // };

    // class LogCost : public CostFn
    // {
    // public:
    //     double func(double (*f)(double), Math::Matrix x, Math::Matrix w, double b) override;
    //     double funcDx(double (*f)(double), Math::Matrix x, Math::Matrix w, double b) override;
    //     double funcDw(double (*f)(double), Math::Matrix x, Math::Matrix w, double b) override;
    //     double funcDb(double (*f)(double), Math::Matrix x, Math::Matrix w, double b) override;
    // };
};