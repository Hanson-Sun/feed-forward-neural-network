#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include "vector.h"
#include "matrix.h"
#include "cost.h"
#include "activation.h"

class FFNN
{
private:
    std::vector<int> sizes;
    std::vector<Math::nVector> layers;
    Cost::CostFn *costFn;
    std::vector<Math::Matrix> weights;
    std::vector<Math::nVector> bias;
    Activation::ActivationFn *activationFn;

public:
    FFNN(std::vector<int> s, Cost::CostFn *cFn = new Cost::L2Cost(), Activation::ActivationFn *activationFn = new Activation::Sigmoid());
    // stochastic mini-batch gradient descent
    void train(std::vector<std::tuple<Math::nVector, Math::nVector>> &trainingData,
               int epochs, int miniBatchSize, double learningRate,
               const std::vector<std::tuple<Math::nVector, Math::nVector>> &testData = std::vector<std::tuple<Math::nVector, Math::nVector>>());

    void updateMiniBatch(const std::vector<std::tuple<Math::nVector, Math::nVector>> &miniBatch, double learningRate);
    Math::nVector feedForward(Math::nVector input);
    std::tuple<std::vector<Math::nVector>, std::vector<Math::Matrix>> backPropagate(const Math::nVector &x, const Math::nVector &y);

    double evaluate(const std::vector<std::tuple<Math::nVector, Math::nVector>> &testingData);

    std::vector<Math::Matrix> getWeights()
    {
        return weights;
    }

    std::vector<Math::nVector> getBias()
    {
        return bias;
    }

    std::vector<Math::nVector> getLayers()
    {
        return layers;
    }
};

