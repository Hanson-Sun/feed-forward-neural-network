#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include <span>
#include "matrix.h"
#include "cost.h"
#include "activation.h"

class FFNN
{
private:
    std::vector<int> sizes;
    std::vector<Math::Matrix> layers;
    Cost::CostFn *costFn;
    std::vector<Math::Matrix> weights;
    std::vector<Math::Matrix> bias;
    std::vector<Activation::ActivationFn*> activationFns;
    Activation::ActivationFn *activationFn;

public:
    FFNN(std::vector<int> s, Cost::CostFn *cFn = new Cost::L2Cost(), Activation::ActivationFn *activationFn = new Activation::Sigmoid());
    ~FFNN();

    void addLayer(int size, Activation::ActivationFn *activationFn = new Activation::Sigmoid(), int index = -1);
    void initializeWeightsAndBias(double scaling = 1);
    void print();
    // stochastic mini-batch gradient descent
    void train(std::vector<std::tuple<Math::Matrix, Math::Matrix>> &trainingData,
               int epochs, int miniBatchSize, double learningRate,
               const std::vector<std::tuple<Math::Matrix, Math::Matrix>> &testData = std::vector<std::tuple<Math::Matrix, Math::Matrix>>());

    void updateMiniBatch(const std::span<std::tuple<Math::Matrix, Math::Matrix>> &miniBatch, double learningRate);

    Math::Matrix feedForward(Math::Matrix input);

    std::tuple<std::vector<Math::Matrix>, std::vector<Math::Matrix>> backPropagate(const Math::Matrix &x, const Math::Matrix &y);

    double evaluate(const std::vector<std::tuple<Math::Matrix, Math::Matrix>> &testingData);

    double evaluateCost(const std::vector<std::tuple<Math::Matrix, Math::Matrix>> &testingData);

    std::vector<Math::Matrix> getWeights()
    {
        return weights;
    }

    std::vector<Math::Matrix> getBias()
    {
        return bias;
    }

    std::vector<Math::Matrix> getLayers()
    {
        return layers;
    }
};

