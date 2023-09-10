#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include <span>
#include <random>
#include "Matrix.h"
#include "Cost.h"
#include "Activation.h"
#include "Constants.h"

class FFNN
{
private:
    std::vector<int> sizes;
    std::vector<Math::Matrix> layers;
    Cost::CostFn *costFn;
    std::vector<Math::Matrix> weights;
    std::vector<Math::Matrix> bias;
    std::vector<shared_afn_ptr> activationFns;
    
    void printEval(const dataset_span_const &testSpan);
    void evalPair(double &correct, double &cost, const data_pair &tup);

public:
    FFNN(std::vector<int> s, Cost::CostFn *cFn = new Cost::L2Cost(), Activation::ActivationFn *activationFn = new Activation::Sigmoid());
    ~FFNN();

    void addLayer(int size, Activation::ActivationFn *activationFn = new Activation::Sigmoid(), int index = -1);
    void initializeWeightsAndBias(double scaling = 1);
    void print();
    // stochastic mini-batch gradient descent

    void train(dataset &data, int epochs, int miniBatchSize, double learningRate, double split = 0.75);

    void train(dataset &trainingData,
               int epochs, int miniBatchSize, double learningRate,
               const dataset_span_const &testData = dataset());

    void updateMiniBatch(const dataset_span_const &miniBatch, double learningRate);

    Math::Matrix feedForward(Math::Matrix input);

    std::pair<std::vector<Math::Matrix>, std::vector<Math::Matrix>> backPropagate(const Math::Matrix &x, const Math::Matrix &y);

    std::pair<double, double> evaluate(const dataset_span_const &testingData);
    std::pair<double, double> evaluate(const dataset &testingData);

    std::vector<Math::Matrix> getWeights() {return weights;}

    std::vector<Math::Matrix> getBias() {return bias;}

    std::vector<Math::Matrix> getLayers() {return layers;}
};
