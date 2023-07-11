#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <tuple>
#include <algorithm>
#include "headers/vector.h"
#include "headers/matrix.h"
#include "headers/cost.h"
#include "headers/activation.h"
#include "headers/nn.h"

FFNN::FFNN(std::vector<int> s, Cost::CostFn *cFn, Activation::ActivationFn *fn)
{
    srand((unsigned)time(0));
    costFn = cFn;
    sizes = s;
    int len = s.size();
    activationFn = fn;

    for (int s : sizes)
    {
        layers.push_back(Math::nVector::generateRandom(s));
    }

    // initialize a vector of bias vectors per layer
    // output from input layer is unbiased
    for (int j = 1; j < len; j++)
    {
        int s = sizes[j];
        bias.push_back(Math::nVector::generateRandom(s));
    }
    // initialize matrix of weight per layer
    for (int j = 0; j < len - 1; j++)
    {
        int rowSize = sizes[j];
        int rowNum = sizes[j + 1];
        weights.push_back(Math::Matrix::generateRandom(rowSize, rowNum));
    }
}

// stochastic mini-batch gradient descent
// also tests inputs against testing data for accuracy, but is really slow
void FFNN::train(std::vector<std::tuple<Math::nVector, Math::nVector>> &trainingData,
                 int epochs, int miniBatchSize, double learningRate,
                 const std::vector<std::tuple<Math::nVector, Math::nVector>> &testData)
{
    int nTest = (testData.size() != 0) ? testData.size() : 0;
    int n = trainingData.size();

    auto rng = std::default_random_engine{};
    for (int i = 0; i < epochs; i++)
    {
        std::shuffle(trainingData.begin(), trainingData.end(), rng);
        std::vector<std::vector<std::tuple<Math::nVector, Math::nVector>>> miniBatches;
        for (int j = 0; j < n; j += miniBatchSize)
        {
            auto end = (j + miniBatchSize < n) ? (trainingData.begin() + j + miniBatchSize) : trainingData.end();
            std::vector<std::tuple<Math::nVector, Math::nVector>> miniBatch(trainingData.begin() + j, end);
            miniBatches.push_back(miniBatch);
        }

        for (std::vector<std::tuple<Math::nVector, Math::nVector>> mb : miniBatches)
            updateMiniBatch(mb, learningRate);

        if (nTest != 0)
        {
            std::cout << "Epoch " << i << " completed.  ||  "
                      << "Accuracy  of "
                      << evaluate(testData)
                      << "." << std::endl;
        }
        else
        {
            std::cout << "Epoch " << i << " completed." << std::endl;
        }
        // for (auto b : layers)
        // {
        //     std::cout << "layer-print ";
        //     b.print();
        // }

        // for (auto v : bias)
        // {
        //     std::cout << "db-print ";
        //     v.print();
        // }

        // for (auto v : weights)
        // {
        //     std::cout << "dw-print ";
        //     v.print();
        // }
    }
}

// update weights and bias with back propagation
void FFNN::updateMiniBatch(const std::vector<std::tuple<Math::nVector, Math::nVector>> &miniBatch, double learningRate)
{
    std::vector<Math::nVector> db;
    std::vector<Math::Matrix> dw;

    double c = learningRate / miniBatch.size();

    for (Math::nVector b : bias)
        db.push_back(Math::nVector::generateEmptyCopy(b));

    for (Math::Matrix w : weights)
        dw.push_back(Math::Matrix::generateEmptyCopy(w));

    for (std::tuple<Math::nVector, Math::nVector> tup : miniBatch)
    {
        std::tuple<std::vector<Math::nVector>, std::vector<Math::Matrix>> backProp = backPropagate(std::get<0>(tup), std::get<1>(tup));
        std::vector<Math::nVector> ddb = std::get<0>(backProp);
        std::vector<Math::Matrix> ddw = std::get<1>(backProp);

        for (int i = 0; i < db.size(); i++)
            db[i] += ddb[i];

        for (int i = 0; i < dw.size(); i++)
            dw[i] += ddw[i];
    }

    for (int i = 0; i < weights.size(); i++)
    {
        weights[i] -= dw[i] * c;
    }

    for (int i = 0; i < bias.size(); i++)
    {
        bias[i] -= db[i] * c;
    }
}

Math::nVector FFNN::feedForward(Math::nVector input)
{
    // returns network output for input
    for (int i = 0; i < weights.size(); i++)
    {
        auto b = bias[i];
        auto w = weights[i];
        input = (w * input) + b;
    }
    return input;
}

std::tuple<std::vector<Math::nVector>, std::vector<Math::Matrix>> FFNN::backPropagate(const Math::nVector &x, const Math::nVector &y)
{
    std::vector<Math::nVector> db;
    std::vector<Math::Matrix> dw;

    for (Math::nVector b : bias)
        db.push_back(Math::nVector::generateEmptyCopy(b));

    for (Math::Matrix w : weights)
        dw.push_back(Math::Matrix::generateEmptyCopy(w));

    Math::nVector activation = x;
    layers[0] = x;

    std::vector<Math::nVector> zVals;

    // propagate the values forward starting at x, and keep track of each layer
    for (int i = 0; i < bias.size(); i++)
    {
        Math::Matrix w = weights[i];
        Math::nVector b = bias[i];
        Math::nVector z = w * activation + b;
        zVals.push_back(z);
        activation = activationFn->fn(z);
        layers[i + 1] = activation;
    }

    Math::nVector delta = Math::nVector::pairWiseMult(costFn->funcDx(layers[layers.size() - 1], y), activationFn->fnDerv(zVals[zVals.size() - 1]));

    db[db.size() - 1] = delta;
    dw[dw.size() - 1] = Math::Matrix::outerProduct(delta, layers[layers.size() - 2]);

    for (int l = 2; l < layers.size(); l++)
    {
        Math::nVector z = zVals[zVals.size() - l];
        Math::nVector activeDeriv = activationFn->fnDerv(z);
        // bro why do i need to implement all these stupid math libraries...
        delta = Math::nVector::pairWiseMult(weights[weights.size() - l + 1].transpose() * delta, activeDeriv);
        db[db.size() - l] = delta;
        dw[dw.size() - l] = Math::Matrix::outerProduct(delta, layers[layers.size() - l - 1]);
    }

    return std::make_tuple(db, dw);
}

double FFNN::evaluate(const std::vector<std::tuple<Math::nVector, Math::nVector>> &testingData)
{
    double correct = 0;
    for (std::tuple<Math::nVector, Math::nVector> tup : testingData)
    {
        Math::nVector output = feedForward(std::get<0>(tup));
        Math::nVector actual = std::get<1>(tup);

        std::vector<double> ov = output.getVals();
        std::vector<double> av = actual.getVals();

        int outputMax = std::distance(ov.begin(), std::max_element(ov.begin(), ov.end()));
        int actualMax = std::distance(av.begin(), std::max_element(av.begin(), av.end()));

        if (outputMax == actualMax)
            correct++;
    }
    return correct / testingData.size();
}
