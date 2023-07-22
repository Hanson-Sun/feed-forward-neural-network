#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <tuple>
#include <algorithm>
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
        layers.push_back(Math::Matrix::generateRandom(s, 1, 1));
    }

    // initialize a vector of bias vectors per layer
    // output from input layer is unbiased
    for (int j = 1; j < len; j++)
    {
        int s = sizes[j];
        bias.push_back(Math::Matrix::generateRandom(s, 1, 1));
    }
    // initialize matrix of weight per layer
    for (int j = 0; j < len - 1; j++)
    {
        int cols = sizes[j];
        int rows = sizes[j + 1];
        weights.push_back(Math::Matrix::generateRandom(rows, cols, 1));
    }
}

// thanks stackoverflow
template <class BidiIter>
BidiIter random_unique(BidiIter begin, BidiIter end, size_t num_random)
{
    size_t left = std::distance(begin, end);
    while (num_random--)
    {
        BidiIter r = begin;
        std::advance(r, rand() % left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

// stochastic mini-batch gradient descent
// also tests inputs against testing data for accuracy, but is really slow
void FFNN::train(std::vector<std::tuple<Math::Matrix, Math::Matrix>> &trainingData,
                 int epochs, int miniBatchSize, double learningRate,
                 const std::vector<std::tuple<Math::Matrix, Math::Matrix>> &testData)
{
    int nTest = (testData.size() != 0) ? testData.size() : 0;
    int n = trainingData.size();
    //int size = ((n % miniBatchSize == 0) ? (n / miniBatchSize) : (n / miniBatchSize + 1));

    auto rng = std::default_random_engine{};
    for (int i = 0; i < epochs; i++)
    {

        random_unique(trainingData.begin(), trainingData.end(), miniBatchSize);
        std::vector<std::tuple<Math::Matrix, Math::Matrix>> miniBatch(trainingData.begin(), trainingData.begin() + miniBatchSize);

        updateMiniBatch(miniBatch, learningRate);

        // std::shuffle(trainingData.begin(), trainingData.end(), rng);
        // std::vector<std::vector<std::tuple<Math::Matrix, Math::Matrix>>> miniBatches(size);
        // int count = 0;
        // for (int j = 0; j < n; j += miniBatchSize)
        // {
        //     auto end = (j + miniBatchSize < n) ? (trainingData.begin() + j + miniBatchSize) : trainingData.end();
        //     std::vector<std::tuple<Math::Matrix, Math::Matrix>> miniBatch(trainingData.begin() + j, end);
        //     miniBatches[count] = miniBatch;
        //     count++;
        // }

        // for (const std::vector<std::tuple<Math::Matrix, Math::Matrix>> &mb : miniBatches)
        //     updateMiniBatch(mb, learningRate);

        if (nTest != 0)
        {
            std::cout << "Epoch " << i << " completed.  ||  "
                      << "Accuracy of "
                      << evaluate(testData)
                      << ". With cost "
                      << evaluateCost(testData)
                      << "." << std::endl;
        }
        else
        {
            std::cout << "Epoch " << i << " completed." << std::endl;
        }

        // for (auto b : layers)
        // {
        //     std::cout << "layer-print " << std::endl;
        //     b.print();
        // }

        // for (auto v : bias)
        // {
        //     std::cout << "db-print " << std::endl;
        //     v.print();
        // }

        // for (auto v : weights)
        // {
        //     std::cout << "dw-print " << std::endl;
        //     v.print();
        // }
    }
}

// update weights and bias with back propagation
void FFNN::updateMiniBatch(const std::vector<std::tuple<Math::Matrix, Math::Matrix>> &miniBatch, double learningRate)
{
    std::vector<Math::Matrix> db(bias.size());
    std::vector<Math::Matrix> dw(weights.size());

    double c = learningRate / miniBatch.size();

    for (int i = 0; i < bias.size(); i++)
        db[i] = Math::Matrix(bias[i]);

    for (int i = 0; i < weights.size(); i++)
        dw[i] = Math::Matrix(weights[i]);

    for (const std::tuple<Math::Matrix, Math::Matrix> &tup : miniBatch)
    {
        std::tuple<std::vector<Math::Matrix>, std::vector<Math::Matrix>> backProp = backPropagate(std::get<0>(tup), std::get<1>(tup));
        std::vector<Math::Matrix> ddb = std::get<0>(backProp);
        std::vector<Math::Matrix> ddw = std::get<1>(backProp);

        for (int i = 0; i < db.size(); i++)
            db[i] += ddb[i];

        for (int i = 0; i < dw.size(); i++)
            dw[i] += ddw[i];
    }

    for (int i = 0; i < weights.size(); i++)
        weights[i] -= dw[i] * c;

    for (int i = 0; i < bias.size(); i++)
        bias[i] -= db[i] * c;

    // std::cout << "c values " << c << std::endl;
    // for (auto v : dw)
    // {
    //     std::cout << "dw-print " << std::endl;
    //     v.print();
    // }
}

Math::Matrix FFNN::feedForward(Math::Matrix input)
{
    // returns network output for input
    for (int i = 0; i < weights.size(); i++)
    {
        input = activationFn->fn((weights[i] * input) + bias[i]);
    }
    return input;
}

std::tuple<std::vector<Math::Matrix>, std::vector<Math::Matrix>> FFNN::backPropagate(const Math::Matrix &x, const Math::Matrix &y)
{
    std::vector<Math::Matrix> db(bias.size());
    std::vector<Math::Matrix> dw(weights.size());

    for (int i = 0; i < bias.size(); i++)
        db[i] = Math::Matrix(bias[i]);

    for (int i = 0; i < weights.size(); i++)
        dw[i] = Math::Matrix(weights[i]);

    Math::Matrix activation = x;
    layers[0] = x;
    // std::vector<Math::Matrix> activations{x};

    std::vector<Math::Matrix> zVals;

    // propagate the values forward starting at x, and keep track of each layer
    for (int i = 0; i < bias.size(); i++)
    {
        Math::Matrix z = weights[i] * activation + bias[i];
        zVals.push_back(z);
        activation = activationFn->fn(z);
        layers[i + 1] = activation;
        // activations.push_back(activation);
    }

    Math::Matrix delta = Math::Matrix::hProd(costFn->funcDx(layers[layers.size() - 1], y), activationFn->fnDerv(zVals[zVals.size() - 1]));
    // Math::Matrix delta = Math::Matrix::hProd(costFn->funcDx(activations[activations.size() - 1], y), activationFn->fnDerv(zVals[zVals.size() - 1]));

    db[db.size() - 1] = delta;
    dw[dw.size() - 1] = Math::Matrix::oProd(delta, layers[layers.size() - 2]);
    // dw[dw.size() - 1] = Math::Matrix::oProd(delta, activations[activations.size() - 2]);

    for (int l = 2; l < layers.size(); l++)
    {
        Math::Matrix z = zVals[zVals.size() - l];
        delta = Math::Matrix::hProd(Math::Matrix::iProd(weights[weights.size() - l + 1], delta), activationFn->fnDerv(z));
        db[db.size() - l] = delta;
        dw[dw.size() - l] = Math::Matrix::oProd(delta, layers[layers.size() - l - 1]);
        // dw[dw.size() - l] = Math::Matrix::oProd(delta, activations[activations.size() - l - 1]);
    }

    return std::make_tuple(db, dw);
}

double FFNN::evaluate(const std::vector<std::tuple<Math::Matrix, Math::Matrix>> &testingData)
{
    double correct = 0;
    for (std::tuple<Math::Matrix, Math::Matrix> tup : testingData)
    {
        Math::Matrix output = feedForward(std::get<0>(tup));
        Math::Matrix actual = std::get<1>(tup);

        // output.print();
        // actual.print();

        std::vector<double> ov = output.getVals();
        std::vector<double> av = actual.getVals();

        int outputMax = std::distance(ov.begin(), std::max_element(ov.begin(), ov.end()));
        int actualMax = std::distance(av.begin(), std::max_element(av.begin(), av.end()));

        if (outputMax == actualMax)
            correct++;
    }
    return correct / testingData.size();
}

double FFNN::evaluateCost(const std::vector<std::tuple<Math::Matrix, Math::Matrix>> &testingData)
{
    double cost = 0;
    for (std::tuple<Math::Matrix, Math::Matrix> tup : testingData)
    {
        Math::Matrix output = feedForward(std::get<0>(tup));
        Math::Matrix actual = std::get<1>(tup);
        cost += costFn->func(actual, output);
    }
    return cost;
}
