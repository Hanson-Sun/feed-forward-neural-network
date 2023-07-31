#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <tuple>
#include <algorithm>
#include <span>
#include "headers/matrix.h"
#include "headers/cost.h"
#include "headers/activation.h"
#include "headers/nn.h"

#define DEBUG false

// minimum two layers, ok it makes sense to initialize weights after everything is added...
// this might be slow but idgaf tbh
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

    for (int i = 0; i < s.size(); i++)
    {
        activationFns.push_back(fn);
    }

    initializeWeightsAndBias();
}

FFNN::~FFNN()
{
    // delete...
    // for (Activation::ActivationFn* f : activationFns)
    //     delete f;

    activationFns.clear();
    delete activationFn;
    delete costFn;
}

void FFNN::print()
{
    std::cout << "====== Printing Layers ======" << std::endl;
    for (auto b : layers)
    {
        std::cout << "layer-print " << std::endl;
        b.print();
    }

    std::cout << "====== Printing Biases ======" << std::endl;
    for (auto v : bias)
    {
        std::cout << "b-print " << std::endl;
        v.print();
    }

    std::cout << "====== Printing Weights ======" << std::endl;
    for (auto v : weights)
    {
        std::cout << "w-print " << std::endl;
        v.print();
    }

    std::cout << "====== END ======" << std::endl;
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

void FFNN::addLayer(int size, Activation::ActivationFn *activationFn, int index)
{
    // ok ill do this later probably
    if (index < 0)
    {
        layers.push_back(Math::Matrix::generateRandom(size, 1, 1));
        activationFns.push_back(activationFn);
    }
    else
    {
        layers.insert(layers.begin() + index, Math::Matrix::generateRandom(size, 1, 1));
        activationFns.insert(activationFns.begin() + index, activationFn);
    }

    initializeWeightsAndBias();
}

// do i really need this?? yeah i do
void FFNN::initializeWeightsAndBias(double scaling)
{
    weights = {};
    bias = {};
    int len = layers.size();

    for (int j = 1; j < len; j++)
    {
        int s = layers[j].getSize();
        bias.push_back(Math::Matrix::generateRandom(s, 1, scaling));
    }
    // initialize matrix of weight per layer
    for (int j = 0; j < len - 1; j++)
    {
        int cols = layers[j].getSize();
        int rows = layers[j + 1].getSize();
        weights.push_back(Math::Matrix::generateRandom(rows, cols, scaling));
    }
}

// stochastic mini-batch gradient descent
// also tests inputs against testing data for accuracy, but is really slow
void FFNN::train(std::vector<std::tuple<Math::Matrix, Math::Matrix>> &trainingData,
                 int epochs, int miniBatchSize, double learningRate,
                 const std::vector<std::tuple<Math::Matrix, Math::Matrix>> &testData)
{
    int nTest = (testData.size() != 0) ? testData.size() : 0;
    int n = trainingData.size();
    int size = ((n % miniBatchSize == 0) ? (n / miniBatchSize) : (n / miniBatchSize + 1));

    auto rng = std::default_random_engine{};
    for (int i = 0; i < epochs; i++)
    {
        // random_unique(trainingData.begin(), trainingData.end(), miniBatchSize);

        std::shuffle(trainingData.begin(), trainingData.end(), rng);

        for (int j = 0; j < n; j += miniBatchSize)
        {
            auto end = (j + miniBatchSize < n) ? (trainingData.begin() + j + miniBatchSize) : trainingData.end();
            std::span<std::tuple<Math::Matrix, Math::Matrix>> miniBatchSpan = std::span<std::tuple<Math::Matrix, Math::Matrix>>(trainingData.begin() + j, end);

            updateMiniBatch(miniBatchSpan, learningRate);
        }

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

#if DEBUG
        for (auto b : layers)
        {
            std::cout << "layer-print " << std::endl;
            b.print();
        }

        for (auto v : bias)
        {
            std::cout << "b-print " << std::endl;
            v.print();
        }

        for (auto v : weights)
        {
            std::cout << "w-print " << std::endl;
            v.print();
        }
#endif
    }
}

// update weights and bias with back propagation
void FFNN::updateMiniBatch(const std::span<std::tuple<Math::Matrix, Math::Matrix>> &miniBatch, double learningRate)
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
        std::tuple<std::vector<Math::Matrix>, std::vector<Math::Matrix>> backProp;
        try
        {
            backProp = backPropagate(std::get<0>(tup), std::get<1>(tup));
        }
        catch (const std::exception &e)
        {
            std::cout << ("Exception occurred at line " + std::to_string(__LINE__) + ": ") << e.what() << std::endl;
            std::runtime_error("Back prop sucks");
        }
        std::vector<Math::Matrix> ddb = std::get<0>(backProp);
        std::vector<Math::Matrix> ddw = std::get<1>(backProp);

        for (int i = 0; i < db.size(); i++)
            db[i] += ddb[i];

        for (int i = 0; i < dw.size(); i++)
            dw[i] += ddw[i];
    }

    for (int i = 0; i < weights.size(); i++)
    {
        dw[i] *= c;
        weights[i] -= dw[i];
    }

    for (int i = 0; i < bias.size(); i++)
    {
        db[i] *= c;
        bias[i] -= db[i];
    }

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
        input = activationFns[i + 1]->fn((weights[i] * input) + bias[i]);
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

    std::vector<Math::Matrix> zVals(bias.size());

    // propagate the values forward starting at x, and keep track of each layer
    for (int i = 0; i < bias.size(); i++)
    {
        Math::Matrix z = weights[i] * activation + bias[i];
        zVals[i] = z;
        activation = activationFns[i + 1]->fn(z);
        layers[i + 1] = activation;
    }

    Math::Matrix delta = Math::Matrix::hProd(costFn->funcDx(layers[layers.size() - 1], y), activationFns[layers.size() - 1]->fnDerv(zVals[zVals.size() - 1]));

    db[db.size() - 1] = delta;
    dw[dw.size() - 1] = Math::Matrix::oProd(delta, layers[layers.size() - 2]);

    for (int l = 2; l < layers.size(); l++)
    {
        Math::Matrix z = zVals[zVals.size() - l];

        delta = Math::Matrix::hProd(Math::Matrix::iProd(weights[weights.size() - l + 1], delta), activationFns[layers.size() - l]->fnDerv(z));
        db[db.size() - l] = delta;
        dw[dw.size() - l] = Math::Matrix::oProd(delta, layers[layers.size() - l - 1]);
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
