#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <span>
#include "Matrix.h"
#include "Cost.h"
#include "Activation.h"
#include "FFNN.h"

#define DEBUG true

// minimum two layers, ok it makes sense to initialize weights after everything is added...
// this might be slow but idgaf tbh
FFNN::FFNN(std::vector<int> s, Cost::CostFn *cFn, Activation::ActivationFn *fn)
{
    srand((unsigned)time(0));
    costFn = cFn;
    sizes = s;

    for (int s : sizes)
    {
        layers.push_back(Math::Matrix::generateRandom(s, 1));
    }

    for (std::size_t i = 0; i < s.size(); i++)
    {
        activationFns.push_back(shared_afn_ptr(fn));
    }

    initializeWeightsAndBias();
}

FFNN::~FFNN()
{
    activationFns.clear();
    delete costFn;
}

void FFNN::addLayer(int size, Activation::ActivationFn *activationFn, int index)
{
    // ok ill do this later probably
    if (index < 0)
    {
        layers.push_back(Math::Matrix::generateRandom(size, 1));
        activationFns.push_back(shared_afn_ptr(activationFn));
    }
    else
    {
        layers.insert(layers.begin() + index, Math::Matrix::generateRandom(size, 1));
        activationFns.insert(activationFns.begin() + index, shared_afn_ptr(activationFn));
    }

    initializeWeightsAndBias();
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

// do i really need this?? yeah i do
void FFNN::initializeWeightsAndBias(double scaling)
{
    weights = {};
    bias = {};
    std::size_t len = layers.size();

    for (std::size_t j = 1; j < len; j++)
    {
        int s = layers[j].getSize();
        bias.push_back(Math::Matrix::generateRandom(s, 1, scaling));
    }
    // initialize matrix of weight per layer
    for (std::size_t j = 0; j < len - 1; j++)
    {
        int cols = layers[j].getSize();
        int rows = layers[j + 1].getSize();
        weights.push_back(Math::Matrix::generateRandom(rows, cols, scaling));
    }
}

void FFNN::printEval(const dataset_span_const &testSpan)
{
    auto eval = evaluate(testSpan);
    std::cout << "||   Accuracy of "
              << eval.first
              << ". With cost "
              << eval.second
              << "." << std::endl;
}

// stochastic mini-batch gradient descent
// also tests inputs against testing data for accuracy, but is really slow
void FFNN::train(dataset &trainingData,
                 int epochs, int miniBatchSize, double learningRate,
                 const dataset_span_const &testData)
{
    int nTest = (testData.size() != 0) ? testData.size() : 0;
    int n = trainingData.size();
    int size = ((n % miniBatchSize == 0) ? (n / miniBatchSize) : (n / miniBatchSize + 1));

    dataset_span_const testSpan(testData.begin(), testData.end());

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
    for (int i = 0; i < epochs; i++)
    {
        std::shuffle(trainingData.begin(), trainingData.end(), rng);

        for (int j = 0; j < n; j += miniBatchSize)
        {
            auto end = (j + miniBatchSize < n) ? (trainingData.begin() + j + miniBatchSize) : trainingData.end();
            dataset_span miniBatchSpan(trainingData.begin() + j, end);

            updateMiniBatch(miniBatchSpan, learningRate);
        }

        std::cout << "Epoch " << i << " completed.   ";
        if (nTest != 0)
            printEval(testSpan);

#if DEBUG
        layers[4].print();
#endif
    }
}

void FFNN::train(dataset &data, int epochs, int miniBatchSize, double learningRate, double split)
{
    int size = data.size();
    int trainingSize = std::round(size * split);
    int testingSize = size - trainingSize;
    dataset_span testSpan(data.begin() + trainingSize, data.end());
    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (int i = 0; i < epochs; i++)
    {
        std::shuffle(data.begin(), data.begin() + trainingSize, rng);

        for (int j = 0; j < trainingSize; j += miniBatchSize)
        {
            auto end = (j + miniBatchSize < trainingSize) ? (data.begin() + j + miniBatchSize) : data.begin() + trainingSize;
            dataset_span miniBatchSpan(data.begin() + j, end);

            updateMiniBatch(miniBatchSpan, learningRate);
        }

        std::cout << "Epoch " << i << " completed.   ";
        if (testingSize != 0)
            printEval(testSpan);

#if DEBUG
        print();
#endif
    }
}

// update weights and bias with back propagation
void FFNN::updateMiniBatch(const dataset_span_const &miniBatch, double learningRate)
{
    std::vector<Math::Matrix> db(bias.size());
    std::vector<Math::Matrix> dw(weights.size());

    double c = learningRate / miniBatch.size();

    // for (int i = 0; i < bias.size(); i++)
    //     db[i] = Math::Matrix(bias[i]);

    // for (int i = 0; i < weights.size(); i++)
    //     dw[i] = Math::Matrix(weights[i]);

    for (const data_pair &tup : miniBatch)
    {
        std::pair<std::vector<Math::Matrix>, std::vector<Math::Matrix>> backProp;
        try
        {
            backProp = backPropagate(std::get<0>(tup), std::get<1>(tup));
        }
        catch (const std::exception &e)
        {
            std::cout << ("Exception occurred at line " + std::to_string(__LINE__) + ": ") << e.what() << std::endl;
            std::runtime_error("Back prop sucks");
        }
        std::vector<Math::Matrix> ddb = backProp.first;
        std::vector<Math::Matrix> ddw = backProp.second;

        // for (int i = 0; i < db.size(); i++)
        //     db[i] += ddb[i];

        // for (int i = 0; i < dw.size(); i++)
        //     dw[i] += ddw[i];

        for (std::size_t i = 0; i < db.size(); i++)
        {
            ddb[i] *= c;
            bias[i] -= ddb[i];
        }

        for (std::size_t i = 0; i < dw.size(); i++)
        {
            ddw[i] *= c;
            weights[i] -= ddw[i];
        }
    }

    // for (std::size_t i = 0; i < weights.size(); i++)
    // {
    //     dw[i] *= c;
    //     weights[i] -= dw[i];
    // }

    // for (std::size_t i = 0; i < bias.size(); i++)
    // {
    //     db[i] *= c;
    //     bias[i] -= db[i];
    // }

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
    for (std::size_t i = 0; i < weights.size(); i++)
    {
        input = activationFns[i + 1]->fn((weights[i] * input) + bias[i]);
    }
    return input;
}

std::pair<std::vector<Math::Matrix>, std::vector<Math::Matrix>> FFNN::backPropagate(const Math::Matrix &x, const Math::Matrix &y)
{
    // do i really need this? or can i directly add to the weights
    std::vector<Math::Matrix> db(bias.size());
    std::vector<Math::Matrix> dw(weights.size());

    for (std::size_t i = 0; i < bias.size(); i++)
        db[i] = Math::Matrix(bias[i]);

    for (std::size_t i = 0; i < weights.size(); i++)
        dw[i] = Math::Matrix(weights[i]);

    Math::Matrix activation = x;
    layers[0] = x;

    std::vector<Math::Matrix> zVals(bias.size());

    // propagate the values forward starting at x, and keep track of each layer
    for (std::size_t i = 0; i < bias.size(); i++)
    {
        Math::Matrix z = weights[i] * activation + bias[i];
        zVals[i] = z;
        activation = activationFns[i + 1]->fn(z);
        layers[i + 1] = activation;
    }

    Math::Matrix delta = Math::Matrix::hProd(costFn->funcDx(layers[layers.size() - 1], y), activationFns[layers.size() - 1]->fnDerv(zVals[zVals.size() - 1]));

    db[db.size() - 1] = delta;
    dw[dw.size() - 1] = Math::Matrix::oProd(delta, layers[layers.size() - 2]);

    for (std::size_t l = 2; l < layers.size(); l++)
    {
        Math::Matrix z = zVals[zVals.size() - l];

        delta = Math::Matrix::hProd(Math::Matrix::iProd(weights[weights.size() - l + 1], delta), activationFns[layers.size() - l]->fnDerv(z));
        db[db.size() - l] = delta;
        dw[dw.size() - l] = Math::Matrix::oProd(delta, layers[layers.size() - l - 1]);
    }

    return std::make_pair(db, dw);
}

std::pair<double, double> FFNN::evaluate(const dataset_span_const &testingData)
{
    double correct = 0;
    double cost = 0;

    for (const data_pair &tup : testingData)
    {
        evalPair(correct, cost, tup);
    }
    return std::pair<double, double>(correct / testingData.size(), cost);
}

std::pair<double, double> FFNN::evaluate(const dataset &testingData)
{
    double correct = 0;
    double cost = 0;

    for (const data_pair &tup : testingData)
    {
        evalPair(correct, cost, tup);
    }
    return std::pair<double, double>(correct / testingData.size(), cost);
}

void FFNN::evalPair(double &correct, double &cost, const data_pair &tup)
{
    Math::Matrix output = feedForward(tup.first);
    Math::Matrix actual = tup.second;

    cost += costFn->func(actual, output);

    std::vector<double> ov = output.getVals();
    std::vector<double> av = actual.getVals();

    int outputMax = std::distance(ov.begin(), std::max_element(ov.begin(), ov.end()));
    int actualMax = std::distance(av.begin(), std::max_element(av.begin(), av.end()));

    if (outputMax == actualMax)
        correct++;
}
