#include <iostream>
#include <tuple>
#include "Matrix.h"
#include "FFNN.h"
#include "Read.h"
#include "Constants.h"

dataset generateData(int len)
{
    std::vector<std::pair<Math::Matrix, Math::Matrix>> data;
    for (int i = 0; i < len; i++)
    {
        double x = std::abs(static_cast<double>(rand()) / RAND_MAX);
        double y = std::abs(static_cast<double>(rand()) / RAND_MAX);
        double z = (x) * (x) + (y) * (y)-0.7;
        Math::Matrix input = Math::Matrix(std::vector<double>{x, y});
        Math::Matrix output = (z <= 0) ? Math::Matrix(std::vector<double>{1, 0}) : Math::Matrix(std::vector<double>{0, 1});
        data.push_back(std::make_pair(input, output));
    }

    return data;
}

dataset generateQuarterCircleData(int len)
{
    std::vector<std::pair<Math::Matrix, Math::Matrix>> data;
    for (int i = 0; i < len; i++)
    {
        double x = std::abs(static_cast<double>(rand()) / RAND_MAX) * 10;
        double y = std::abs(static_cast<double>(rand()) / RAND_MAX) * 10;
        double z = (x) * (x) + (y) * (y);
        Math::Matrix input = Math::Matrix(std::vector<double>{x, y});
        Math::Matrix output = (z <= 6.5 * 6.5) ? Math::Matrix(std::vector<double>{1, 0}) : Math::Matrix(std::vector<double>{0, 1});
        data.push_back(std::make_pair(input, output));
    }

    return data;
}

dataset generateLinearData(int len)
{
    std::vector<std::pair<Math::Matrix, Math::Matrix>> data;
    for (int i = 0; i < len; i++)
    {
        double x = static_cast<double>(rand()) / RAND_MAX * 20 - 10;
        double y = static_cast<double>(rand()) / RAND_MAX * 20 - 10;
        double z = -3 * x + 2 * y + 2;
        Math::Matrix input = Math::Matrix(std::vector<double>{x, y});
        // Math::Matrix output = (z > 50) ? Math::Matrix(std::vector<double>{2, -2}) : Math::Matrix(std::vector<double>{-2, 2});
        // Math::Matrix output = (z > 10) ? Math::Matrix(std::vector<double>{1}) : Math::Matrix(std::vector<double>{-1});
        Math::Matrix output = Math::Matrix(std::vector<double>{z});
        data.push_back(std::make_pair(input, output));
    }

    return data;
}

dataset generateLinearClass(int len)
{
    std::vector<std::pair<Math::Matrix, Math::Matrix>> data;
    for (int i = 0; i < len; i++)
    {
        double x = static_cast<double>(rand()) / RAND_MAX * 40 - 20;
        double y = static_cast<double>(rand()) / RAND_MAX * 40 - 20;
        double z = x + y;
        Math::Matrix input = Math::Matrix(std::vector<double>{x, y});
        Math::Matrix output = (z < 10) ? Math::Matrix(std::vector<double>{1, -1}) : Math::Matrix(std::vector<double>{-1, 1});

        data.push_back(std::make_pair(input, output));
    }

    return data;
}

int main()
{

    // Math::Matrix::oProd(Math::Matrix(std::vector<double>{-0.000417, 0.000042, -0.000664, -0.000007}),
    //                     Math::Matrix(std::vector<double>{3.749168, -0.079840, -0.165974, 4.228167})).print();

#if false
    auto fn = new Cost::CrossEntropy();

    std::cout << fn->func(Math::Matrix(std::vector<double>{0.5, 0.5}), Math::Matrix(std::vector<double>{1, 1})) << std::endl;
    std::cout << fn->func(Math::Matrix(std::vector<double>{-0.5, 0.5}), Math::Matrix(std::vector<double>{1, 1})) << std::endl;

    std::cout << fn->funcDx(Math::Matrix(std::vector<double>{0.5, 0.5}), Math::Matrix(std::vector<double>{1, 1})) << std::endl;
    std::cout << fn->funcDx(Math::Matrix(std::vector<double>{-0.5, 0.5}), Math::Matrix(std::vector<double>{1, 1})) << std::endl;
    std::cout << fn->funcDx(Math::Matrix(std::vector<double>{2, -0.5}), Math::Matrix(std::vector<double>{1, 1})) << std::endl;

#endif


// data visualization fr.
#if false

    dataset training = readData("data/mnist_train.csv", 1);
    auto data = training[0];
    auto input = data.first;
    auto output = data.first;

    output.print();

    for (int i = 0; i < input.getVals().size(); i++)
    {
        if (input[i] < 10)
            std::cout << input[i] << "  ";
        else if (input[i] > 10 && input[i] < 100)
            std::cout << input[i] << " ";
        else
            std::cout << input[i];
        if ((i + 1) % 28 == 0)
            std::cout << std::endl;
    }

#endif
#if false
    {
        dataset training = readData("C:\\Users\\docto\\Documents\\GitHub\\feed-forward-neural-network\\data\\mnist_test.csv", 10000);

        std::cout << std::thread::hardware_concurrency() << std::endl;

        FFNN nn(std::vector<int>{784, 128, 64, 32},new Cost::L2Cost(), new Activation::LeakyRelu());
        // FFNN nn(std::vector<int>{784, 32}, new Cost::L2Cost(), new Activation::LeakyRelu());
        nn.addLayer(10, new Activation::Sigmoid());

        //nn.print();

        nn.train(training, 300, 2500, 1, 0.90);

        std::cout << nn.evaluate(training).first << ", " << nn.evaluate(training).second << std::endl;
    }
#endif

#if false
    {
        FFNN nn(std::vector<int>{2, 1, 1, 1}, new Cost::L2Cost(), new Activation::Sigmoid());
        dataset training = generateLinearData(1000);
        dataset testing = generateLinearData(1);

        nn.train(training, 250, 1000, 0.2, testing);

        std::cout << nn.evaluate(testing).first << std::endl;
        nn.print();
    }
#endif

#if false
    {
        FFNN nn(std::vector<int>{2, 10}, new Cost::CrossEntropy(), new Activation::LeakyRelu());
        nn.addLayer(2, new Activation::Sigmoid());
        dataset training = generateLinearClass(5000);
        dataset testing = generateLinearClass(200);

        nn.train(training, 250, 1000, 0.05, testing);

        std::cout << nn.evaluate(testing).first << ", " << nn.evaluate(testing).second << std::endl;
        nn.print();
    }
#endif

#if true
    {

        FFNN nn(std::vector<int>{2, 4, 4, 4, 3}, new Cost::L2Cost(), new Activation::LeakyRelu());
        nn.addLayer(2, new Activation::Sigmoid());

        //nn.print();
        // nn.addLayer(4, new Activation::LeakyRelu(), 0);
        nn.print();

        // std::vector<std::tuple<Math::Matrix, Math::Matrix>> training = generateData(2000);
        // std::vector<std::tuple<Math::Matrix, Math::Matrix>> testing = generateData(300);
        dataset training = generateQuarterCircleData(2000);
        dataset testing = generateQuarterCircleData(500);

        nn.train(training, 300, 1000, 2, testing);

        std::cout << nn.evaluate(testing).first << ", " << nn.evaluate(testing).second << std::endl;
    }
#endif

#if false
    Math::Matrix m = Math::Matrix::generateRandom(5, 3);
    Math::Matrix m2 = Math::Matrix::generateRandom(5, 3);
    m.print();
    m2.print();

    m -= m2;
    m.print();
    //m.transpose().print();
    std::cout << std::endl;
    Math::Matrix v1 = (Math::Matrix({3, 4, 5}));
    Math::Matrix v2 = (Math::Matrix({1, 2, 3}));

    v1.print();
    v2.print();
    Math::Matrix::outerProduct(v1, v2).print();


    //Math::Matrix::outerProduct(Math::Matrix(std::vector<double>{1, 2}), Math::Matrix(std::vector<double>{3, 4})).print();
    //Math::Matrix::pairWiseMult(Math::Matrix(std::vector<double>{1, 2}), Math::Matrix(std::vector<double>{3, 4})).print();

#endif

#if false
    FFNN nn(std::vector<int>{2, 4, 2},new Cost::L2Cost(), new Activation::Sigmoid());

    auto output = nn.feedForward(Math::Matrix(std::vector<double>{2.0,3.0}));

    for (auto b : nn.getWeights())
    {
        std::cout << "layer-print " << std::endl;
        b.print();
    }

    output.print();

    // (m3 * vect1).print();
#endif

#if false
    FFNN nn(std::vector<int>{2, 4, 2},new Cost::L2Cost(), new Activation::Sigmoid());
    auto t = nn.backPropagate(Math::Matrix(std::vector<double>{1, 1}), Math::Matrix(std::vector<double>{2, 2}));
    auto db = std::get<0>(t);
    auto dw = std::get<1>(t);

    for (auto b : nn.getLayers())
    {
        std::cout << "layer-print ";
        b.print();
    }

    for (auto v : db)
    {
        std::cout << "db-print ";
        v.print();
    }

    for (auto v : dw)
    {
        std::cout << "dw-print ";
        v.print();
    }
    // (m3 * vect1).print();
#endif

#if false
    Math::Matrix M1({{1, 2}, {4, 5}, {7, 8}});
    Math::Matrix M2({{1, 2, 3}, {4, 5, 6}});

    Math::Matrix M3 = Math::Matrix(512, 512, 2);
    Math::Matrix M4 = Math::Matrix(512, 1, 2);

    Math::Matrix M5 = Math::Matrix::generateRandom(1024, 1024);
    Math::Matrix M6 = Math::Matrix::generateRandom(1024, 1024);

    Math::Matrix sum(1024, 1024, 0);

    int n = 10000;

    std::cout << "begin multiplication" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {
        M1 * M2;
        M3 * M3;
        M3 * M4;
        sum += M5 * M6;
        std::cout << i << " finished" << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << (duration.count() / n) << std::endl;
    std::cout << "end" << std::endl;
#endif
    return 0;
}
