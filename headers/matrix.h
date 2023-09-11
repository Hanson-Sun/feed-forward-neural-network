#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <functional>
#include <future>
#include <queue>
#include <thread>
#include <condition_variable>

/**
 * @brief A namespace that contains a ThreadPool and Matrix class.
 */
namespace Math
{
    /**
     * @brief A threadPool class that holds a list of threads and a queue of task.
     */
    class ThreadPool
    {
    public:
        // basic function wrapper
        using Task = std::function<void()>;

        /**
         * @brief Construct a new Thread Pool object
         *
         * @param numThreads number of threads that the thread pool contains
         */
        explicit ThreadPool(std::size_t numThreads)
        {
            start(numThreads);
        }

        /**
         * @brief Destroy the Thread Pool object
         */
        ~ThreadPool()
        {
            stop();
        }

        /**
         * @brief Get the Pool Size object
         *
         * @return int
         */
        int getPoolSize() const
        {
            return threads.size();
        }

        /**
         * @brief Get the number of tasks in queue
         *
         * @return int
         */
        int getQueueSize() const
        {
            return tasks.size();
        }

        // im gonna be real honest, i barely know how this works
        /**
         * @brief adds a task to the queue for processes by the threads
         *
         * @tparam T generic class
         * @param task generic task (some type of function)
         * @return std::future<decltype(task())> whatever value the task returns when it completes
         */
        template <class T>
        auto enqueue(T task) -> std::future<decltype(task())>
        {
            // something about wrapping the task to get a future value.
            auto wrapper = std::make_shared<std::packaged_task<decltype(task())()>>(std::move(task));
            {
                std::unique_lock<std::mutex> lock{eventMutex};
                tasks.emplace([=, this]
                              { (*wrapper)(); });
            }

            // remainingTasks.fetch_add(1);
            event.notify_one();
            return wrapper->get_future();
        }

        void waitUntilEmpty()
        {
            std::unique_lock<std::mutex> lock(eventMutex);
            event.wait(lock, [this]
                       { return remainingTasks == 0; });
        }

    private:
        // the "pool" of threads
        std::vector<std::thread> threads;
        /**
        synchronization primitive that allows threads to wait until a certain condition is satisfied.
        It enables "wait and notify" behavior between threads, where one or multiple
        threads can be blocked until another thread sends a notification.
         */
        std::condition_variable event;
        std::mutex eventMutex;
        bool stopping = false;

        std::queue<Task> tasks;
        std::atomic<int> remainingTasks;

        /**
         * @brief initialize the threads
         *
         * @param numThreads
         */
        void start(std::size_t numThreads)
        {
            for (int i = 0; i < numThreads; i++)
            {
                threads.emplace_back([=, this]
                                     {
                while (true)
                {
                    Task task;
                    {
                        // locks the following block so only one thread can access the following code 
                        std::unique_lock<std::mutex> lock{eventMutex};

                        // blocks current thread from executing unless stopping or non-empty tasks
                        event.wait(lock, [&] { return stopping || !tasks.empty();});

                        // only stop if tasks are completed and notified to stop
                        if (stopping && tasks.empty())
                            break;
                        
                        //removes the first task and prepares it
                        task = std::move(tasks.front());
                        tasks.pop();
                        //remainingTasks.fetch_sub(1);
                    }

                    //performs the first task, this is out of scope because it should not be locked by the mutex during execution
                    try
                    {
                        task();
                    }
                    catch(const std::exception& e)
                    {
                        std::cerr << "ok so my tasks suck " << e.what() << '\n';
                        throw std::runtime_error("wow this sucks");
                    }

                } });
            }
        }

        /**
         * @brief stops all the threads in the thread pool
         */
        void stop() noexcept
        {
            {
                std::unique_lock<std::mutex> lock{eventMutex};
                stopping = true;
            }
            event.notify_all();

            for (std::thread &thread : threads)
            {
                thread.join();
            }
        }
    };

    /**
     * @brief A matrix class defined by rows * cols (height * width). Vectors will be treated like column matrices.
     * Matrix operations support multi-threading; however, it still uses naive algorithms.
     */
    class Matrix
    {
    private:
        std::vector<double> vals;
        int rows;
        int cols;
        int size;

        /**
         * @brief an abstracted method that allows the easy creation of operations optimized by thread pools
         *
         * @param fn the function should be void, make sure it doesn't cause deadlocks
         * @param rows the rows of the matrix that it operates on
         */
        static void threadPoolProcess(std::function<void(int start, int end)> fn, int rows)
        {
            try
            {
                /* code */

                std::condition_variable event;
                static std::mutex eventMutex;
                std::atomic<int> completedTasksCount(0);

                const int MAX_THREADS = Matrix::threadPool.getPoolSize();
                const int threadNum = std::min({rows, MAX_THREADS});

                int block = std::max(rows / threadNum, 1);

                int start = 0;
                int end = block;

                for (int i = 0; i < threadNum - 1; i++)
                {

                    Matrix::threadPool.enqueue([start, end, &fn, &completedTasksCount, &event]
                                               {fn(start, end);
                                            {
                                                std::unique_lock<std::mutex> lock{eventMutex};
                                                completedTasksCount.fetch_add(1);
                                                event.notify_one();
                                            } });
                    start = end;
                    end += block;
                }

                Matrix::threadPool.enqueue([start, rows, &fn, &completedTasksCount, &event]
                                           {fn(start, rows);
                                        {
                                            std::unique_lock<std::mutex> lock{eventMutex};
                                            completedTasksCount.fetch_add(1);
                                            event.notify_one();
                                        } });

                {
                    std::unique_lock<std::mutex> lock{eventMutex};
                    event.wait(lock, [&completedTasksCount, threadNum]
                               { return completedTasksCount == threadNum; });
                    // event.notify_all();
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "thread process dies " << e.what() << '\n';
                throw std::runtime_error("krill myself");
            }
        }
        /**
         * @brief an abstracted function that just uses basic multi-threading.
         *
         * @param fn the function should be void, make sure it doesn't cause deadlocks
         * @param rows the rows of the matrix that it operates on
         */
        static void simpleThreadProcess(std::function<void(int start, int end)> fn, int rows)
        {
            const int MAX_THREADS = Matrix::threadPool.getPoolSize();
            const int threadNum = std::min({rows, MAX_THREADS});
            const int block = std::max(rows / threadNum, 1);
            std::vector<std::thread> threads(threadNum);
            std::reference_wrapper funcRef = std::ref(fn);

            int start = 0;
            int end = block;

            for (int i = 0; i < threadNum - 1; i++)
            {
                threads[i] = std::thread(funcRef, start, end);
                start = end;
                end += block;
            }
            threads[threadNum - 1] = std::thread(funcRef, start, rows);

            for (int i = 0; i < threadNum; i++)
                threads[i].join();
        }

    public:
        /**
         * @brief a static thread pool that is used by all matrix objects.
         */
        inline static ThreadPool threadPool = ThreadPool(std::thread::hardware_concurrency());

        /**
         * @brief Construct a new default Matrix object
         *
         */
        Matrix()
        {
            rows = 0;
            cols = 0;
            vals = std::vector<double>();
            size = 0;
        }

        /**
         * @brief Construct a new Matrix object
         *
         * @param rows
         * @param cols
         * @param val
         */
        Matrix(int rows, int cols, double val = 0) : cols(cols), rows(rows)
        {
            vals = std::vector<double>(rows * cols, val);
            size = rows * cols;
        }

        /**
         * @brief Construct a new Matrix object
         *
         * @param rows
         * @param cols
         * @param v
         */
        Matrix(int rows, int cols, const std::vector<double> &v)
        {
            if (rows * cols != v.size())
                throw std::invalid_argument("Row and column dimensions do not match input vector.");
            Matrix::cols = cols;
            Matrix::rows = rows;
            vals = v;
            size = rows * cols;
        }

        /**
         * @brief Construct a new Matrix object
         *
         * @param v
         * @param isCol whether the matrix is a column vector or not, default true
         */
        Matrix(const std::vector<double> &v, bool isCol = true)
        {
            if (isCol)
            {
                cols = 1;
                rows = v.size();
            }
            else
            {
                cols = v.size();
                rows = 1;
            }

            vals = v;
            size = rows * cols;
        }

        /**
         * @brief Construct a new Matrix object
         *
         * @param vv some nested vector
         */
        Matrix(const std::vector<std::vector<double>> &vv)
        {
            if (vv.size() == 0)
                throw std::invalid_argument("Nested vector must have values.");

            cols = vv[0].size();
            rows = vv.size();
            vals.reserve(cols * rows);

            for (int i = 0; i < rows; i++)
            {
                std::vector<double> v = vv[i];
                if (v.size() != cols)
                    throw std::invalid_argument("Rows must be same length.");

                vals.insert(vals.end(), v.begin(), v.end());
            }
            size = rows * cols;
        }

        /**
         * @brief Copy constructor
         *
         * @param other
         */
        Matrix(const Matrix &other)
        {
            cols = other.cols;
            rows = other.rows;
            vals = other.vals;
            size = other.size;
        }

        /**
         * @brief Get the Cols object
         *
         * @return int
         */
        int getCols() const
        {
            return cols;
        }

        /**
         * @brief Get the Rows object
         *
         * @return int
         */
        int getRows() const
        {
            return rows;
        }

        /**
         * @brief Get the Size object
         *
         * @return int
         */
        int getSize() const
        {
            return size;
        }

        /**
         * @brief Get the Vals object
         *
         * @return std::vector<double>
         */
        std::vector<double> getVals() const
        {
            return vals;
        }

        /**
         * @brief get the value of the matrix at the specified position
         *
         * @param row
         * @param col
         * @return double
         */
        double get(int row, int col)
        {
            return vals[row * cols + col];
        }

        /**
         * @brief sets value of matrix at specified location
         *
         * @param row
         * @param col
         * @param val
         */
        void set(int row, int col, double val)
        {
            vals[row * cols + col] = val;
        }

        /**
         * @brief Generates a random matrix with specified dimensions, values are normalized between 0 and 1.
         *
         * @param rows
         * @param cols
         * @param scale value scaling
         * @return Matrix
         */
        static Matrix generateRandom(int rows, int cols, double scale = 1)
        {
            int size = rows * cols;
            std::vector<double> v(size, 0.0);
            for (int i = 0; i < size; i++)
                v[i] = scale * (static_cast<double>(rand()) / RAND_MAX * 2 - 1);
            return Matrix(rows, cols, v);
        }

        /**
         * @brief returns a transposed matrix
         *
         * @return Matrix
         */
        Matrix transpose() const
        {
            std::vector<double> v(size);
            for (int i = 0; i < cols; i++)
                for (int j = 0; j < rows; j++)
                    v[i * cols + j] = vals[j * cols + i];

            return Matrix(cols, rows, v);
        }

        /**
         * @brief transposes *this matrix
         *
         */
        void transposed()
        {
            std::vector<double> v(size);
            for (int i = 0; i < cols; i++)
                for (int j = 0; j < rows; j++)
                    v[i * rows + j] = vals[j * cols + i];

            int temp = cols;
            cols = rows;
            rows = temp;
            vals = v;
        }

        /**
         * @brief prints value of matrix
         *
         */
        void print() const
        {
            for (int i = 0; i < size; i++)
            {
                std::string v = std::to_string(vals[i]);
                std::cout << v << " ";
                if ((i + 1) % cols == 0)
                    std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        /**
         * @brief applies function on every element on a copy of *this matrix
         *
         * @param fn
         * @return Matrix
         */
        Matrix applyFn(std::function<double(double)> fn) const
        {
            std::vector<double> v(size);
            for (int i = 0; i < size; i++)
                v[i] = fn(vals[i]);

            return Matrix(rows, cols, v);
        }

        /**
         * @brief applies function on every element on *this matrix
         *
         * @param fn
         * @return Matrix
         */
        void applyFnHere(std::function<double(double)> fn)
        {
            for (int i = 0; i < size; i++)
                vals[i] = fn(vals[i]);
        }

        /**
         * @brief gets value of matrix at the specified index (data is flattened)
         *
         * @param i
         * @return double
         */
        double operator[](int i) const
        {
            return vals[i];
        }

        /**
         * @brief sets value of matrix at the specified index (data is flattened)
         *
         * @param i
         * @return double&
         */
        double &operator[](int i)
        {
            return vals[i];
        }

        /**
         * @brief adds matrix
         *
         * @param m
         * @return Matrix&
         */
        Matrix &operator+=(const Matrix &m)
        {
            if (rows != m.rows)
                throw std::invalid_argument("Height mismatch.");
            else if (cols != m.cols)
                throw std::invalid_argument("Width mismatch.");

            for (int i = 0; i < size; i++)
                vals[i] += m[i];

            return *this;
        }

        /**
         * @brief subtracts matrix
         *
         * @param m
         * @return Matrix&
         */
        Matrix &operator-=(const Matrix &m)
        {
            if (rows != m.rows)
                throw std::invalid_argument("Height mismatch.");
            else if (cols != m.cols)
                throw std::invalid_argument("Width mismatch.");

            for (int i = 0; i < size; i++)
                vals[i] -= m[i];

            return *this;
        }

        /**
         * @brief multiplies a matrix
         *
         * @param m
         * @return Matrix&
         */
        Matrix &operator*=(const Matrix &m)
        {
            int mCols = m.cols;
            int mRows = m.rows;

            if (cols != mRows)
                throw std::invalid_argument("Dimension mismatch.");

            std::vector<double> v(rows * mCols);
            threadPoolProcess([&](int start, int end)
                              {
                              for (int i = start; i < end; i++)
                                  for (int j = 0; j < mCols; j++)
                                      for (int k = 0; k < mRows; k++)
                                          v[i * mCols + j] += vals[i * cols + k] * m[k * mCols + j]; },
                              rows);

            vals = v;
            cols = mCols;
            size = rows * cols;

            return *this;
        }

        /**
         * @brief adds a double to every element in the matrix
         *
         * @param d
         * @return Matrix&
         */
        Matrix &operator+=(const double &d)
        {
            for (int i = 0; i < size; i++)
                vals[i] += d;

            return *this;
        }

        /**
         * @brief subtracts a double to every element in the matrix
         *
         * @param d
         * @return Matrix&
         */
        Matrix &operator-=(const double &d)
        {
            for (int i = 0; i < size; i++)
                vals[i] -= d;

            return *this;
        }

        /**
         * @brief multiplies a double to every element in the matrix
         *
         * @param d
         * @return Matrix&
         */
        Matrix &operator*=(const double &d)
        {
            for (int i = 0; i < size; i++)
                vals[i] *= d;

            return *this;
        }

        /**
         * @brief divides a double to every element in the matrix
         *
         * @param d
         * @return Matrix&
         */
        Matrix &operator/=(const double &d)
        {
            for (int i = 0; i < size; i++)
                vals[i] /= d;

            return *this;
        }

        /**
         * @brief performs the dot product between two column or row matrices
         *
         * @param m1
         * @param m2
         * @return double
         */
        static double dot(const Matrix &m1, const Matrix &m2)
        {
            std::vector<double> m1Vals = m1.getVals();
            std::vector<double> m2Vals = m2.getVals();

            if ((m1Vals.size() != m2Vals.size()) && (m1.getCols() != 1 || m2.getCols() != 1) && (m1.getRows() != 1 || m2.getRows() != 1))
                throw std::invalid_argument("Dot product must be performed between two column matrices or two row matrices of the same size");

            double sum = 0;
            for (int i = 0; i < m1Vals.size(); i++)
                sum += m1Vals[i] * m2Vals[i];

            return sum;
        }

        /**
         * @brief performs the general inner product between two matrices
         *
         * @param m1
         * @param m2
         * @return Matrix
         */
        static Matrix iProd(Matrix m1, const Matrix &m2)
        {
            int m1Cols = m1.getCols();
            int m1Rows = m1.getRows();
            int m2Cols = m2.getCols();
            int m2Rows = m2.getRows();

            if (m1.getRows() != m2.getRows())
                throw std::invalid_argument("Dimension mismatch.");

            std::vector<double> v(m1Cols * m2Cols);

            // for (int i = 0; i < m1.getCols(); i++)
            //     for (int j = 0; j < m2.getCols(); j++)
            //         for (int k = 0; k < m1.getRows(); k++)
            //             v[i * m2.getCols() + j] += m1[k * m1.getCols() + i] * m2[k * m2.getCols() + j];

            threadPoolProcess([&](int start, int end)
                              {
                              for (int i = start; i < end; i++)
                                  for (int j = 0; j < m2Cols; j++)
                                      for (int k = 0; k < m1Rows; k++)
                                          v[i * m2Cols + j] += m1[k * m1Cols + i] * m2[k * m2Cols + j]; },
                              m1Cols);

            return Matrix(m1Cols, m2Cols, v);
        }

        /**
         * @brief performs the general outer product between two matrices
         *
         * @param m1
         * @param m2
         * @return Matrix
         */
        static Matrix oProd(const Matrix &m1, const Matrix &m2)
        {
            int m1Cols = m1.getCols();
            int m1Rows = m1.getRows();
            int m2Cols = m2.getCols();
            int m2Rows = m2.getRows();

            if (m1Cols != m2Cols)
                throw std::invalid_argument("Dimension mismatch.");

            std::vector<double> v(m1Rows * m2Rows);

            threadPoolProcess([&](int start, int end)
                              {
                                for (int i = start; i < end; i++)
                                    for (int j = 0; j < m2Rows; j++)
                                        for (int k = 0; k < m1Cols; k++)
                                            v[i * m2Rows + j] += m1[i * m1Cols + k] * m2[j * m2Cols + k]; },
                              m1Rows);

            return Matrix(m1Rows, m2Rows, v);
        }

        /**
         * @brief performs the Hadamard product between two matrices
         *
         * @param m1
         * @param m2
         * @return Matrix
         */
        static Matrix hProd(Matrix m1, const Matrix &m2)
        {
            std::vector<double> m1Vals = m1.getVals();
            std::vector<double> m2Vals = m2.getVals();

            if (m1.getCols() != m2.getCols() || m1.getRows() != m2.getRows())
                throw std::invalid_argument("Hadamard product must be performed between two matrices of the same dimensions.");

            for (int i = 0; i < m1Vals.size(); i++)
                m1[i] *= m2Vals[i];

            return m1;
        }

        /**
         * @brief returns the sum of two matrices
         *
         * @param mat1
         * @param mat2
         * @return Matrix
         */
        friend Matrix operator+(Matrix mat1, const Matrix &mat2)
        {
            mat1 += mat2;
            return mat1;
        }

        /**
         * @brief returns the difference of two matrices
         *
         * @param mat1
         * @param mat2
         * @return Matrix
         */
        friend Matrix operator-(Matrix mat1, const Matrix &mat2)
        {
            mat1 -= mat2;
            return mat1;
        }

        /**
         * @brief returns the product of two matrices
         *
         * @param mat1
         * @param mat2
         * @return Matrix
         */
        friend Matrix operator*(Matrix mat1, const Matrix &mat2)
        {
            mat1 *= mat2;
            return mat1;
        }

        /**
         * @brief returns the sum of a matrix and a double
         *
         * @param mat1
         * @param d
         * @return Matrix
         */
        friend Matrix operator+(Matrix mat1, const double &d)
        {
            mat1 += d;
            return mat1;
        }

        /**
         * @brief returns the difference of a matrix and a double
         *
         * @param mat1
         * @param d
         * @return Matrix
         */
        friend Matrix operator-(Matrix mat1, const double &d)
        {
            mat1 -= d;
            return mat1;
        }

        /**
         * @brief returns the product of a matrix and a double
         *
         * @param mat1
         * @param d
         * @return Matrix
         */
        friend Matrix operator*(Matrix mat1, const double &d)
        {
            mat1 *= d;
            return mat1;
        }

        /**
         * @brief returns the quotient of a matrix and a double
         *
         * @param mat1
         * @param d
         * @return Matrix
         */
        friend Matrix operator/(Matrix mat1, const double &d)
        {
            mat1 /= d;
            return mat1;
        }

        /**
         * @brief serializes the matrix
         *
         * @return std::string
         */
        std::string toString() const
        {
            std::string s = "";
            for (int i = 0; i < size; i++)
            {
                std::string v = std::to_string(vals[i]);
                s += v + " ";
                if ((i + 1) % cols == 0)
                    s += "\n";
            }

            return s;
        }

        /**
         * @brief overload for std::cout
         *
         * @param out
         * @param m
         * @return std::ostream&
         */
        friend std::ostream &operator<<(std::ostream &out, const Matrix &m)
        {
            out << m.toString();
            return out;
        }

        /**
         * @brief type cast override
         *
         * @return vector<double>
         */
        operator std::vector<double>() const { return vals; }

        /**
         * @brief type cast override
         *
         * @return std::vector<double>
         */
        operator std::vector<std::vector<double>>() const
        {
            std::vector<std::vector<double>> vv(rows, std::vector<double>(cols));
            int count = 0;
            for (int i = 0; i < size; i++)
            {
                vv[count][i % cols] = vals[i];
                if ((i + 1) % cols == 0)
                    count++;
            }
            return vv;
        }

        /**
         * @deprecated
         * @brief for testing
         *
         * @param ogm
         * @param m
         * @return Matrix
         */
        static Matrix poolThreaded(const Matrix &ogm, const Matrix &m)
        {
            int rows = ogm.getRows();
            int cols = ogm.getCols();
            int mCols = m.getCols();
            int mRows = m.getRows();

            if (ogm.getCols() != m.getRows())
                throw std::invalid_argument("Dimension mismatch.");

            std::vector<double> v(rows * m.getCols());

            threadPoolProcess([&](int start, int end)
                              {
                              for (int i = start; i < end; i++)
                                  for (int j = 0; j < mCols; j++)
                                      for (int k = 0; k < mRows; k++)
                                          v[i * mCols + j] += ogm[i * cols + k] * m[k * mCols + j]; },
                              rows);

            return Matrix(rows, m.getCols(), v);
        }

        /**
         * @deprecated
         * @brief for testing
         *
         * @param ogm
         * @param m
         * @return Matrix
         */
        static Matrix simpleThreaded(const Matrix &ogm, const Matrix &m)
        {
            int rows = ogm.getRows();
            int cols = ogm.getCols();
            int mCols = m.getCols();
            int mRows = m.getRows();

            if (ogm.getCols() != m.getRows())
                throw std::invalid_argument("Dimension mismatch.");

            std::vector<double> v(rows * m.getCols());

            simpleThreadProcess([&](const int &start, const int &end)
                                {
                                for (int i = start; i < end; i++)
                                    for (int j = 0; j < mCols; j++)
                                        for (int k = 0; k < mRows; k++)
                                            v[i * mCols + j] += ogm[i * cols + k] * m[k * mCols + j]; },
                                rows);

            return Matrix(rows, m.getCols(), v);
        }
    };
};