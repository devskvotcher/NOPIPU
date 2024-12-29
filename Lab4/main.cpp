#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>      
#include <iomanip>        


#include "mkl.h" 


template <typename T>
double measureAndRun(void (*func)(T*, T*, T*, int),
    T* a, T* b, T* c, int N)
{
    auto start = std::chrono::high_resolution_clock::now();
    func(a, b, c, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

template <typename T>
double measureAndRunWithBlock(void (*func)(T*, T*, T*, int, int),
    T* a, T* b, T* c, int N, int blockSize)
{
    auto start = std::chrono::high_resolution_clock::now();
    func(a, b, c, N, blockSize);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

// 2.1 sumUsualSquare: c[i*N + j] = a[i*N + j]^2 + b[j*N + i]^2
template <typename T>
void sumUsualSquare(T* a, T* b, T* c, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            T valA = a[i * N + j];
            T valB = b[j * N + i];
            c[i * N + j] = valA * valA + valB * valB;
        }
}

// 2.2 sumBlockingSquare: блочная версия
template <typename T>
void sumBlockingSquare(T* a, T* b, T* c, int N, int blockSize)
{
    for (int i = 0; i < N; i += blockSize)
        for (int j = 0; j < N; j += blockSize)
            for (int ii = i; ii < std::min(i + blockSize, N); ++ii)
                for (int jj = j; jj < std::min(j + blockSize, N); ++jj)
                {
                    T valA = a[ii * N + jj];
                    T valB = b[jj * N + ii];
                    c[ii * N + jj] = valA * valA + valB * valB;
                }
}

// 2.3 sumUsualSinCos: c[i*N + j] = sin(a[i*N + j]) + cos(b[j*N + i])
template <typename T>
void sumUsualSinCos(T* a, T* b, T* c, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            c[i * N + j] = static_cast<T>(std::sin(a[i * N + j])
                + std::cos(b[j * N + i]));
}

// 2.4 sumBlockingSinCos
template <typename T>
void sumBlockingSinCos(T* a, T* b, T* c, int N, int blockSize)
{
    for (int i = 0; i < N; i += blockSize)
        for (int j = 0; j < N; j += blockSize)
            for (int ii = i; ii < std::min(i + blockSize, N); ++ii)
                for (int jj = j; jj < std::min(j + blockSize, N); ++jj)
                    c[ii * N + jj] = static_cast<T>(
                        std::sin(a[ii * N + jj]) + std::cos(b[jj * N + ii])
                        );
}

// 2.5 sumUsualLogMul: c[i*N + j] = log(1 + a[i*N + j]*b[j*N + i])
template <typename T>
void sumUsualLogMul(T* a, T* b, T* c, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            T prod = a[i * N + j] * b[j * N + i];
            c[i * N + j] = static_cast<T>(std::log(1.0 + prod));
        }
}

// 2.6 sumBlockingLogMul
template <typename T>
void sumBlockingLogMul(T* a, T* b, T* c, int N, int blockSize)
{
    for (int i = 0; i < N; i += blockSize)
        for (int j = 0; j < N; j += blockSize)
            for (int ii = i; ii < std::min(i + blockSize, N); ++ii)
                for (int jj = j; jj < std::min(j + blockSize, N); ++jj)
                {
                    T prod = a[ii * N + jj] * b[jj * N + ii];
                    c[ii * N + jj] = static_cast<T>(std::log(1.0 + prod));
                }
}

// 2.7 sumUsual4 (пример с искусственной зависимостью)
template <typename T>
void sumUsual4(T* a, T* b, T* c, int N)
{
    a[0] = static_cast<T>(0);
    a[1] = static_cast<T>(1);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            T s = a[i * N + j] + b[j * N + i];
            T s2 = s * s;
            T denom = static_cast<T>(1) + a[i * N + j] * a[i * N + j];

            c[i * N + j] = s2 / denom;
            if ((i * N + j) < N * N - 1)
                a[i * N + j + 1] += s / denom;
        }
}

// 2.8 sumBlocking4
template <typename T>
void sumBlocking4(T* a, T* b, T* c, int N, int blockSize)
{
    a[0] = static_cast<T>(0);
    a[1] = static_cast<T>(1);

    for (int i = 0; i < N; i += blockSize)
        for (int j = 0; j < N; j += blockSize)
            for (int ii = i; ii < std::min(i + blockSize, N); ++ii)
                for (int jj = j; jj < std::min(j + blockSize, N); ++jj)
                {
                    T s = a[ii * N + jj] + b[jj * N + ii];
                    T s2 = s * s;
                    T denom = static_cast<T>(1) + a[ii * N + jj] * a[ii * N + jj];

                    c[ii * N + jj] = s2 / denom;
                    if ((ii * N + jj) < N * N - 1)
                        a[ii * N + jj + 1] += s / denom;
                }
}

void myBestMatMul(double* A, double* B, double* C, int N, int blockSize)
{
    for (int iBlock = 0; iBlock < N; iBlock += blockSize)
        for (int jBlock = 0; jBlock < N; jBlock += blockSize)
            for (int kBlock = 0; kBlock < N; kBlock += blockSize)
            {
                int iMax = std::min(iBlock + blockSize, N);
                int jMax = std::min(jBlock + blockSize, N);
                int kMax = std::min(kBlock + blockSize, N);

                for (int i = iBlock; i < iMax; i++)
                    for (int j = jBlock; j < jMax; j++)
                    {
                        double sum = C[i * N + j];
                        for (int k = kBlock; k < kMax; k++)
                            sum += A[i * N + k] * B[k * N + j];
                        C[i * N + j] = sum;
                    }
            }
}

double measureMyBestMatMul(double* A, double* B, double* C, int N, int blockSize)
{
    auto start = std::chrono::high_resolution_clock::now();
    // Обнуляем C
    for (int i = 0; i < N * N; i++)
        C[i] = 0.0;
    myBestMatMul(A, B, C, N, blockSize);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

//-----------------------------------------------------------------
// 4) Используем dgemm из MKL для умножения матриц
//-----------------------------------------------------------------
double measureMKL_dgemm(double* A, double* B, double* C, int N)
{
    auto start = std::chrono::high_resolution_clock::now();    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N,
        1.0, A, N,
        B, N,
        0.0, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

//-----------------------------------------------------------------
// 5) Умножение симметричной матрицы (BLAS3) cblas_dsymm
//-----------------------------------------------------------------
double measureMKL_dsymm(double* A, double* B, double* C, int N)
{
    // A — симметричная, B — N x N
    // C = A*B (или C = alpha*A*B + beta*C)
    auto start = std::chrono::high_resolution_clock::now();    
    cblas_dsymm(CblasRowMajor,
        CblasLeft, CblasUpper,  
        N, N,
        1.0, A, N,
        B, N,
        0.0, C, N);

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

//-----------------------------------------------------------------
// 6) LAPACK dpotrf (разложение Холецкого)
//-----------------------------------------------------------------
double measureMKL_dpotrf(double* A, int N)
{
    
    auto start = std::chrono::high_resolution_clock::now();    
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', N, A, N);

    auto end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();
    if (info != 0)
        std::cerr << "dpotrf error, info=" << info << "\n";
    return secs;
}


int main()
{
    std::vector<int> sizesN = { 1024, 2048 };
    std::vector<int> blockSizes = { 16, 32, 64, 128, 256, 512 };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Matrix Summation Experiment\n\n";

    for (auto N : sizesN)
    {
        std::cout << "========== N = " << N << " ==========\n";
        std::vector<double> A(N * N), B(N * N), C(N * N);

        std::mt19937 gen(0);
        std::uniform_real_distribution<double> dist(0.01, 10.0);
        for (int i = 0; i < N * N; ++i)
        {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        // ---- sumUsualSquare ----
        {
            auto A_copy = A;
            auto B_copy = B;
            auto C_copy = C;
            double t = measureAndRun(sumUsualSquare<double>,
                A_copy.data(), B_copy.data(),
                C_copy.data(), N);
            std::cout << "sumUsualSquare time = " << t << " s\n";
        }
        std::cout << "BlockSize | time(blockSquare)\n";
        for (auto bs : blockSizes)
        {
            auto A_copy = A;
            auto B_copy = B;
            auto C_copy = C;
            double t = measureAndRunWithBlock(sumBlockingSquare<double>,
                A_copy.data(), B_copy.data(),
                C_copy.data(), N, bs);
            std::cout << std::setw(6) << bs << "     | " << t << "\n";
        }
        std::cout << "\n";

        // ---- sumUsualSinCos ----
        {
            auto A_copy = A;
            auto B_copy = B;
            auto C_copy = C;
            double t = measureAndRun(sumUsualSinCos<double>,
                A_copy.data(), B_copy.data(),
                C_copy.data(), N);
            std::cout << "sumUsualSinCos time = " << t << " s\n";
        }
        std::cout << "BlockSize | time(blockSinCos)\n";
        for (auto bs : blockSizes)
        {
            auto A_copy = A;
            auto B_copy = B;
            auto C_copy = C;
            double t = measureAndRunWithBlock(sumBlockingSinCos<double>,
                A_copy.data(), B_copy.data(),
                C_copy.data(), N, bs);
            std::cout << std::setw(6) << bs << "     | " << t << "\n";
        }
        std::cout << "\n";

        // ---- sumUsualLogMul ----
        {
            auto A_copy = A;
            auto B_copy = B;
            auto C_copy = C;
            double t = measureAndRun(sumUsualLogMul<double>,
                A_copy.data(), B_copy.data(),
                C_copy.data(), N);
            std::cout << "sumUsualLogMul time = " << t << " s\n";
        }
        std::cout << "BlockSize | time(blockLogMul)\n";
        for (auto bs : blockSizes)
        {
            auto A_copy = A;
            auto B_copy = B;
            auto C_copy = C;
            double t = measureAndRunWithBlock(sumBlockingLogMul<double>,
                A_copy.data(), B_copy.data(),
                C_copy.data(), N, bs);
            std::cout << std::setw(6) << bs << "     | " << t << "\n";
        }
        std::cout << "\n";

        // ---- sumUsual4 ----
        {
            auto A_copy = A;
            auto B_copy = B;
            auto C_copy = C;
            double t = measureAndRun(sumUsual4<double>,
                A_copy.data(), B_copy.data(),
                C_copy.data(), N);
            std::cout << "sumUsual4 time = " << t << " s\n";
        }
        std::cout << "BlockSize | time(block4)\n";
        for (auto bs : blockSizes)
        {
            auto A_copy = A;
            auto B_copy = B;
            auto C_copy = C;
            double t = measureAndRunWithBlock(sumBlocking4<double>,
                A_copy.data(), B_copy.data(),
                C_copy.data(), N, bs);
            std::cout << std::setw(6) << bs << "     | " << t << "\n";
        }
        std::cout << "\n";
    }

    // ==============================================================
    // 7.2. myBestMatMul vs cblas_dgemm (BLAS)
    // ==============================================================
    std::vector<int> mulSizes = { 512, 1024, 2048 };
    int bestBlockSize = 64; // или 128, если у вас лучше

    std::cout << "\nCompare myBestMatMul vs MKL dgemm:\n";
    std::cout << "   N    MyTime[s]   MKLTime[s]  SpeedUp(MKL/My)\n";

    for (auto N : mulSizes)
    {
        std::vector<double> A(N * N), B(N * N), C(N * N);

        std::mt19937 gen(0);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < N * N; i++)
        {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        double myT = measureMyBestMatMul(A.data(), B.data(), C.data(), N, bestBlockSize);

        // Замер MKL dgemm
        std::fill(C.begin(), C.end(), 0.0);
        double mklT = measureMKL_dgemm(A.data(), B.data(), C.data(), N);

        // Во сколько раз MKL быстрее или медленнее
        double speedup = mklT / myT;
        std::cout << std::setw(5) << N << "   "
            << std::setw(9) << myT << "   "
            << std::setw(9) << mklT << "   "
            << std::setw(9) << speedup << "\n";
    }

    // ==============================================================
    // 7.3. BLAS3: cblas_dsymm (симметрическая матрица)
    // ==============================================================
    {
        int N = 1024;
        std::cout << "\nExample cblas_dsymm, N=" << N << "\n";
        std::vector<double> A(N * N), B(N * N), C(N * N);

        std::mt19937 gen(1);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < N; i++)
            for (int j = i; j < N; j++)
            {
                double val = dist(gen);
                A[i * N + j] = val;   
                A[j * N + i] = val;   
            }
        for (int i = 0; i < N * N; i++)
            B[i] = dist(gen);

        double tSymm = measureMKL_dsymm(A.data(), B.data(), C.data(), N);
        std::cout << "Time for dsymm: " << tSymm << " s\n";
    }

    // ==============================================================
    // 7.4. LAPACK: dpotrf (разложение Холецкого)
    // ==============================================================
    {
        int N = 1024;
        std::cout << "\nExample dpotrf, N=" << N << "\n";

        std::vector<double> A(N * N);

        // Генерируем A = M^T * M, чтобы A была симметричной положительно-определённой
        {
            std::vector<double> M(N * N);
            std::mt19937 gen(2);
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (int i = 0; i < N * N; i++)
                M[i] = dist(gen);

            // A = M^T * M
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < N; k++)
                        sum += M[k * N + i] * M[k * N + j];
                    A[j * N + i] = sum;
                }
        }

        double tPotrf = measureMKL_dpotrf(A.data(), N);
        std::cout << "Time for dpotrf: " << tPotrf << " s\n";
        // После dpotrf в A лежит верхняя часть R (если 'U'), так что
        // A – фактически R.
        // Проверку можно сделать (при малых N).
    }

    std::cout << "\nDone.\n";
    return 0;
}
