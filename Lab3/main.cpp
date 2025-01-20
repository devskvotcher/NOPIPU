#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>   
#include <emmintrin.h>
#include <cmath>      

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
    T* a, T* b, T* c, int N, int block_size)
{
    auto start = std::chrono::high_resolution_clock::now();
    func(a, b, c, N, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count(); 
}

// Обычная (неблочная)
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

// Блочная
template <typename T>
void sumBlockingSquare(T* a, T* b, T* c, int N, int block_size)
{
    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < N; j += block_size)
            for (int ii = i; ii < std::min(i + block_size, N); ++ii)
                for (int jj = j; jj < std::min(j + block_size, N); ++jj)
                {
                    T valA = a[ii * N + jj];
                    T valB = b[jj * N + ii];
                    c[ii * N + jj] = valA * valA + valB * valB;
                }
}

// Векторизованная (SSE2) — только для double
template <typename T>
void sumUsualIntrSquare(T* a, T* b, T* c, int N)
{
    if constexpr (!std::is_same_v<T, double>) {
        sumUsualSquare(a, b, c, N);
        return;
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j += 2)
        {
            __m128d Aij = _mm_load_pd(&a[i * N + j]);
            __m128d Bij = _mm_set_pd(b[(j + 1) * N + i], b[j * N + i]);

            __m128d A2 = _mm_mul_pd(Aij, Aij);
            __m128d B2 = _mm_mul_pd(Bij, Bij);
            __m128d sum = _mm_add_pd(A2, B2);

            _mm_store_pd(&c[i * N + j], sum);
        }
    }
}

// Блочная с SSE2
template <typename T>
void sumBlockingIntrSquare(T* a, T* b, T* c, int N, int block_size)
{
    if constexpr (!std::is_same_v<T, double>) {
        // fallback
        sumBlockingSquare(a, b, c, N, block_size);
        return;
    }

    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < N; j += block_size)
            for (int ii = i; ii < i + block_size && ii < N; ++ii)
                for (int jj = j; jj < j + block_size && jj < N; jj += 2)
                {
                    __m128d Aij = _mm_load_pd(&a[ii * N + jj]);
                    __m128d Bij = _mm_set_pd(b[(jj + 1) * N + ii], b[jj * N + ii]);

                    __m128d A2 = _mm_mul_pd(Aij, Aij);
                    __m128d B2 = _mm_mul_pd(Bij, Bij);
                    __m128d sum = _mm_add_pd(A2, B2);

                    _mm_store_pd(&c[ii * N + jj], sum);
                }
}

//-----------------------------------------------------------
// 2) c[i*N + j] = sin(a[i*N + j]) + cos(b[j*N + i])
//-----------------------------------------------------------

template <typename T>
void sumUsualSinCos(T* a, T* b, T* c, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            c[i * N + j] = static_cast<T>(std::sin(a[i * N + j]) + std::cos(b[j * N + i]));
}

template <typename T>
void sumBlockingSinCos(T* a, T* b, T* c, int N, int block_size)
{
    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < N; j += block_size)
            for (int ii = i; ii < std::min(i + block_size, N); ++ii)
                for (int jj = j; jj < std::min(j + block_size, N); ++jj)
                    c[ii * N + jj] = static_cast<T>(
                        std::sin(a[ii * N + jj]) + std::cos(b[jj * N + ii])
                        );
}

//-----------------------------------------------------------
// 3) c[i*N + j] = log(1 + a[i*N + j] * b[j*N + i]) 
//-----------------------------------------------------------

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

template <typename T>
void sumBlockingLogMul(T* a, T* b, T* c, int N, int block_size)
{
    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < N; j += block_size)
            for (int ii = i; ii < std::min(i + block_size, N); ++ii)
                for (int jj = j; jj < std::min(j + block_size, N); ++jj)
                {
                    T prod = a[ii * N + jj] * b[jj * N + ii];
                    c[ii * N + jj] = static_cast<T>(std::log(1.0 + prod));
                }
}

//-----------------------------------------------------------
// 4) c[i*N + j] = ( (a[i*N + j]+b[j*N + i])^2 ) / (1 + a[i*N + j]^2 )
//-----------------------------------------------------------
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

template <typename T>
void sumBlocking4(T* a, T* b, T* c, int N, int block_size)
{
    a[0] = static_cast<T>(0);
    a[1] = static_cast<T>(1);

    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < N; j += block_size)
            for (int ii = i; ii < std::min(i + block_size, N); ++ii)
                for (int jj = j; jj < std::min(j + block_size, N); ++jj)
                {
                    T s = a[ii * N + jj] + b[jj * N + ii];
                    T s2 = s * s;
                    T denom = static_cast<T>(1) + a[ii * N + jj] * a[ii * N + jj];

                    c[ii * N + jj] = s2 / denom;

                    if ((ii * N + jj) < N * N - 1)
                        a[ii * N + jj + 1] += s / denom;
                }
}

//-----------------------------------------------------------
// SSE-версия (float) sumBlockingIntrSquare
//   c[i*N + j] = (a[i*N + j])^2 + (b[j*N + i])^2
//   с блочным проходом
//-----------------------------------------------------------

// Специализация шаблона для float:
template <>
void sumBlockingIntrSquare<float>(float* a, float* b, float* c, int N, int block_size)
{
    for (int i = 0; i < N; i += block_size)
    {
        for (int j = 0; j < N; j += block_size)
        {
            for (int ii = i; ii < std::min(i + block_size, N); ++ii)
            {
                for (int jj = j; jj < std::min(j + block_size, N); jj += 4)
                {
                    if (jj + 3 >= std::min(j + block_size, N))
                    {
                        for (int x = jj; x < std::min(j + block_size, N); x++)
                        {
                            float Aij = a[ii * N + x];
                            float Bij = b[x * N + ii];
                            c[ii * N + x] = Aij * Aij + Bij * Bij;
                        }
                        break;
                    }

                    
                    __m128 Aij = _mm_loadu_ps(&a[ii * N + jj]);                    
                    float tmp0 = b[(jj + 0) * N + ii];
                    float tmp1 = b[(jj + 1) * N + ii];
                    float tmp2 = b[(jj + 2) * N + ii];
                    float tmp3 = b[(jj + 3) * N + ii];
                    __m128 Bij = _mm_set_ps(tmp3, tmp2, tmp1, tmp0);                    
                    __m128 A2 = _mm_mul_ps(Aij, Aij);
                    __m128 B2 = _mm_mul_ps(Bij, Bij);
                    __m128 sum = _mm_add_ps(A2, B2);
                    _mm_storeu_ps(&c[ii * N + jj], sum);
                }
            }
        }
    }
}

int main()
{    
    std::vector<int> sizesN = { 1024, 2048 };
    std::vector<int> blockSizes = { 16, 32, 64, 128, 256, 512 };
    // Переключатель double/false
    bool useDouble = true;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Matrix Summation Experiment\n\n";

    for (auto N : sizesN)
    {
        std::cout << "========== N = " << N << " ==========\n";

        if (useDouble)
        {
            std::vector<double> A(N * N), B(N * N), C(N * N);
            std::mt19937 gen(0);
            std::uniform_real_distribution<double> dist(0.01, 5.0);
            for (int i = 0; i < N * N; i++) {
                A[i] = dist(gen);
                B[i] = dist(gen);
            }

            // ----------- 1) sumUsualSquare -----------
            {
                auto Acopy = A;
                auto Bcopy = B;
                auto Ccopy = C;
                double t = measureAndRun(sumUsualSquare<double>,
                    Acopy.data(), Bcopy.data(), Ccopy.data(), N);
                std::cout << "sumUsualSquare time = " << t << " s\n";
            }

            // ----------- 2) sumBlockingSquare -----------
            std::cout << "BlockSize | time(blockSquare)\n";
            for (auto bs : blockSizes)
            {
                auto Acopy = A;
                auto Bcopy = B;
                auto Ccopy = C;
                double t = measureAndRunWithBlock(sumBlockingSquare<double>,
                    Acopy.data(), Bcopy.data(), Ccopy.data(),
                    N, bs);
                std::cout << "   " << bs << "     | " << t << "\n";
            }
            std::cout << "\n";

            // ----------- 3) sumUsualSinCos -----------
            {
                auto Acopy = A;
                auto Bcopy = B;
                auto Ccopy = C;
                double t = measureAndRun(sumUsualSinCos<double>,
                    Acopy.data(), Bcopy.data(), Ccopy.data(), N);
                std::cout << "sumUsualSinCos time = " << t << " s\n";
            }

            // ----------- 4) sumBlockingSinCos -----------
            std::cout << "BlockSize | time(blockSinCos)\n";
            for (auto bs : blockSizes)
            {
                auto Acopy = A;
                auto Bcopy = B;
                auto Ccopy = C;
                double t = measureAndRunWithBlock(sumBlockingSinCos<double>,
                    Acopy.data(), Bcopy.data(), Ccopy.data(),
                    N, bs);
                std::cout << "   " << bs << "     | " << t << "\n";
            }
            std::cout << "\n";

            // ----------- 5) sumUsualLogMul -----------
            {
                auto Acopy = A;
                auto Bcopy = B;
                auto Ccopy = C;
                double t = measureAndRun(sumUsualLogMul<double>,
                    Acopy.data(), Bcopy.data(), Ccopy.data(), N);
                std::cout << "sumUsualLogMul time = " << t << " s\n";
            }

            // ----------- 6) sumBlockingLogMul -----------
            std::cout << "BlockSize | time(blockLogMul)\n";
            for (auto bs : blockSizes)
            {
                auto Acopy = A;
                auto Bcopy = B;
                auto Ccopy = C;
                double t = measureAndRunWithBlock(sumBlockingLogMul<double>,
                    Acopy.data(), Bcopy.data(), Ccopy.data(),
                    N, bs);
                std::cout << "   " << bs << "     | " << t << "\n";
            }
            std::cout << "\n";

            // ----------- 7) sumUsual4 -----------
            {
                auto Acopy = A;
                auto Bcopy = B;
                auto Ccopy = C;
                double t = measureAndRun(sumUsual4<double>,
                    Acopy.data(), Bcopy.data(), Ccopy.data(), N);
                std::cout << "sumUsual4 time = " << t << " s\n";
            }

            // ----------- 8) sumBlocking4 -----------
            std::cout << "BlockSize | time(block4)\n";
            for (auto bs : blockSizes)
            {
                auto Acopy = A;
                auto Bcopy = B;
                auto Ccopy = C;
                double t = measureAndRunWithBlock(sumBlocking4<double>,
                    Acopy.data(), Bcopy.data(), Ccopy.data(),
                    N, bs);
                std::cout << "   " << bs << "     | " << t << "\n";
            }
            std::cout << "\n";
        }
        else
        {
            std::vector<float> A(N* N), B(N* N), C(N* N);
            std::mt19937 gen(0);
            std::uniform_real_distribution<float> dist(0.01f, 10.0f);
            for (int i = 0; i < N * N; ++i)
            {
                A[i] = dist(gen);
                B[i] = dist(gen);
            }

            auto A_copy = A;
            auto B_copy = B;
            auto C_copy = C;
            double tUsualSq = measureAndRun(
                sumUsualSquare<float>,             
                A_copy.data(), B_copy.data(),
                C_copy.data(), N
            );
            std::cout << "sumUsualSquare(float) time = " << tUsualSq << " sec\n";

            std::cout << "BlockSize | sumBlockingSquare(float) | speedUp\n";
            for (auto bs : blockSizes)
            {
                A_copy = A;
                B_copy = B;
                C_copy = C;
                double tBlSq = measureAndRunWithBlock(
                    sumBlockingSquare<float>,       
                    A_copy.data(), B_copy.data(),
                    C_copy.data(), N, bs
                );

                double speedUpSq = (tBlSq > 1e-12) ? (tUsualSq / tBlSq) : 0.0;
                std::cout << bs << " | " << tBlSq << " | " << speedUpSq << "\n";
            }
            std::cout << "\n";

            std::cout << "BlockSize | sumBlockingIntrSquare(float) | speedUp\n";
            for (auto bs : blockSizes)
            {
                A_copy = A;
                B_copy = B;
                C_copy = C;
                double tIntrSq = measureAndRunWithBlock(
                    sumBlockingIntrSquare<float>,   
                    A_copy.data(), B_copy.data(),
                    C_copy.data(), N, bs
                );

                double speedUpIntr = (tIntrSq > 1e-12) ? (tUsualSq / tIntrSq) : 0.0;
                std::cout << bs << " | " << tIntrSq << " | " << speedUpIntr << "\n";
            }
            std::cout << "\n";


            {
                A_copy = A; B_copy = B; C_copy = C;
                double tSinCos = measureAndRun(
                    sumUsualSinCos<float>,
                    A_copy.data(), B_copy.data(),
                    C_copy.data(), N
                );
                std::cout << "sumUsualSinCos(float) = " << tSinCos << " s\n";

                std::cout << "BlockSize | sumBlockingSinCos(float) | speedUp\n";
                for (auto bs : blockSizes)
                {
                    A_copy = A; B_copy = B; C_copy = C;
                    double tBlSinCos = measureAndRunWithBlock(
                        sumBlockingSinCos<float>,
                        A_copy.data(), B_copy.data(),
                        C_copy.data(), N, bs
                    );
                    double sp = (tBlSinCos > 1e-12) ? (tSinCos / tBlSinCos) : 0.0;
                    std::cout << bs << " | " << tBlSinCos << " | " << sp << "\n";
                }
                std::cout << "\n";
            }
        }
    }

    return 0;
}
