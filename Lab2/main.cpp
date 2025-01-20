#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>   
#include <cmath>     
#include <omp.h>
#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <string>

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
#ifdef _MSC_VER

#include <intrin.h>
#else
#include <immintrin.h>
#endif
enum class DEVICE_MODE
{
    CPU = 1,
    GPU = 2
};
DEVICE_MODE device_mode = DEVICE_MODE::CPU;
void multSerial(float* C, const float* A, const float* B, int N)
{
    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < N; c++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[r * N + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

void multParallel(float* C, const float* A, const float* B, int N)
{
#pragma omp parallel for
    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < N; c++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[r * N + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

// Блочное умножение (с OpenMP)
void multBlock(float* C, const float* A, const float* B, int N)
{
    const int BLOCK_SIZE = 32;
#pragma omp parallel for
    for (int iBlock = 0; iBlock < N; iBlock += BLOCK_SIZE)
    {
        for (int jBlock = 0; jBlock < N; jBlock += BLOCK_SIZE)
        {
            for (int kBlock = 0; kBlock < N; kBlock += BLOCK_SIZE)
            {
                for (int i = iBlock; i < iBlock + BLOCK_SIZE && i < N; i++)
                {
                    for (int j = jBlock; j < jBlock + BLOCK_SIZE && j < N; j++)
                    {
                        float sum = 0.0f;
                        for (int k = kBlock; k < kBlock + BLOCK_SIZE && k < N; k++)
                        {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] += sum;
                    }
                }
            }
        }
    }
}

#if defined(__AVX2__)  // SIMD (AVX2)
void multSIMD(float* C, const float* A, const float* B, int N)
{
    // 1) Транспонируем B -> B_T (размер N*N)
    std::vector<float> B_T(N * N);
    // B_T[col * N + row] = B[row * N + col], но тут row, col = i, j
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            B_T[j * N + i] = B[i * N + j];
        }
    }

#pragma omp parallel for
    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < N; c++)
        {
            __m256 sumVec = _mm256_setzero_ps();
            int k = 0;
            for (; k + 7 < N; k += 8)
            {
                __m256 aVec = _mm256_loadu_ps(&A[r * N + k]);
                __m256 bVec = _mm256_loadu_ps(&B_T[c * N + k]);
                __m256 mulVec = _mm256_mul_ps(aVec, bVec);
                sumVec = _mm256_add_ps(sumVec, mulVec);
            }
            float partial[8];
            _mm256_storeu_ps(partial, sumVec);
            float sum = 0.0f;
            for (int idx = 0; idx < 8; idx++)
            {
                sum += partial[idx];
            }
            for (; k < N; k++)
            {
                sum += A[r * N + k] * B_T[c * N + k];
            }
            C[r * N + c] = sum;
        }
    }
}
#else
void multSIMD(float* C, const float* A, const float* B, int N)
{
    multParallel(C, A, B, N);
}
#endif

bool sameArrays(const float* pC1, const float* pC2, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float v1 = pC1[i * N + j];
            float v2 = pC2[i * N + j];
            if (fabs(v1 - v2) > 1e-2)
                return false;
        }
    }
    return true;
}

std::string loadKernelFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

const char* getKernelSource(int kernel_id, size_t& src_size)
{
    std::string filename;
    switch (kernel_id)
    {
    case 0: filename = "kernel0.cl"; break;
    case 1: filename = "kernel1.cl"; break;
    case 2: filename = "kernel2.cl"; break;
    case 3: filename = "kernel3.cl"; break;
    case 4: filename = "kernel4.cl"; break;
    case 5: filename = "kernel5.cl"; break;
    default: filename = "kernel0.cl"; break;
    }

    static std::string kernelSource;  
    kernelSource = loadKernelFile(filename);

    src_size = kernelSource.size();
    return kernelSource.c_str();  
}


void multWithOpenCL(float* pC, const float* pA, const float* pB,
    int N, int witems_deg, int gsize, int kernel_id)
{
    cl_int ret;
    cl_platform_id platform_id = NULL;
    cl_uint num_platforms = 0;

    ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (ret != CL_SUCCESS)
    {
        std::cerr << "Error: clGetPlatformIDs\n";
        return;
    }

    cl_device_id device_id = NULL;
    cl_uint num_devices = 0;
    switch (device_mode)
    {
    case DEVICE_MODE::CPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &num_devices);
        break;
    case DEVICE_MODE::GPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
        break;
    default:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices);
        break;
    }
    if (ret != CL_SUCCESS)
    {
        std::cerr << "Error: clGetDeviceIDs\n";
        return;
    }

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        std::cerr << "Error: clCreateContext\n";
        return;
    }

    cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);
    if (ret != CL_SUCCESS)
    {
        std::cerr << "Error: clCreateCommandQueue\n";
        clReleaseContext(context);
        return;
    }

    size_t sizeBytes = (size_t)N * (size_t)N * sizeof(float);
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeBytes, NULL, &ret);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeBytes, NULL, &ret);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, bufA, CL_TRUE, 0, sizeBytes, pA, 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, bufB, CL_TRUE, 0, sizeBytes, pB, 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, bufC, CL_TRUE, 0, sizeBytes, pC, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        std::cerr << "Error: clEnqueueWriteBuffer\n";
    }

    size_t src_size = 0;
    const char* src = getKernelSource(kernel_id, src_size);
    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_size, &ret);
    if (ret != CL_SUCCESS)
    {
        std::cerr << "Error: clCreateProgramWithSource\n";
    }

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* buildLog = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, buildLog, NULL);
        buildLog[log_size] = '\0';
        std::cerr << "Build log:\n" << buildLog << "\n";
        free(buildLog);
    }

    cl_kernel kernel = clCreateKernel(program, "MultKernel", &ret);
    if (ret != CL_SUCCESS)
    {
        std::cerr << "Error: clCreateKernel\n";
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufC);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufA);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufB);
    ret |= clSetKernelArg(kernel, 3, sizeof(int), &N);

    // Для тайлового ядра (kernel_id=5) нужно 2D NDRange, иначе 1D
    if (kernel_id == 5)
    {
        // 2D
        size_t TS = 32;
        size_t globalSizeX = (N % TS == 0) ? N : (N + TS - (N % TS));
        size_t globalSizeY = (N % TS == 0) ? N : (N + TS - (N % TS));
        size_t globalSize[2] = { globalSizeX, globalSizeY };
        size_t localSize[2] = { TS, TS };
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
            globalSize, localSize,
            0, NULL, NULL);
    }
    else
    {
        // 1D
        size_t globalSize = (witems_deg == 1)
            ? (size_t)N
            : (size_t)N * (size_t)N;
        size_t localSize = gsize;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &globalSize, &localSize,
            0, NULL, NULL);
    }
    if (ret != CL_SUCCESS)
    {
        std::cerr << "Error: clEnqueueNDRangeKernel\n";
    }

    ret = clEnqueueReadBuffer(command_queue, bufC, CL_TRUE, 0, sizeBytes, pC, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        std::cerr << "Error: clEnqueueReadBuffer\n";
    }

    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

void printTableRow(int N, int workGroupSize,
    double tSerial,
    double tOmp,
    double tOpenCL,
    double tBlock,
    double tSimd)
{
    std::cout << "| " << N << "\t| "
        << workGroupSize << "\t| "
        << tSerial << "\t| "
        << tOmp << "\t| "
        << tOpenCL << "\t| "
        << tBlock << "\t| "
        << tSimd << "\t|\n";
}

int main()
{
    std::srand((unsigned)std::time(nullptr));
    std::setlocale(LC_ALL, "Russian");

    int N;
    std::cout << "Введите размер матрицы (N): ";
    std::cin >> N;

    std::cout << "Выберите устройство (1=CPU, 2=GPU): ";
    int mode;
    std::cin >> mode;
    device_mode = (mode == 2) ? DEVICE_MODE::GPU : DEVICE_MODE::CPU;

    std::cout << "Введите степень witems (1 или 2): ";
    int witems_deg;
    std::cin >> witems_deg;

    std::cout << "Введите размер группы (local size): ";
    int g_size;
    std::cin >> g_size;

    std::cout << "Введите номер ядра (0..5): ";
    int k_id;
    std::cin >> k_id;

    std::vector<float> A(N * N), B(N * N), C1(N * N), C2(N * N), Ctmp(N * N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            B[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            C1[i * N + j] = 0.0f;
            C2[i * N + j] = 0.0f;
            Ctmp[i * N + j] = 0.0f;
        }
    }

    // 1) Serial
    double t1 = omp_get_wtime();
    multSerial(C1.data(), A.data(), B.data(), N);
    double t2 = omp_get_wtime();
    double tSerial = t2 - t1;

    // 2) OMP
    for (int i = 0; i < N * N; i++) C1[i] = 0.0f;
    t1 = omp_get_wtime();
    multParallel(C1.data(), A.data(), B.data(), N);
    t2 = omp_get_wtime();
    double tOmp = t2 - t1;

    // 3) OpenCL
    for (int i = 0; i < N * N; i++) C2[i] = 0.0f;
    t1 = omp_get_wtime();
    multWithOpenCL(C2.data(), A.data(), B.data(), N, witems_deg, g_size, k_id);
    t2 = omp_get_wtime();
    double tOpenCL = t2 - t1;
    bool correctOpenCL = sameArrays(C1.data(), C2.data(), N);

    // 4) Блочное (обнулим Ctmp перед вызовом)
    for (int i = 0; i < N * N; i++) Ctmp[i] = 0.0f;
    t1 = omp_get_wtime();
    multBlock(Ctmp.data(), A.data(), B.data(), N);
    t2 = omp_get_wtime();
    double tBlock = t2 - t1;
    bool correctBlock = sameArrays(C1.data(), Ctmp.data(), N);

    // 5) SIMD (если AVX2 есть)
    for (int i = 0; i < N * N; i++) Ctmp[i] = 0.0f;
    t1 = omp_get_wtime();
    multSIMD(Ctmp.data(), A.data(), B.data(), N);
    t2 = omp_get_wtime();
    double tSimd = t2 - t1;
    bool correctSimd = sameArrays(C1.data(), Ctmp.data(), N);

    std::cout << "\n--- РЕЗУЛЬТАТЫ ---\n";
    std::cout << "Time Serial = " << tSerial << " s\n";
    std::cout << "Time OMP    = " << tOmp << " s\n";
    std::cout << "Time OpenCL = " << tOpenCL << " s    (kernel_id=" << k_id << ")\n";
    std::cout << "Time Block  = " << tBlock << " s\n";
    std::cout << "Time SIMD   = " << tSimd << " s\n";

    std::cout << "\nКорректность:\n";
    std::cout << "  OpenCL -> " << (correctOpenCL ? "OK" : "FAIL") << "\n";
    std::cout << "  Block  -> " << (correctBlock ? "OK" : "FAIL") << "\n";
    std::cout << "  SIMD   -> " << (correctSimd ? "OK" : "FAIL") << "\n";

    std::cout << "\nТаблица (одна строка):\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "| Size\t| WG\t| Serial\t| OMP\t| OpenCL\t| Block\t| SIMD\t|\n";
    std::cout << "-----------------------------------------------------------------\n";
    printTableRow(N, g_size, tSerial, tOmp, tOpenCL, tBlock, tSimd);
    std::cout << "-----------------------------------------------------------------\n";

    return 0;
}
