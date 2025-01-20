#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <CL/cl.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)

enum DEVICE_MODE
{
    CPU = 1,
    GPU = 2
};

DEVICE_MODE device_mode = CPU;

void calcNorm2DSerial(const float* a, const float* b, float* out, size_t n);
void calcNorm2DOMP(const float* a, const float* b, float* out, size_t n);

void calcNorm2DOpenCL(const float* a, const float* b, float* out,
    size_t n, size_t workGroupSize,
    double* calcTime);

void calcPolySerial(const float* x, float* out, size_t n);
void calcPolyOMP(const float* x, float* out, size_t n);

void calcPolyOpenCL(const float* x, float* out, size_t n,
    size_t workGroupSize,
    double* calcTime);

void subsetsMeanSerial(const float* X, float* out, size_t Nsubs,
    const vector<int>& starts, int subsetSize);
void subsetsMeanOMP(const float* X, float* out, size_t Nsubs,
    const vector<int>& starts, int subsetSize);

void subsetsMeanOpenCL(const float* X, const int* starts,
    float* out, size_t Nsubs,
    int subsetSize, size_t workGroupSize,
    double* calcTime);

void movingAvgSerial(const float* inArr, float* outArr, size_t n, int windowSize);
void movingAvgOMP(const float* inArr, float* outArr, size_t n, int windowSize);

void movingAvgOpenCL(const float* inArr, float* outArr, size_t n,
    int windowSize, size_t workGroupSize,
    double* calcTime);

void printTableRow(size_t arraySize, size_t workGroupSize,
    double tSerial, double tOmp, double tOpenCL)
{
    cout << "| " << arraySize << "\t| "
        << workGroupSize << "\t| "
        << tSerial << "\t| "
        << tOmp << "\t| "
        << tOpenCL << "\t|\n";
}

int main()
{
    setlocale(LC_ALL, "Russian");
    srand((unsigned)time(0));

    cout << "Выберите задание:\n"
        << " 1) Нормы 2D-вектора\n"
        << " 2) Полином одной переменной\n"
        << " 3) Среднее значения f(x) для подмножеств\n"
        << " 4) Скользящее среднее\n";
    int taskNumber;
    cin >> taskNumber;

    cout << "Выберите устройство для OpenCL:\n"
        << " 1 - CPU\n"
        << " 2 - GPU\n";
    int mode;
    cin >> mode;
    device_mode = (DEVICE_MODE)mode;

    int subsetSize = 0;
    size_t nSubsets = 0;
    if (taskNumber == 3)
    {
        cout << "Укажите размер каждого подмножества: ";
        cin >> subsetSize;
        cout << "Укажите количество подмножеств: ";
        cin >> nSubsets;
    }

    int windowSize = 0;
    if (taskNumber == 4)
    {
        cout << "Укажите размер окна скользящего среднего: ";
        cin >> windowSize;
    }

    cout << "Сколько различных размеров (arraySize) будем тестировать? ";
    int countSizes;
    cin >> countSizes;
    vector<size_t> sizes(countSizes);
    for (int i = 0; i < countSizes; i++)
    {
        cout << "Введите размер " << (i + 1) << ": ";
        cin >> sizes[i];
    }

    cout << "Сколько вариантов Work Group Size? ";
    int countWG;
    cin >> countWG;
    vector<size_t> groupSizes(countWG);
    for (int i = 0; i < countWG; i++)
    {
        cout << "Введите WG Size " << (i + 1) << ": ";
        cin >> groupSizes[i];
    }

    cout << "\nРЕЗУЛЬТАТЫ ДЛЯ ЗАДАНИЯ №" << taskNumber << ":\n";
    cout << "-------------------------------------------------------\n";
    cout << "| Size\t| WG\t| Serial\t| OMP\t| OpenCL\t|\n";
    cout << "-------------------------------------------------------\n";

    for (auto n : sizes)
    {
        vector<float> A(n), B(n), OutSerial(n), OutOmp(n), OutOcl(n);
        vector<float> OutSerialSubs(nSubsets), OutOmpSubs(nSubsets), OutOclSubs(nSubsets);

        for (size_t i = 0; i < n; i++)
        {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
        }

        vector<int> starts(nSubsets);
        if (taskNumber == 3)
        {
            for (size_t k = 0; k < nSubsets; k++)
                starts[k] = (int)(k * subsetSize);
        }

        for (auto wg : groupSizes)
        {
            double tSerial = 0.0, tOmp = 0.0, tOpenCL = 0.0;

            {
                auto t1 = chrono::high_resolution_clock::now();
                switch (taskNumber)
                {
                case 1:
                    calcNorm2DSerial(A.data(), B.data(), OutSerial.data(), n);
                    break;
                case 2:
                    calcPolySerial(A.data(), OutSerial.data(), n);
                    break;
                case 3:
                    subsetsMeanSerial(A.data(), OutSerialSubs.data(),
                        nSubsets, starts, subsetSize);
                    break;
                case 4:
                    movingAvgSerial(A.data(), OutSerial.data(), n, windowSize);
                    break;
                }
                auto t2 = chrono::high_resolution_clock::now();
                tSerial = chrono::duration<double>(t2 - t1).count();
            }

            {
                auto t1 = chrono::high_resolution_clock::now();
                switch (taskNumber)
                {
                case 1:
                    calcNorm2DOMP(A.data(), B.data(), OutOmp.data(), n);
                    break;
                case 2:
                    calcPolyOMP(A.data(), OutOmp.data(), n);
                    break;
                case 3:
                    subsetsMeanOMP(A.data(), OutOmpSubs.data(),
                        nSubsets, starts, subsetSize);
                    break;
                case 4:
                    movingAvgOMP(A.data(), OutOmp.data(), n, windowSize);
                    break;
                }
                auto t2 = chrono::high_resolution_clock::now();
                tOmp = chrono::duration<double>(t2 - t1).count();
            }

            
            {
                
                switch (taskNumber)
                {
                case 1:
                    calcNorm2DOpenCL(A.data(), B.data(), OutOcl.data(), n,
                        wg, &tOpenCL);
                    break;
                case 2:
                    calcPolyOpenCL(A.data(), OutOcl.data(), n, wg, &tOpenCL);
                    break;
                case 3:
                    subsetsMeanOpenCL(A.data(), starts.data(),
                        OutOclSubs.data(), nSubsets,
                        subsetSize, wg, &tOpenCL);
                    break;
                case 4:
                    movingAvgOpenCL(A.data(), OutOcl.data(), n,
                        windowSize, wg, &tOpenCL);
                    break;
                }
            }

            printTableRow(n, wg, tSerial, tOmp, tOpenCL);
        }
    }

    cout << "-------------------------------------------------------\n";
    cout << "Программа завершена.\n";
    return 0;
}


void calcNorm2DSerial(const float* a, const float* b, float* out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = sqrt(a[i] * a[i] + b[i] * b[i]);
}

void calcNorm2DOMP(const float* a, const float* b, float* out, size_t n)
{
#pragma omp parallel for
    for (int i = 0; i < (int)n; i++)
        out[i] = sqrt(a[i] * a[i] + b[i] * b[i]);
}

void calcPolySerial(const float* x, float* out, size_t n)
{
    for (size_t idx = 0; idx < n; idx++)
    {
        float val = 0.0f;
        for (int i = 1; i <= 9; i++)
            val += powf(x[idx], (float)i) / i;
        out[idx] = val;
    }
}

void calcPolyOMP(const float* x, float* out, size_t n)
{
#pragma omp parallel for
    for (int idx = 0; idx < (int)n; idx++)
    {
        float val = 0.0f;
        for (int i = 1; i <= 9; i++)
            val += powf(x[idx], (float)i) / i;
        out[idx] = val;
    }
}

void subsetsMeanSerial(const float* X, float* out, size_t Nsubs,
    const vector<int>& starts, int subsetSize)
{
    for (size_t k = 0; k < Nsubs; k++)
    {
        float sum = 0.0f;
        int startIdx = starts[k];
        for (int j = 0; j < subsetSize; j++)
            sum += (X[startIdx + j] * X[startIdx + j]);
        out[k] = sum / subsetSize;
    }
}

void subsetsMeanOMP(const float* X, float* out, size_t Nsubs,
    const vector<int>& starts, int subsetSize)
{
#pragma omp parallel for
    for (int k = 0; k < (int)Nsubs; k++)
    {
        float sum = 0.0f;
        int startIdx = starts[k];
        for (int j = 0; j < subsetSize; j++)
            sum += (X[startIdx + j] * X[startIdx + j]);
        out[k] = sum / subsetSize;
    }
}

void movingAvgSerial(const float* inArr, float* outArr, size_t n, int windowSize)
{
    for (size_t i = 0; i < n; i++)
    {
        if (i + windowSize <= n)
        {
            float sum = 0.0f;
            for (int j = 0; j < windowSize; j++)
                sum += inArr[i + j];
            outArr[i] = sum / windowSize;
        }
        else
        {
            outArr[i] = 0.0f;
        }
    }
}

void movingAvgOMP(const float* inArr, float* outArr, size_t n, int windowSize)
{
#pragma omp parallel for
    for (int i = 0; i < (int)n; i++)
    {
        if (i + windowSize <= (int)n)
        {
            float sum = 0.0f;
            for (int j = 0; j < windowSize; j++)
                sum += inArr[i + j];
            outArr[i] = sum / windowSize;
        }
        else
        {
            outArr[i] = 0.0f;
        }
    }
}

void calcNorm2DOpenCL(const float* a,
    const float* b,
    float* out,
    size_t n,
    size_t workGroupSize,
    double* calcTime)
{
    cl_int ret = 0;
    cl_platform_id platform_id = nullptr;
    cl_uint ret_num_platforms = 0;

    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices = 0;

    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;

    cl_mem a_mem = nullptr;
    cl_mem b_mem = nullptr;
    cl_mem o_mem = nullptr;

    FILE* fp = nullptr;
    char* source_str = nullptr;
    size_t source_size = 0;

    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_int errcode_ret = 0;

    int n_int = 0;
    size_t global_work_size = 0;

    auto t1 = std::chrono::high_resolution_clock::time_point{};
    auto t2 = std::chrono::high_resolution_clock::time_point{};

    if (calcTime) {
        *calcTime = 0.0;
    }


    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS || ret_num_platforms == 0) {
        cerr << "[ERROR] Не удалось получить OpenCL-платформу!\n";
        goto clean_exit;
    }

    switch (device_mode)
    {
    case CPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        break;
    case GPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
        break;
    default:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
        break;
    }
    if (ret != CL_SUCCESS || ret_num_devices == 0) {
        cerr << "[ERROR] Не найдено подходящее устройство OpenCL!\n";
        goto clean_exit;
    }

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateContext failed!\n";
        goto clean_exit;
    }

    {
        cl_command_queue_properties props[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
        };
        command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &errcode_ret);
        if (errcode_ret != CL_SUCCESS) {
            cerr << "[ERROR] clCreateCommandQueueWithProperties failed!\n";
            goto clean_exit;
        }
    }

    a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &errcode_ret);
    if (!a_mem || errcode_ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(a_mem) failed!\n";
        goto clean_exit;
    }
    b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &errcode_ret);
    if (!b_mem || errcode_ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(b_mem) failed!\n";
        goto clean_exit;
    }
    o_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &errcode_ret);
    if (!o_mem || errcode_ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(o_mem) failed!\n";
        goto clean_exit;
    }

    ret = clEnqueueWriteBuffer(command_queue, a_mem, CL_TRUE,
        0, n * sizeof(float),
        a, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        cerr << "[ERROR] clEnqueueWriteBuffer(a_mem) failed!\n";
        goto clean_exit;
    }
    ret = clEnqueueWriteBuffer(command_queue, b_mem, CL_TRUE,
        0, n * sizeof(float),
        b, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        cerr << "[ERROR] clEnqueueWriteBuffer(b_mem) failed!\n";
        goto clean_exit;
    }

    {
        fp = fopen("Kernel.cl", "r");
        if (!fp) {
            cerr << "[ERROR] Не удалось открыть Kernel.cl\n";
            goto clean_exit;
        }
        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        if (!source_str) {
            cerr << "[ERROR] malloc(source_str) failed!\n";
            goto clean_exit;
        }
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);
        fp = nullptr;

        program = clCreateProgramWithSource(context, 1,
            (const char**)&source_str,
            &source_size, &errcode_ret);
        free(source_str);
        source_str = nullptr;

        if (errcode_ret != CL_SUCCESS || !program) {
            cerr << "[ERROR] clCreateProgramWithSource failed!\n";
            goto clean_exit;
        }
    }

    errcode_ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        cerr << "[ERROR] clBuildProgram failed!\n";
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_id,
            CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
        if (log_size) {
            char* logbuf = (char*)malloc(log_size + 1);
            clGetProgramBuildInfo(program, device_id,
                CL_PROGRAM_BUILD_LOG,
                log_size, logbuf, NULL);
            logbuf[log_size] = '\0';
            cerr << "[BUILD LOG]\n" << logbuf << endl;
            free(logbuf);
        }
        goto clean_exit;
    }

    kernel = clCreateKernel(program, "norm2D", &errcode_ret);
    if (errcode_ret != CL_SUCCESS || !kernel) {
        cerr << "[ERROR] clCreateKernel(norm2D) failed!\n";
        goto clean_exit;
    }

    n_int = (int)n;
    errcode_ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    errcode_ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    errcode_ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &o_mem);
    errcode_ret |= clSetKernelArg(kernel, 3, sizeof(int), &n_int);
    if (errcode_ret != CL_SUCCESS) {
        cerr << "[ERROR] clSetKernelArg failed!\n";
        goto clean_exit;
    }

    global_work_size = n;

    t1 = std::chrono::high_resolution_clock::now();

    ret = clEnqueueNDRangeKernel(command_queue, kernel,
        1,
        nullptr,
        &global_work_size,
        &workGroupSize,
        0, nullptr, nullptr);
    clFinish(command_queue);

    ret = clEnqueueReadBuffer(command_queue, o_mem, CL_TRUE,
        0, n * sizeof(float),
        out, 0, nullptr, nullptr);

    t2 = std::chrono::high_resolution_clock::now();

    if (calcTime) {
        *calcTime = std::chrono::duration<double>(t2 - t1).count();
    }

clean_exit:
    if (kernel)        clReleaseKernel(kernel);
    if (program)       clReleaseProgram(program);
    if (fp)            fclose(fp);
    if (source_str)    free(source_str);

    if (a_mem)         clReleaseMemObject(a_mem);
    if (b_mem)         clReleaseMemObject(b_mem);
    if (o_mem)         clReleaseMemObject(o_mem);
    if (command_queue) clReleaseCommandQueue(command_queue);
    if (context)       clReleaseContext(context);
}


void calcPolyOpenCL(const float* x,
    float* out,
    size_t n,
    size_t workGroupSize,
    double* calcTime)
{
    cl_int ret = 0;
    cl_platform_id platform_id = nullptr;
    cl_uint ret_num_platforms = 0;

    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices = 0;

    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;

    cl_mem x_mem = nullptr;
    cl_mem out_mem = nullptr;

    FILE* fp = nullptr;
    char* source_str = nullptr;
    size_t source_size = 0;

    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_int errcode_ret = 0;
    int n_int = 0;
    size_t global_work_size = 0;
    size_t local_work_size = 0;

    auto t1 = std::chrono::high_resolution_clock::time_point{};
    auto t2 = std::chrono::high_resolution_clock::time_point{};

    if (calcTime) {
        *calcTime = 0.0;
    }

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS || ret_num_platforms == 0) {
        cerr << "[ERROR] No valid platform!\n";
        goto CLEANUP;
    }

    switch (device_mode)
    {
    case CPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        break;
    case GPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
        break;
    default:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
        break;
    }
    if (ret != CL_SUCCESS || ret_num_devices == 0) {
        cerr << "[ERROR] No valid device!\n";
        goto CLEANUP;
    }

    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS || !context) {
        cerr << "[ERROR] clCreateContext failed!\n";
        goto CLEANUP;
    }

    {
        cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);
        if (ret != CL_SUCCESS || !command_queue) {
            cerr << "[ERROR] clCreateCommandQueueWithProperties failed!\n";
            goto CLEANUP;
        }
    }

    x_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &ret);
    if (!x_mem || ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(x_mem) failed!\n";
        goto CLEANUP;
    }
    out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &ret);
    if (!out_mem || ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(out_mem) failed!\n";
        goto CLEANUP;
    }

    ret = clEnqueueWriteBuffer(command_queue, x_mem, CL_TRUE,
        0, n * sizeof(float),
        x, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        cerr << "[ERROR] clEnqueueWriteBuffer(x_mem) failed!\n";
        goto CLEANUP;
    }

    fp = fopen("Kernel.cl", "r");
    if (!fp) {
        cerr << "[ERROR] Cannot open Kernel.cl\n";
        goto CLEANUP;
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    if (!source_str) {
        cerr << "[ERROR] malloc(source_str) failed!\n";
        goto CLEANUP;
    }
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    fp = nullptr;

    program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
        &source_size, &ret);
    free(source_str);
    source_str = nullptr;
    if (ret != CL_SUCCESS || !program) {
        cerr << "[ERROR] clCreateProgramWithSource failed!\n";
        goto CLEANUP;
    }

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size > 0) {
            vector<char> logBuf(log_size + 1);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, logBuf.data(), NULL);
            logBuf[log_size] = '\0';
            cerr << logBuf.data() << endl;
        }
    }

    kernel = clCreateKernel(program, "polynomial", &ret);
    if (ret != CL_SUCCESS || !kernel) {
        cerr << "[ERROR] clCreateKernel(polynomial) failed!\n";
        goto CLEANUP;
    }

    n_int = (int)n;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x_mem);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
    ret |= clSetKernelArg(kernel, 2, sizeof(int), &n_int);
    if (ret != CL_SUCCESS) {
        cerr << "[ERROR] clSetKernelArg failed!\n";
        goto CLEANUP;
    }

    global_work_size = n;
    local_work_size = workGroupSize;

    t1 = std::chrono::high_resolution_clock::now();

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_work_size, &local_work_size,
        0, NULL, NULL);
    clFinish(command_queue);

    ret = clEnqueueReadBuffer(command_queue, out_mem, CL_TRUE,
        0, n * sizeof(float),
        out, 0, NULL, NULL);

    t2 = std::chrono::high_resolution_clock::now();
    if (calcTime) {
        *calcTime = std::chrono::duration<double>(t2 - t1).count();
    }

CLEANUP:
    if (fp)         fclose(fp);
    if (source_str) free(source_str);

    if (kernel)     clReleaseKernel(kernel);
    if (program)    clReleaseProgram(program);

    if (x_mem)      clReleaseMemObject(x_mem);
    if (out_mem)    clReleaseMemObject(out_mem);
    if (command_queue) clReleaseCommandQueue(command_queue);
    if (context)    clReleaseContext(context);
}


void subsetsMeanOpenCL(const float* X,
    const int* starts,
    float* out,
    size_t Nsubs,
    int subsetSize,
    size_t workGroupSize,
    double* calcTime)
{
    cl_int ret = 0;
    cl_platform_id platform_id = nullptr;
    cl_uint ret_num_platforms = 0;

    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices = 0;

    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;

    cl_mem X_mem = nullptr;
    cl_mem starts_mem = nullptr;
    cl_mem out_mem = nullptr;

    FILE* fp = nullptr;
    char* source_str = nullptr;
    size_t source_size = 0;

    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_int errcode_ret = 0;

    int maxIndex = 0;
    int xLength = 0;
    int Nsubs_int = 0;

    size_t global_work_size = 0;
    size_t local_work_size = 0;

    auto t1 = std::chrono::high_resolution_clock::time_point{};
    auto t2 = std::chrono::high_resolution_clock::time_point{};

    if (calcTime) {
        *calcTime = 0.0;
    }

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS || ret_num_platforms == 0) {
        cerr << "[ERROR] No valid platform!\n";
        goto CLEANUP;
    }

    switch (device_mode)
    {
    case CPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        break;
    case GPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
        break;
    default:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
        break;
    }
    if (ret != CL_SUCCESS || ret_num_devices == 0) {
        cerr << "[ERROR] No valid device!\n";
        goto CLEANUP;
    }

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS || !context) {
        cerr << "[ERROR] clCreateContext failed!\n";
        goto CLEANUP;
    }

    {
        cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);
        if (ret != CL_SUCCESS || !command_queue) {
            cerr << "[ERROR] clCreateCommandQueueWithProperties failed!\n";
            goto CLEANUP;
        }
    }

    if (Nsubs > 0) {
        maxIndex = starts[Nsubs - 1];  
    }
    xLength = maxIndex + subsetSize;

    X_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, xLength * sizeof(float), NULL, &ret);
    if (!X_mem || ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(X_mem) failed!\n";
        goto CLEANUP;
    }
    starts_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, Nsubs * sizeof(int), NULL, &ret);
    if (!starts_mem || ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(starts_mem) failed!\n";
        goto CLEANUP;
    }
    out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, Nsubs * sizeof(float), NULL, &ret);
    if (!out_mem || ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(out_mem) failed!\n";
        goto CLEANUP;
    }

    ret = clEnqueueWriteBuffer(command_queue, X_mem, CL_TRUE,
        0, xLength * sizeof(float),
        X, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        cerr << "[ERROR] clEnqueueWriteBuffer(X_mem) failed!\n";
        goto CLEANUP;
    }
    ret = clEnqueueWriteBuffer(command_queue, starts_mem, CL_TRUE,
        0, Nsubs * sizeof(int),
        starts, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        cerr << "[ERROR] clEnqueueWriteBuffer(starts_mem) failed!\n";
        goto CLEANUP;
    }

    fp = fopen("Kernel.cl", "r");
    if (!fp) {
        cerr << "[ERROR] Cannot open Kernel.cl\n";
        goto CLEANUP;
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    if (!source_str) {
        cerr << "[ERROR] malloc(source_str) failed!\n";
        goto CLEANUP;
    }
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    fp = nullptr;

    program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, &source_size, &ret);
    free(source_str);
    source_str = nullptr;
    if (ret != CL_SUCCESS || !program) {
        cerr << "[ERROR] clCreateProgramWithSource failed!\n";
        goto CLEANUP;
    }

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id,
            CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
        if (log_size > 0) {
            vector<char> logBuf(log_size + 1);
            clGetProgramBuildInfo(program, device_id,
                CL_PROGRAM_BUILD_LOG,
                log_size, logBuf.data(), NULL);
            logBuf[log_size] = '\0';
            cerr << logBuf.data() << endl;
        }
    }

    kernel = clCreateKernel(program, "subsetsMeanOptim", &ret);
    if (ret != CL_SUCCESS || !kernel) {
        cerr << "[ERROR] clCreateKernel(subsetsMeanOptim) failed!\n";
        goto CLEANUP;
    }

    Nsubs_int = (int)Nsubs;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_mem);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &starts_mem);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem);
    ret |= clSetKernelArg(kernel, 3, sizeof(int), &Nsubs_int);
    ret |= clSetKernelArg(kernel, 4, sizeof(int), &subsetSize);
    if (ret != CL_SUCCESS) {
        cerr << "[ERROR] clSetKernelArg failed!\n";
        goto CLEANUP;
    }

    global_work_size = Nsubs;
    local_work_size = workGroupSize;

    t1 = std::chrono::high_resolution_clock::now();

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_work_size, &local_work_size,
        0, NULL, NULL);
    clFinish(command_queue);

    ret = clEnqueueReadBuffer(command_queue, out_mem, CL_TRUE,
        0, Nsubs * sizeof(float),
        out, 0, NULL, NULL);

    t2 = std::chrono::high_resolution_clock::now();
    if (calcTime) {
        *calcTime = std::chrono::duration<double>(t2 - t1).count();
    }

CLEANUP:
    if (fp)         fclose(fp);
    if (source_str) free(source_str);
    if (kernel)     clReleaseKernel(kernel);
    if (program)    clReleaseProgram(program);

    if (X_mem)      clReleaseMemObject(X_mem);
    if (starts_mem) clReleaseMemObject(starts_mem);
    if (out_mem)    clReleaseMemObject(out_mem);

    if (command_queue) clReleaseCommandQueue(command_queue);
    if (context)    clReleaseContext(context);
}



void movingAvgOpenCL(const float* inArr,
    float* outArr,
    size_t n,
    int windowSize,
    size_t workGroupSize,
    double* calcTime)
{
    cl_int ret = 0;
    cl_platform_id platform_id = nullptr;
    cl_uint ret_num_platforms = 0;

    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices = 0;

    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;

    cl_mem in_mem = nullptr;
    cl_mem out_mem = nullptr;

    FILE* fp = nullptr;
    char* source_str = nullptr;
    size_t source_size = 0;

    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_int errcode_ret = 0;

    int n_int = 0;
    size_t global_work_size = 0;
    size_t local_work_size = 0;

    auto t1 = std::chrono::high_resolution_clock::time_point{};
    auto t2 = std::chrono::high_resolution_clock::time_point{};

    if (calcTime) {
        *calcTime = 0.0;
    }

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS || ret_num_platforms == 0) {
        cerr << "[ERROR] No valid platform!\n";
        goto CLEANUP;
    }

    switch (device_mode)
    {
    case CPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        break;
    case GPU:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
        break;
    default:
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
        break;
    }
    if (ret != CL_SUCCESS || ret_num_devices == 0) {
        cerr << "[ERROR] No valid device!\n";
        goto CLEANUP;
    }

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS || !context) {
        cerr << "[ERROR] clCreateContext failed!\n";
        goto CLEANUP;
    }

    {
        cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);
        if (ret != CL_SUCCESS || !command_queue) {
            cerr << "[ERROR] clCreateCommandQueueWithProperties failed!\n";
            goto CLEANUP;
        }
    }

    in_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &ret);
    if (!in_mem || ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(in_mem) failed!\n";
        goto CLEANUP;
    }
    out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &ret);
    if (!out_mem || ret != CL_SUCCESS) {
        cerr << "[ERROR] clCreateBuffer(out_mem) failed!\n";
        goto CLEANUP;
    }

    ret = clEnqueueWriteBuffer(command_queue, in_mem, CL_TRUE,
        0, n * sizeof(float),
        inArr, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        cerr << "[ERROR] clEnqueueWriteBuffer(in_mem) failed!\n";
        goto CLEANUP;
    }

    fp = fopen("Kernel.cl", "r");
    if (!fp) {
        cerr << "[ERROR] Cannot open Kernel.cl\n";
        goto CLEANUP;
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    if (!source_str) {
        cerr << "[ERROR] malloc(source_str) failed!\n";
        goto CLEANUP;
    }
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    fp = nullptr;

    program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, &source_size, &ret);
    free(source_str);
    source_str = nullptr;
    if (ret != CL_SUCCESS || !program) {
        cerr << "[ERROR] clCreateProgramWithSource failed!\n";
        goto CLEANUP;
    }

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id,
            CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
        if (log_size > 0) {
            vector<char> logBuf(log_size + 1);
            clGetProgramBuildInfo(program, device_id,
                CL_PROGRAM_BUILD_LOG,
                log_size, logBuf.data(), NULL);
            logBuf[log_size] = '\0';
            cerr << logBuf.data() << endl;
        }
    }

    kernel = clCreateKernel(program, "slidingAvg", &ret);
    if (ret != CL_SUCCESS || !kernel) {
        cerr << "[ERROR] clCreateKernel(slidingAvg) failed!\n";
        goto CLEANUP;
    }

    n_int = (int)n;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
    ret |= clSetKernelArg(kernel, 2, sizeof(int), &n_int);
    ret |= clSetKernelArg(kernel, 3, sizeof(int), &windowSize);
    if (ret != CL_SUCCESS) {
        cerr << "[ERROR] clSetKernelArg failed!\n";
        goto CLEANUP;
    }

    global_work_size = n;
    local_work_size = workGroupSize;

    t1 = std::chrono::high_resolution_clock::now();

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_work_size, &local_work_size,
        0, NULL, NULL);
    clFinish(command_queue);

    ret = clEnqueueReadBuffer(command_queue, out_mem, CL_TRUE,
        0, n * sizeof(float),
        outArr, 0, NULL, NULL);

    t2 = std::chrono::high_resolution_clock::now();
    if (calcTime) {
        *calcTime = std::chrono::duration<double>(t2 - t1).count();
    }

CLEANUP:
    if (fp)         fclose(fp);
    if (source_str) free(source_str);

    if (kernel)     clReleaseKernel(kernel);
    if (program)    clReleaseProgram(program);

    if (in_mem)     clReleaseMemObject(in_mem);
    if (out_mem)    clReleaseMemObject(out_mem);

    if (command_queue) clReleaseCommandQueue(command_queue);
    if (context)    clReleaseContext(context);
}


