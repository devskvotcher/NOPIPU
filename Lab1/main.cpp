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
    size_t n, size_t workGroupSize);

void calcPolySerial(const float* x, float* out, size_t n);
void calcPolyOMP(const float* x, float* out, size_t n);
void calcPolyOpenCL(const float* x, float* out, size_t n, size_t workGroupSize);

void subsetsMeanSerial(const float* X, float* out, size_t Nsubs,
    const vector<int>& starts, int subsetSize);
void subsetsMeanOMP(const float* X, float* out, size_t Nsubs,
    const vector<int>& starts, int subsetSize);
void subsetsMeanOpenCL(const float* X, const int* starts,
    float* out, size_t Nsubs,
    int subsetSize, size_t workGroupSize);

void movingAvgSerial(const float* inArr, float* outArr, size_t n, int windowSize);
void movingAvgOMP(const float* inArr, float* outArr, size_t n, int windowSize);
void movingAvgOpenCL(const float* inArr, float* outArr, size_t n,
    int windowSize, size_t workGroupSize);

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
    cerr << "[DEBUG] taskNumber = " << taskNumber << endl;  // DEBUG

    cout << "Выберите устройство для OpenCL:\n"
        << " 1 - CPU\n"
        << " 2 - GPU\n";
    int mode;
    cin >> mode;
    device_mode = (DEVICE_MODE)mode;
    cerr << "[DEBUG] device_mode = " << device_mode << endl;  // DEBUG

    // Доп. параметры для заданий 3 и 4
    int subsetSize = 0;
    size_t nSubsets = 0;
    if (taskNumber == 3)
    {
        cout << "Укажите размер каждого подмножества: ";
        cin >> subsetSize;
        cout << "Укажите количество подмножеств: ";
        cin >> nSubsets;
        cerr << "[DEBUG] subsetSize = " << subsetSize
            << ", nSubsets = " << nSubsets << endl;
    }

    int windowSize = 0;
    if (taskNumber == 4)
    {
        cout << "Укажите размер окна скользящего среднего: ";
        cin >> windowSize;
        cerr << "[DEBUG] windowSize = " << windowSize << endl;
    }

    // Считываем, сколько раз будем тестировать (сколько разных n)
    cout << "Сколько различных размеров (arraySize) будем тестировать? ";
    int countSizes;
    cin >> countSizes;
    cerr << "[DEBUG] countSizes = " << countSizes << endl;
    vector<size_t> sizes(countSizes);
    for (int i = 0; i < countSizes; i++)
    {
        cout << "Введите размер " << (i + 1) << ": ";
        cin >> sizes[i];
        cerr << "[DEBUG] sizes[" << i << "] = " << sizes[i] << endl;
    }

    // Считываем, сколько вариантов WGS
    cout << "Сколько вариантов Work Group Size? ";
    int countWG;
    cin >> countWG;
    cerr << "[DEBUG] countWG = " << countWG << endl;
    vector<size_t> groupSizes(countWG);
    for (int i = 0; i < countWG; i++)
    {
        cout << "Введите WG Size " << (i + 1) << ": ";
        cin >> groupSizes[i];
        cerr << "[DEBUG] groupSizes[" << i << "] = " << groupSizes[i] << endl;
    }

    cout << "\nРЕЗУЛЬТАТЫ ДЛЯ ЗАДАНИЯ №" << taskNumber << ":\n";
    cout << "-------------------------------------------------------\n";
    cout << "| Size\t| WG\t| Serial\t| OMP\t| OpenCL\t|\n";
    cout << "-------------------------------------------------------\n";

    // Перебираем все размеры, которые заданы
    for (auto n : sizes)
    {
        cerr << "[DEBUG] ===> START experiment: n = " << n << endl;

        // Готовим массивы
        vector<float> A(n), B(n), OutSerial(n), OutOmp(n), OutOcl(n);
        vector<float> OutSerialSubs(nSubsets), OutOmpSubs(nSubsets), OutOclSubs(nSubsets);

        cerr << "[DEBUG] A.size() = " << A.size()
            << ", B.size() = " << B.size() << endl;
        cerr << "[DEBUG] OutSerial.size() = " << OutSerial.size() << endl;
        cerr << "[DEBUG] OutSerialSubs.size() = " << OutSerialSubs.size()
            << ", nSubsets = " << nSubsets << endl;

        // Инициализируем A и B
        for (size_t i = 0; i < n; i++)
        {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
        }

        // Если задание 3: формируем вектор starts
        vector<int> starts(nSubsets);
        if (taskNumber == 3)
        {
            for (size_t k = 0; k < nSubsets; k++)
            {
                starts[k] = (int)(k * subsetSize);
            }
            // Проверка: если k*subsetSize >= n, будет выход за границы.
            // Выведем для отладки последний элемент:
            if (!starts.empty())
            {
                int last = starts[nSubsets - 1];
                cerr << "[DEBUG] starts[nSubsets-1] = " << last
                    << ", last+subsetSize = " << (last + subsetSize)
                    << " (should be <= n=" << n << "?)" << endl;
            }
        }

        // Перебираем все WG
        for (auto wg : groupSizes)
        {
            cerr << "[DEBUG]   -> WG = " << wg << endl;
            double tSerial = 0, tOmp = 0, tOpenCL = 0;

            // ---------------- SERIAL ----------------
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

            // ---------------- OMP ----------------
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

            // ---------------- OpenCL ----------------
            {
                cerr << "[DEBUG] *** OpenCL call ***" << endl;
                auto t1 = chrono::high_resolution_clock::now();
                switch (taskNumber)
                {
                case 1:
                    calcNorm2DOpenCL(A.data(), B.data(), OutOcl.data(), n, wg);
                    break;
                case 2:
                    calcPolyOpenCL(A.data(), OutOcl.data(), n, wg);
                    break;
                case 3:
                    subsetsMeanOpenCL(A.data(), starts.data(),
                        OutOclSubs.data(), nSubsets,
                        subsetSize, wg);
                    break;
                case 4:
                    movingAvgOpenCL(A.data(), OutOcl.data(), n, windowSize, wg);
                    break;
                }
                auto t2 = chrono::high_resolution_clock::now();
                tOpenCL = chrono::duration<double>(t2 - t1).count();
            }

            // Печатаем результат
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

void calcNorm2DOpenCL(const float* a, const float* b, float* out,
    size_t n, size_t workGroupSize)
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

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    std::cerr << "[DEBUG] clGetPlatformIDs => ret=" << ret
        << ", ret_num_platforms=" << ret_num_platforms << std::endl;
    if (ret != CL_SUCCESS || ret_num_platforms == 0) {
        std::cerr << "[ERROR] Не удалось получить OpenCL-платформу!" << std::endl;
        return;
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
    std::cerr << "[DEBUG] clGetDeviceIDs => ret=" << ret
        << ", ret_num_devices=" << ret_num_devices << std::endl;
    if (ret != CL_SUCCESS || ret_num_devices == 0) {
        std::cerr << "[ERROR] Не найдено подходящее устройство OpenCL!" << std::endl;
        return;
    }

    context = clCreateContext(NULL, 1, &device_id,
        NULL, NULL, &errcode_ret);
    std::cerr << "[DEBUG] clCreateContext => " << errcode_ret << std::endl;
    if (errcode_ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clCreateContext failed: err=" << errcode_ret << std::endl;
        return;
    }

    {
        cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &errcode_ret);
        std::cerr << "[DEBUG] clCreateCommandQueueWithProperties => " << errcode_ret << std::endl;
        if (errcode_ret != CL_SUCCESS) {
            std::cerr << "[ERROR] clCreateCommandQueueWithProperties failed: err=" << errcode_ret << std::endl;
            clReleaseContext(context);
            return;
        }
    }

    a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clCreateBuffer a_mem failed: err=" << errcode_ret << std::endl;
        goto clean_exit;
    }
    b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clCreateBuffer b_mem failed: err=" << errcode_ret << std::endl;
        goto clean_exit;
    }
    o_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clCreateBuffer o_mem failed: err=" << errcode_ret << std::endl;
        goto clean_exit;
    }

    ret = clEnqueueWriteBuffer(command_queue, a_mem, CL_TRUE, 0, n * sizeof(float), a, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clEnqueueWriteBuffer(a_mem) failed: ret=" << ret << std::endl;
        goto clean_exit;
    }
    ret = clEnqueueWriteBuffer(command_queue, b_mem, CL_TRUE, 0, n * sizeof(float), b, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clEnqueueWriteBuffer(b_mem) failed: ret=" << ret << std::endl;
        goto clean_exit;
    }

    {
        fp = fopen("Kernel.cl", "r");
        if (!fp) {
            std::cerr << "[ERROR] Не удалось открыть Kernel.cl" << std::endl;
            goto clean_exit;
        }
        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);
        fp = nullptr;

        program = clCreateProgramWithSource(context, 1,
            (const char**)&source_str, &source_size, &errcode_ret);
        free(source_str);
        source_str = nullptr;
        if (errcode_ret != CL_SUCCESS) {
            std::cerr << "[ERROR] clCreateProgramWithSource failed: err=" << errcode_ret << std::endl;
            goto clean_exit;
        }
    }

    errcode_ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clBuildProgram failed: err=" << errcode_ret << std::endl;
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size) {
            char* logbuf = (char*)malloc(log_size + 1);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, logbuf, NULL);
            logbuf[log_size] = '\0';
            std::cerr << "[BUILD LOG]\n" << logbuf << std::endl;
            free(logbuf);
        }
        goto clean_exit;
    }

    kernel = clCreateKernel(program, "norm2D", &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clCreateKernel(norm2D) failed: err=" << errcode_ret << std::endl;
        goto clean_exit;
    }

    n_int = (int)n;
    errcode_ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    errcode_ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    errcode_ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &o_mem);
    errcode_ret |= clSetKernelArg(kernel, 3, sizeof(int), &n_int);
    if (errcode_ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clSetKernelArg failed: err=" << errcode_ret << std::endl;
        goto clean_exit;
    }

    global_work_size = n;

    errcode_ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_work_size,
        /* &workGroupSize */ NULL,
        0, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clEnqueueNDRangeKernel failed: err=" << errcode_ret << std::endl;
        goto clean_exit;
    }

    errcode_ret = clEnqueueReadBuffer(command_queue, o_mem, CL_TRUE, 0,
        n * sizeof(float), out, 0, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        std::cerr << "[ERROR] clEnqueueReadBuffer failed: err=" << errcode_ret << std::endl;
        goto clean_exit;
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

void calcPolyOpenCL(const float* x, float* out, size_t n, size_t workGroupSize)
{
    cl_int ret;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
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

    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);

    cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);

    cl_mem x_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &ret);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, x_mem, CL_TRUE, 0,
        n * sizeof(float), x, 0, NULL, NULL);

    FILE* fp = fopen("Kernel.cl", "r");
    if (!fp)
    {
        cerr << "Не удалось открыть Kernel.cl\n";
        exit(1);
    }
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, &source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (ret == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        cerr << log << endl;
        free(log);
    }

    cl_kernel kernel = clCreateKernel(program, "polynomial", &ret);

    int n_int = (int)n;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(int), &n_int);

    size_t global_work_size = n;
    size_t local_work_size = workGroupSize;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_work_size, &local_work_size,
        0, NULL, NULL);

    ret = clEnqueueReadBuffer(command_queue, out_mem, CL_TRUE, 0,
        n * sizeof(float), out, 0, NULL, NULL);

    free(source_str);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(x_mem);
    clReleaseMemObject(out_mem);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
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

void subsetsMeanOpenCL(const float* X, const int* starts,
    float* out, size_t Nsubs,
    int subsetSize, size_t workGroupSize)
{
    cl_int ret;
    cl_platform_id platform_id;
    cl_uint ret_num_platforms;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    cl_device_id device_id;
    cl_uint ret_num_devices;
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

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);


    int maxIndex = starts[Nsubs - 1];
    int xLength = maxIndex + subsetSize;

    cl_mem X_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, xLength * sizeof(float), NULL, &ret);
    cl_mem starts_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, Nsubs * sizeof(int), NULL, &ret);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, Nsubs * sizeof(float), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, X_mem, CL_TRUE, 0, xLength * sizeof(float), X, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, starts_mem, CL_TRUE, 0, Nsubs * sizeof(int), starts, 0, NULL, NULL);

    FILE* fp = fopen("Kernel.cl", "r");
    if (!fp)
    {
        cerr << "Не удалось открыть Kernel.cl\n";
        exit(1);
    }
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, &source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (ret == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        cerr << log << endl;
        free(log);
    }


    cl_kernel kernel = clCreateKernel(program, "subsetsMeanOptim", &ret);


    int Nsubs_int = (int)Nsubs;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &X_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &starts_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &Nsubs_int);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &subsetSize);

    size_t global_work_size = Nsubs;
    size_t local_work_size = workGroupSize;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_work_size, &local_work_size,
        0, NULL, NULL);


    ret = clEnqueueReadBuffer(command_queue, out_mem, CL_TRUE, 0,
        Nsubs * sizeof(float), out, 0, NULL, NULL);

    free(source_str);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(X_mem);
    clReleaseMemObject(starts_mem);
    clReleaseMemObject(out_mem);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
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

void movingAvgOpenCL(const float* inArr, float* outArr,
    size_t n, int windowSize, size_t workGroupSize)
{
    cl_int ret;
    cl_platform_id platform_id;
    cl_uint ret_num_platforms;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    cl_device_id device_id;
    cl_uint ret_num_devices;
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

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, props, &ret);

    cl_mem in_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &ret);
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, in_mem, CL_TRUE, 0,
        n * sizeof(float), inArr, 0, NULL, NULL);

    FILE* fp = fopen("Kernel.cl", "r");
    if (!fp)
    {
        cerr << "Не удалось открыть Kernel.cl\n";
        exit(1);
    }
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, &source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (ret == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        cerr << log << endl;
        free(log);
    }

    cl_kernel kernel = clCreateKernel(program, "slidingAvg", &ret);

    int n_int = (int)n;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(int), &n_int);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &windowSize);

    size_t global_work_size = n;
    size_t local_work_size = workGroupSize;

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_work_size, &local_work_size,
        0, NULL, NULL);

    ret = clEnqueueReadBuffer(command_queue, out_mem, CL_TRUE, 0,
        n * sizeof(float), outArr, 0, NULL, NULL);

    free(source_str);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(in_mem);
    clReleaseMemObject(out_mem);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}
