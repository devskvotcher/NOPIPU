// Kernel 1: “о же, но с временной переменной (смысл почти тот же)
__kernel void MultKernel(__global float* C,
    __global float* A,
    __global float* B,
    const int N)
{
    int gid = get_global_id(0);
    int row = gid / N;
    int col = gid % N;
    if (row >= N || col >= N) return;

    float tmp = 0.0f;
    for (int k = 0; k < N; k++)
    {
        tmp += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = tmp;
}
