// Kernel 0: Простейший скалярный вариант (один поток = один элемент)
__kernel void MultKernel(__global float* C,
    __global float* A,
    __global float* B,
    const int N)
{
    int gid = get_global_id(0);
    int row = gid / N;
    int col = gid % N;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++)
    {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
