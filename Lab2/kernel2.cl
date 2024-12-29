// Kernel 2: Один поток = одна строка (row). Идём по всем col внутри ядра.
__kernel void MultKernel(__global float* C,
    __global float* A,
    __global float* B,
    const int N)
{
    int row = get_global_id(0);
    if (row >= N) return;

    for (int col = 0; col < N; col++)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
