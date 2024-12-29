// Kernel 3: Копируем строку A в приватный буфер Awrk, далее умножаем
__kernel void MultKernel(__global float* C,
    __global float* A,
    __global float* B,
    const int N)
{
    int row = get_global_id(0);
    if (row >= N) return;

    float Awrk[1536];
    for (int k = 0; k < N; k++)
    {
        Awrk[k] = A[row * N + k];
    }

    for (int col = 0; col < N; col++)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        {
            sum += Awrk[k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
