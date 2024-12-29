// Kernel 5: Тайловое умножение с локальной памятью
#define TS 32
__kernel void MultKernel(__global float* C,
    __global const float* A,
    __global const float* B,
    const int N)
{
    int localCol = get_local_id(0);
    int localRow = get_local_id(1);

    int groupCol = get_group_id(0);
    int groupRow = get_group_id(1);

    int globalCol = groupCol * TS + localCol;
    int globalRow = groupRow * TS + localRow;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float sum = 0.0f;

    int numTiles = (N + TS - 1) / TS;

    for (int t = 0; t < numTiles; t++)
    {
        int tiledCol = t * TS + localCol;
        int tiledRow = t * TS + localRow;

        if (globalRow < N && tiledCol < N)
            Asub[localRow][localCol] = A[globalRow * N + tiledCol];
        else
            Asub[localRow][localCol] = 0.0f;

        if (tiledRow < N && globalCol < N)
            Bsub[localRow][localCol] = B[tiledRow * N + globalCol];
        else
            Bsub[localRow][localCol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++)
        {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalRow < N && globalCol < N)
    {
        C[globalRow * N + globalCol] = sum;
    }
}
