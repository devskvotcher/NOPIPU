__kernel void norm2D(
    __global const float* a,
    __global const float* b,
    __global float* out,
    int n
)
{
    int gid = get_global_id(0);
    if (gid < n)
    {
        float aa = a[gid];
        float bb = b[gid];
        out[gid] = sqrt(aa * aa + bb * bb);
    }
}

__kernel void polynomial(
    __global const float* x,
    __global float* out,
    int n
)
{
    int gid = get_global_id(0);
    if (gid < n)
    {
        float val = 0.0f;
        for (int i = 1; i <= 9; i++)
        {
            float tmp = pow(x[gid], (float)i) / i;
            val += tmp;
        }
        out[gid] = val;
    }
}

__kernel void subsetsMeanOptim(
    __global const float* X_mem,
    __global const int* starts_mem,
    __global float* out_mem,
    int Nsubs,
    int subsetSize
)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int groupSize = get_local_size(0);

    if (gid >= Nsubs) return;

    int startIndex = starts_mem[gid];

    float partialSum = 0.0f;
    for (int i = lid; i < subsetSize; i += groupSize)
    {
        float val = X_mem[startIndex + i];
        partialSum += val * val;
    }

    __local float localBuf[256];
    localBuf[lid] = partialSum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = groupSize / 2; offset > 0; offset /= 2)
    {
        if (lid < offset)
            localBuf[lid] += localBuf[lid + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        float s = localBuf[0];
        out_mem[gid] = s / subsetSize;
    }
}

__kernel void slidingAvg(
    __global const float* inArr,
    __global float* outArr,
    int n,
    int windowSize
)
{
    int gid = get_global_id(0);
    if (gid < n)
    {
        if (gid + windowSize <= n)
        {
            float sum = 0.0f;
            for (int j = 0; j < windowSize; j++)
                sum += inArr[gid + j];
            outArr[gid] = sum / windowSize;
        }
        else
        {
            outArr[gid] = 0.0f;
        }
    }
}
