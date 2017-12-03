__kernel void myFilter(
__global uchar* data, __global uchar* data2){
    float kernelMatrix[] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1};
    uint kernelWidth = 3;
    uint kernelHeight = 3,width = 640,height = 480;
    uint N = get_global_id(0);
    uint x = N % width;
    uint y = N / width;

    float rSum = 0, kSum = 0;
    for (uint i = 0; i < kernelWidth; i++)
    {
        for (uint j = 0; j < kernelHeight; j++)
        {
            int pixelPosX = x + (i - (kernelWidth / 2));
            int pixelPosY = y + (j - (kernelHeight / 2));
            
            if (pixelPosX >= width)
                pixelPosX = (width<<1)-1-pixelPosX;
            if (pixelPosY >= height)
                pixelPosY = (height<<1)-1-pixelPosY;
            if (pixelPosX < 0) 
                pixelPosX = -pixelPosX;
            if (pixelPosY < 0) 
                pixelPosY = -pixelPosY;
            
            uchar r = data[pixelPosX + pixelPosY * width];
            float kernelVal = kernelMatrix[i + j * kernelWidth];
            rSum += r * kernelVal;
            
            kSum += kernelVal;
        }
    }
    data2[x+y*width] = (uchar)rSum;

    barrier(CLK_LOCAL_MEM_FENCE);
}
