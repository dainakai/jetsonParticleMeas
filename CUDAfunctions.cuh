/**
 * @file CUDAfunctions.cuh
 * @brief Jetson上の位相回復ホログラフィによる流刑分布取得実験用カーネル群
 * @author Dai Nakai
 * @date May, 2022.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include "hostfunctions.h"

/** @def
 * CUDA組み込み関数のチェックマクロ。cudaMalloc や cudaMemcpy に。
 */
#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if(error != cudaSuccess){                                                   \
        std::cout << "Error: " << __FILE__ << ":" << __LINE__ << ", ";          \
        std::cout << "code:" << error << ", reason: " << cudaGetErrorString(error) << std::endl;\
        exit(1);                                                                \
    }                                                                           \
}

__global__ void CuGetCrossCor(float *corArray, float *img1, float *img2, int gridNum, int srchSize, int intrSize, int gridSize, int imgLen){
    int x = blockIdx.x*blockDim.x +threadIdx.x;
    int y = blockIdx.y*blockDim.y +threadIdx.y;

    if (x < (srchSize-intrSize+1)*(gridNum-1) && y < (srchSize-intrSize+1)*(gridNum-1)){
        int gridIdxx = x/(srchSize-intrSize+1);
        int gridIdxy = y/(srchSize-intrSize+1);
        int idxx = x - gridIdxx*(srchSize-intrSize+1);
        int idxy = y - gridIdxy*(srchSize-intrSize+1);

        int a1 = gridIdxy*gridSize + (int)(intrSize/2);
        int a2 = gridIdxx*gridSize + (int)(intrSize/2);
        int b1 = gridIdxy*gridSize + idxy;
        int b2 = gridIdxx*gridSize + idxx;

        float meanA = 0.0;
        float meanB = 0.0;
        float num = 0.0;
        float denomA = 0.0;
        float denomB = 0.0;

        for (int i = 0; i < intrSize; i++){
            for (int j = 0; j < intrSize; j++){
                meanA += img1[(a1+i)*imgLen + a2+j];
                meanB += img2[(b1+i)*imgLen + b2+j];
            }
        }

        meanA /= (float)(intrSize*intrSize);
        meanB /= (float)(intrSize*intrSize);

        for (int i = 0; i < intrSize; i++){
            for (int j = 0; j < intrSize; j++){
                num += (img1[(a1+i)*imgLen + a2+j]-meanA)*(img2[(b1+i)*imgLen + b2+j]-meanB);
                denomA += (img1[(a1+i)*imgLen + a2+j]-meanA)*(img1[(a1+i)*imgLen + a2+j]-meanA);
                denomB += (img2[(b1+i)*imgLen + b2+j]-meanB)*(img2[(b1+i)*imgLen + b2+j]-meanB);
            }
        }

        corArray[y*(srchSize-intrSize+1)*(gridNum-1) + x] = num/(sqrt(denomA)*sqrt(denomB));
    }
}

__global__ void CuGetVector(float *vecArrayX, float *vecArrayY, float *corArray, int gridNum, int corArrSize, int intrSize){
    int gridIdxx = blockIdx.x*blockDim.x +threadIdx.x;
    int gridIdxy = blockIdx.y*blockDim.y +threadIdx.y;

    if (gridIdxx < gridNum-1 && gridIdxy < gridNum -1){
        // printf("%d,%d\n",gridIdxx,gridIdxy);
        int x0 = 0;
        int y0 = 0;
        float tmp = 0.0;

        for (int i = 0; i < corArrSize; i++){
            for (int j = 0; j < corArrSize; j++){
                if (corArray[corArrSize*(gridNum-1)*(corArrSize*gridIdxy+i)+corArrSize*gridIdxx+j]>tmp){
                    x0 = corArrSize*gridIdxx + j;
                    y0 = corArrSize*gridIdxy + i;
                    tmp = corArray[corArrSize*(gridNum-1)*(corArrSize*gridIdxy+i)+corArrSize*gridIdxx+j];
                }
            }
        }

        if (x0==0 || y0==0 || x0==corArrSize*(gridNum-1)-1 || y0==corArrSize*(gridNum-1)-1){
            vecArrayX[(gridNum-1)*gridIdxy+gridIdxx] = (float)x0 - (float)(intrSize)/2.0 - gridIdxx*corArrSize;
            vecArrayY[(gridNum-1)*gridIdxy+gridIdxx] = (float)y0 - (float)(intrSize)/2.0 - gridIdxy*corArrSize;
        }else{
            float valy1x0 = corArray[corArrSize*(gridNum-1)*(y0+1) + x0];
            float valy0x0 = corArray[corArrSize*(gridNum-1)*y0 + x0];
            float valyInv1x0 = corArray[corArrSize*(gridNum-1)*(y0-1) + x0];
            float valy0x1 = corArray[corArrSize*(gridNum-1)*y0 + x0+1];
            float valy0xInv1 = corArray[corArrSize*(gridNum-1)*y0 + x0-1];

            if (valy1x0-2.0*valy0x0+valyInv1x0==0.0 || valy0x1-2.0*valy0x0+valy0xInv1==0.0){
                valy0x0 += 0.00001;
            }

            vecArrayX[(gridNum-1)*gridIdxy+gridIdxx] = (float)x0 - (valy0x1-valy0xInv1)/(valy0x1-2.0*valy0x0+valy0xInv1)/2.0 - (float)(intrSize)/2.0 - gridIdxx*corArrSize;
            vecArrayY[(gridNum-1)*gridIdxy+gridIdxx] = (float)y0 - (valy1x0-valyInv1x0)/(valy1x0-2.0*valy0x0+valyInv1x0)/2.0 - (float)(intrSize)/2.0 - gridIdxy*corArrSize;
        }
    }
}

__global__ void CuErrorCorrect(float *corArrayIn, float *corArrayOut, int corArrSize, int gridNum){
    int x = blockIdx.x*blockDim.x +threadIdx.x;
    int y = blockIdx.y*blockDim.y +threadIdx.y;

    if (x<corArrSize*(gridNum-1) && y<corArrSize*(gridNum-1)){
        int gridIdxx = x/corArrSize;
        int gridIdxy = y/corArrSize;

        float tmp[3][3];
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                tmp[i][j] = 1.0;
            }
        }
        
        if (gridIdxx==0){
            tmp[0][0]=0.0;
            tmp[1][0]=0.0;
            tmp[2][0]=0.0;
        }else if (gridIdxx==(gridNum-2)){
            tmp[0][2]=0.0;
            tmp[1][2]=0.0;
            tmp[2][2]=0.0;
        }else if (gridIdxy==0){
            tmp[0][0]=0.0;
            tmp[0][1]=0.0;
            tmp[0][2]=0.0;
        }else if (gridIdxy==(gridNum-2)){
            tmp[2][0]=0.0;
            tmp[2][1]=0.0;
            tmp[2][2]=0.0;
        }

        corArrayOut[y*corArrSize*(gridNum-1)+x] = 0.0;

        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                if (tmp[i][j]==1.0){
                    corArrayOut[y*corArrSize*(gridNum-1)+x]+=corArrayIn[y*corArrSize*(gridNum-1+(i-1))+x+(j-1)*corArrSize];
                }
            }
        }
    }
}

/**
 * @fn
 * @brief ホストメモリ上のfloat画像配列img1,img2に対してPIVを行う。エラーコレクトはパラボラ。Jetson(Maxwell)では blockSize = 32 はダメそう？
 * @param vecArray 出力ベクトル配列のポインタ。((gridNum-1),(gridNum-1),2)型で(:,:,0)はx成分、(:,:,1)はy成分。gridNum = floor(imgLen/gridSize).
 * @param intrSize 参照窓の大きさ
 * @param srchSize 探査窓の大きさ
 * @param gridSize 参照窓の充填グリッド
 * @param blockSize CUDAブロックサイズ
 * @return なし
 */
void getPIVMapOnGPU(float *vecArrayX, float *vecArrayY, float *img1, float *img2, int imgLen, int gridSize, int intrSize, int srchSize, int blockSize){
    const int gridNum = (int)(imgLen/gridSize);
    
    float *dev_corArray, *dev_corArray2, *dev_vecArrayX, *dev_vecArrayY;
    CHECK(cudaMalloc((void **)&dev_corArray, sizeof(float)*(srchSize-intrSize+1)*(gridNum-1)*(srchSize-intrSize+1)*(gridNum-1)));
    CHECK(cudaMalloc((void **)&dev_corArray2, sizeof(float)*(srchSize-intrSize+1)*(gridNum-1)*(srchSize-intrSize+1)*(gridNum-1)));
    CHECK(cudaMalloc((void **)&dev_vecArrayX, sizeof(float)*(gridNum-1)*(gridNum-1)));
    CHECK(cudaMalloc((void **)&dev_vecArrayY, sizeof(float)*(gridNum-1)*(gridNum-1)));

    dim3 grid((int)ceil((float)(srchSize-intrSize+1)*(gridNum-1)/(float)blockSize), (int)ceil((float)(srchSize-intrSize+1)*(gridNum-1)/(float)blockSize)), block(blockSize,blockSize);
    dim3 grid2((int)ceil((float)(gridNum-1)/(float)blockSize), (int)ceil((float)(gridNum-1)/(float)blockSize));
    // printf("%d\n",(int)ceil((float)(gridNum-1)/(float)blockSize));

    float *dev_img1, *dev_img2;
    CHECK(cudaMalloc((void **)&dev_img1, sizeof(float)*imgLen*imgLen));
    CHECK(cudaMalloc((void **)&dev_img2, sizeof(float)*imgLen*imgLen));

    CHECK(cudaMemcpy(dev_img1, img1, sizeof(float)*imgLen*imgLen, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_img2, img2, sizeof(float)*imgLen*imgLen, cudaMemcpyHostToDevice));

    CuGetCrossCor<<<grid,block>>>(dev_corArray,dev_img1,dev_img2,gridNum,srchSize,intrSize,gridSize,imgLen);
    CuErrorCorrect<<<grid,block>>>(dev_corArray,dev_corArray2,srchSize-intrSize+1,gridNum);
    CuGetVector<<<grid2,block>>>(dev_vecArrayX,dev_vecArrayY,dev_corArray,gridNum,(srchSize-intrSize+1),intrSize);
    
    CHECK(cudaMemcpy(vecArrayX,dev_vecArrayX,sizeof(float)*(gridNum-1)*(gridNum-1),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(vecArrayY,dev_vecArrayY,sizeof(float)*(gridNum-1)*(gridNum-1),cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dev_corArray));
    CHECK(cudaFree(dev_vecArrayX));
    CHECK(cudaFree(dev_vecArrayY));
    CHECK(cudaFree(dev_img1));
    CHECK(cudaFree(dev_img2));
}

__global__ void CuTransFunc(cufftComplex *output, float *sqr, float trans_z, float waveLen, int datLen, float dx){
    // dim3 grid((width+31)/32, (height+31)/32), block(32,32)
    // CHECH deviceQuery and make sure threads per block are 1024!!!!

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    float tmp;
    float tmpx, tmpy;
    float uband = 1.0/waveLen/sqrt(2*trans_z/(float)datLen/dx + 1);

    if( (x < datLen) && (y < datLen) ){
        tmp = 2.0*3.14159265358979*trans_z/waveLen*sqrt(sqr[x + datLen*y]);
        output[x + datLen*y].x = cos(tmp);
        output[x + datLen*y].y = sin(tmp);
        tmpx = abs(((float)x - (float)datLen/2.0)*waveLen/(float)datLen/dx);
        tmpy = abs(((float)y - (float)datLen/2.0)*waveLen/(float)datLen/dx);
        if (tmpx > uband || tmpy > uband){
            output[x + datLen*y].x = 0.0;
            output[x + datLen*y].y = 0.0;
        }
    }
}

__global__ void CuTransSqr(float *d_sqr, int datLen, float waveLen, float dx){
    // dim3 grid((width+31)/32, (height+31)/32), block(32,32)
    // CHECH deviceQuery and make sure threads per block are 1024!!!!

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    float x_f = (float)x;
    float y_f = (float)y;
    float w_f = (float)datLen;

    if( (x < datLen) && (y < datLen) ){
        d_sqr[x + datLen*y] = 1.0 - ((x_f - w_f/2.0)*waveLen/w_f/dx)*((x_f - w_f/2.0)*waveLen/w_f/dx) - ((y_f - w_f/2.0)*waveLen/w_f/dx)*((y_f - w_f/2.0)*waveLen/w_f/dx);
    }
}

__global__ void CuCharToNormFloatArr(float *out, unsigned char *in, int datLen, float Norm){
    // dim3 grid((width+31)/32, (height+31)/32), block(32,32)
    // CHECH deviceQuery and make sure threads per block are 1024!!!!

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        out[y*datLen + x] = (float)((int)in[y*datLen + x])/Norm;
        // printf("%lf ",out[y*datLen+x]);
    }
}

__global__ void CuNormFloatArrToChar(unsigned char *out, float *in, int datLen, float Norm){
    // dim3 grid((width+31)/32, (height+31)/32), block(32,32)
    // CHECH deviceQuery and make sure threads per block are 1024!!!!

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        if (in[y*datLen+x] < 0.0){
            out[y*datLen + x] = (unsigned char)0;
        }else if ((in[y*datLen+x] > 255.0)){
            out[y*datLen + x] = (unsigned char)255;
        }else{
            out[y*datLen + x] = (unsigned char)(in[y*datLen + x]*Norm);
        }
        // printf("%lf ",out[y*datLen+x]);
    }
}

__global__ void CuSetArrayCenterHalf(cufftComplex *out, float *img, int imgLen){
    // dim3 grid((width+31)/32, (height+31)/32), block(32,32)
    // CHECH deviceQuery and make sure threads per block are 1024!!!!

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < imgLen) && (y < imgLen) ){
        out[(y+imgLen/2)*imgLen*2 + (x+imgLen/2)].x = img[y*imgLen+x]; 
        out[(y+imgLen/2)*imgLen*2 + (x+imgLen/2)].y = 0.0; 
        // printf("%lf ",out[(y+imgLen/2)*imgLen*2 + (x+imgLen/2)].x);
    }
}

__global__ void CuFFTshift(cufftComplex *data, int datLen){
    // dim3 grid((width+31)/32, (height+31)/32), block(32,32)
    // CHECH deviceQuery and make sure threads per block are 1024!!!!

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    cufftComplex temp1,temp2;
    
    if((x < datLen/2) && (y < datLen/2)){
        temp1 = data[x + datLen*y];
        data[x + datLen*y] = data[x + datLen/2 + datLen*(y + datLen/2)];
        data[x + datLen/2 + datLen*(y + datLen/2)] = temp1;
    }
    if((x < datLen/2) && (y >= datLen/2)){
        temp2 = data[x + datLen*y];
        data[x + datLen*y] = data[x + datLen/2 + datLen*(y - datLen/2)];
        data[x + datLen/2 + datLen*(y - datLen/2)] = temp2;
    }
}

__global__ void CuComplexMul(cufftComplex *out, cufftComplex *inA, cufftComplex *inB, int datLen){
    // dim3 grid((width+31)/32, (height+31)/32), block(32,32)
    // CHECH deviceQuery and make sure threads per block are 1024!!!!

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    cufftComplex tmp1, tmp2;

    if( (x < datLen) && (y < datLen) ){
        tmp1 = inA[y*datLen + x];
        tmp2 = inB[y*datLen + x];
        out[y*datLen + x].x = tmp1.x * tmp2.x - tmp1.y * tmp2.y;
        out[y*datLen + x].y = tmp1.x * tmp2.y + tmp1.y * tmp2.x;
    }
}

__global__ void CuGetAbsFromComp(float *out, cufftComplex *in, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        cufftComplex tmp = in[y*datLen + x];
        out[y*datLen + x] = sqrt(tmp.x * tmp.x + tmp.y * tmp.y); // Need Sqrt() ?
    }
}

__global__ void CuGetAbs2FromComp(float *out, cufftComplex *in, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        cufftComplex tmp = in[y*datLen + x];
        out[y*datLen + x] = tmp.x * tmp.x + tmp.y * tmp.y; // Need Sqrt() ?
    }
}

__global__ void CuGetAbs2uintFromComp(unsigned char *out, cufftComplex *in, int imgLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < imgLen) && (y < imgLen) ){
        cufftComplex tmp = in[(y+imgLen)*imgLen*2 + x+imgLen];
        out[y*imgLen + x] = (unsigned char)(tmp.x * tmp.x + tmp.y * tmp.y); // Need Sqrt() ?
    }
}

__global__ void CuUpdateImposed(float *imp, float *tmp, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        if (tmp[y*datLen+x] < imp[y*datLen+x]){
            imp[y*datLen+x] = tmp[y*datLen+x];
        }
    }
}

__global__ void CuGetCenterHalf(float *out, float *in, int imgLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < imgLen) && (y < imgLen) ){
        out[y*imgLen + x] = in[(y+imgLen/2)*imgLen*2 + x+imgLen/2];
    }
}

__global__ void CuFillArrayComp(cufftComplex* array, float value, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        array[y*datLen + x].x = value;
        array[y*datLen + x].y = 0.0;
        // printf("%lf ",array[y*datLen+x].x);
    }
}

__global__ void CuFillArrayFloat(float* array, float value, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        array[y*datLen + x] = value;
    }
}

__global__ void CuInvFFTDiv(cufftComplex* array, float div, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        array[y*datLen + x].x /= div;
        array[y*datLen + x].y /= div;
    }
}

float getImgMode(unsigned char *in, int imgLen){
    int vote[256];
    for (int i = 0; i < 256; i++){
        vote[i] = 0;
    }
    
    for (int i = 0; i < imgLen*imgLen; i++){
        vote[(int)in[i]] += 1;
    }
    
    int idx = 0;
    int max = 0;

    for (int i = 0; i < 256; i++){
        if(vote[i]>max){
            max = vote[i];
            idx = i;
        }
    }
    
    return (float)idx;
}

float getImgMean(float *in, int imgLen){
    float mean=0.0;
    for (int idx = 0; idx < imgLen*imgLen; idx++){
        mean += (float)((int)in[idx]);
    }
    mean /= (float)(imgLen*imgLen);
    return mean;
}

float getImgSTD(unsigned char *in, int imgLen){
    unsigned char *tmp;
    tmp = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    float mean=0.0;
    for (int idx = 0; idx < imgLen*imgLen; idx++){
        mean += (float)((int)in[idx]);
    }
    mean /= (float)(imgLen*imgLen);

    float std=0.0;
    for (int idx = 0; idx < imgLen*imgLen; idx++){
        std += ((float)((int)in[idx])-mean)*((float)((int)in[idx])-mean);
    }
    std /= (float)(imgLen*imgLen);
    std = sqrt(std);

    return std;
}

void getGaborImposed(float *floatout, unsigned char *charout, unsigned char *in, cufftComplex *transF, cufftComplex *transInt, int imgLen, int loopCount, int blockSize=16){
    int datLen = imgLen*2;
    dim3 gridImgLen((int)ceil((float)imgLen/(float)blockSize), (int)ceil((float)imgLen/(float)blockSize)), block(blockSize,blockSize);
    dim3 gridDatLen((int)ceil((float)datLen/(float)blockSize), (int)ceil((float)datLen/(float)blockSize));
    
    unsigned char *dev_in;
    CHECK(cudaMalloc((void**)&dev_in,sizeof(unsigned char)*imgLen*imgLen));
    float *dev_img;
    CHECK(cudaMalloc((void**)&dev_img,sizeof(float)*imgLen*imgLen));
    CHECK(cudaMemcpy(dev_in, in, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyHostToDevice));
    CuCharToNormFloatArr<<<gridImgLen,block>>>(dev_img,dev_in,imgLen,1.0);
    // thrust::device_ptr<float> thimg(dev_img);
    // float meanImg = thrust::reduce(thimg,thimg+imgLen*imgLen, (float)0.0, thrust::plus<float>());
    // meanImg /= (float)(imgLen*imgLen);
    // std::cout << "Image mean: " << meanImg << std::endl;

    cufftComplex *dev_holo;
    CHECK(cudaMalloc((void**)&dev_holo,sizeof(cufftComplex)*datLen*datLen));
    CuFillArrayComp<<<gridDatLen,block>>>(dev_holo,getImgMode(in,imgLen),datLen);
    CuSetArrayCenterHalf<<<gridImgLen,block>>>(dev_holo,dev_img,imgLen);

    float *dev_imp;
    CHECK(cudaMalloc((void**)&dev_imp,sizeof(float)*datLen*datLen));
    CuFillArrayFloat<<<gridDatLen,block>>>(dev_imp,255.0,datLen);

    cufftHandle plan;
    cufftPlan2d(&plan, datLen, datLen, CUFFT_C2C);

    cufftExecC2C(plan, dev_holo, dev_holo, CUFFT_FORWARD);
    CuFFTshift<<<gridDatLen,block>>>(dev_holo, datLen);
    CuComplexMul<<<gridDatLen,block>>>(dev_holo, dev_holo, transF, datLen);

    cufftComplex *tmp_holo;
    float *tmp_imp;
    CHECK(cudaMalloc((void**)&tmp_holo,sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void**)&tmp_imp,sizeof(float)*datLen*datLen));
    
    for (int itr = 0; itr < loopCount; itr++){
        CuComplexMul<<<gridDatLen,block>>>(dev_holo,dev_holo,transInt,datLen);
        CHECK(cudaMemcpy(tmp_holo,dev_holo,sizeof(cufftComplex)*datLen*datLen,cudaMemcpyDeviceToDevice));
        CuFFTshift<<<gridDatLen,block>>>(tmp_holo,datLen);
        cufftExecC2C(plan, tmp_holo, tmp_holo, CUFFT_INVERSE);
        CuInvFFTDiv<<<gridDatLen,block>>>(tmp_holo,(float)(datLen*datLen),datLen);
        CuGetAbsFromComp<<<gridDatLen,block>>>(tmp_imp,tmp_holo,datLen);
        CuUpdateImposed<<<gridDatLen,block>>>(dev_imp,tmp_imp,datLen);
    }

    float *dev_outImp;
    CHECK(cudaMalloc((void**)&dev_outImp,sizeof(float)*imgLen*imgLen));
    CuGetCenterHalf<<<gridImgLen,block>>>(dev_outImp,dev_imp,imgLen);

    CHECK(cudaMemcpy(floatout, dev_outImp, sizeof(float)*imgLen*imgLen, cudaMemcpyDeviceToHost));

    unsigned char *saveImp;
    CHECK(cudaMalloc((void**)&saveImp,sizeof(unsigned char)*imgLen*imgLen));
    CuNormFloatArrToChar<<<gridImgLen,block>>>(saveImp,dev_outImp,imgLen,1.0);

    CHECK(cudaMemcpy(charout, saveImp, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyDeviceToHost));

    // std::cout << (float)floatout[0] << std::endl;
    // std::cout << (float)floatout[10] << std::endl;
    // std::cout << (float)floatout[100] << std::endl;
    // std::cout << (float)floatout[1000] << std::endl;

    cufftDestroy(plan);
    CHECK(cudaFree(dev_in));
    CHECK(cudaFree(dev_img));
    CHECK(cudaFree(dev_holo));
    CHECK(cudaFree(dev_imp));
    CHECK(cudaFree(tmp_holo));
    CHECK(cudaFree(tmp_imp));
    CHECK(cudaFree(dev_outImp));
    CHECK(cudaFree(saveImp));
}


void getNewImage(unsigned char *out, unsigned char *img, float a[12], int imgLen){
    long int *tmpImg;
    tmpImg = (long int *)malloc(sizeof(long int)*imgLen*imgLen);
    for (int i = 0; i < imgLen*imgLen; i++){
        tmpImg[i] = (long int)img[i];
    }
    long int bkg = 0;
    for (int j = 0; j < imgLen*imgLen; j++){
        bkg += (long int)tmpImg[j];
    }
    bkg = (long int)(round((float)bkg/(float)(imgLen*imgLen)));


    for (int i = 0; i < imgLen; i++){
        for (int j = 0; j < imgLen; j++){
            int tmpX = (int)(round(a[0]+a[1]*j+a[2]*i+a[3]*j*j+a[4]*i*j+a[5]*i*i));
            int tmpY = (int)(round(a[6]+a[7]*j+a[8]*i+a[9]*j*j+a[10]*i*j+a[11]*i*i));
            if (tmpX>=0 && tmpX<imgLen && tmpY>=0 && tmpY <imgLen){
                out[i*imgLen+j] = (unsigned char)tmpImg[tmpY*imgLen+tmpX];
            }else{
                out[i*imgLen+j] = (unsigned char)bkg;
            }
        }
    }
    free(tmpImg);
}

void getNewFloatImage(float *out, float *img, float a[12], int imgLen){
    float *tmpImg;
    tmpImg = (float *)malloc(sizeof(float)*imgLen*imgLen);
    for (int i = 0; i < imgLen*imgLen; i++){
        tmpImg[i] = img[i];
    }
    
    float bkg = 0;
    for (int j = 0; j < imgLen*imgLen; j++){
        bkg += (float)tmpImg[j];
    }
    bkg = (float)bkg/((float)(imgLen*imgLen));


    for (int i = 0; i < imgLen; i++){
        for (int j = 0; j < imgLen; j++){
            int tmpX = (int)(round(a[0]+a[1]*j+a[2]*i+a[3]*j*j+a[4]*i*j+a[5]*i*i));
            int tmpY = (int)(round(a[6]+a[7]*j+a[8]*i+a[9]*j*j+a[10]*i*j+a[11]*i*i));
            if (tmpX>=0 && tmpX<imgLen && tmpY>=0 && tmpY <imgLen){
                out[i*imgLen+j] = tmpImg[tmpY*imgLen+tmpX];
            }else{
                out[i*imgLen+j] = (float)bkg;
            }
        }
    }
    free(tmpImg);
}



__global__ void CuFloatSqrt(float *dst, cufftComplex *src, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        dst[y*datLen + x] = sqrt(src[y*datLen+x].x);
    }
}

__global__ void CuFloatSqrttoSqrt(float *dst, float *src, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        dst[y*datLen + x] = sqrt(src[y*datLen+x]);
    }
}

__global__ void CuFillCompArrayByFloatArray(cufftComplex *dst, float *src, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        dst[y*datLen + x].x = src[y*datLen+x];
        dst[y*datLen + x].y = 0.0;
    }
}

__global__ void CuGetCompAngle(float *out, cufftComplex *in, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        out[y*datLen + x] = atan2f(in[y*datLen + x].y, in[y*datLen + x].x);
    }
}

__global__ void CuGetComplexArray(cufftComplex *out, float *magnitude, float *angle, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        out[y*datLen + x].x = magnitude[y*datLen + x]*cosf(angle[y*datLen + x]);
        out[y*datLen + x].y = magnitude[y*datLen + x]*sinf(angle[y*datLen + x]);
    }
}

void getPRonGPU(cufftComplex *dev_holo, cufftComplex *holo1, cufftComplex *holo2, cufftComplex *trans, cufftComplex *transInv, int iterations, int datLen, int blockSize){
    dim3 gridDatLen((int)ceil((float)datLen/(float)blockSize), (int)ceil((float)datLen/(float)blockSize)), block(blockSize,blockSize);
    cufftComplex *compAmp1, *compAmp2;
    float *phi1, *phi2;
    CHECK(cudaMalloc((void**)&compAmp1,sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void**)&compAmp2,sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void**)&phi1,sizeof(float)*datLen*datLen));
    CHECK(cudaMalloc((void**)&phi2,sizeof(float)*datLen*datLen));
    CuFillArrayFloat<<<gridDatLen,block>>>(phi1,0.0,datLen);

    float *sqrtImg1, *sqrtImg2;
    CHECK(cudaMalloc((void**)&sqrtImg1,sizeof(float)*datLen*datLen));
    CHECK(cudaMalloc((void**)&sqrtImg2,sizeof(float)*datLen*datLen));
    CuFloatSqrt<<<gridDatLen,block>>>(sqrtImg1,holo1,datLen);
    CuFloatSqrt<<<gridDatLen,block>>>(sqrtImg2,holo2,datLen);
    CuFillCompArrayByFloatArray<<<gridDatLen,block>>>(compAmp1,sqrtImg1,datLen);

    cufftHandle plan;
    cufftPlan2d(&plan, datLen, datLen, CUFFT_C2C);

    for (int itr = 0; itr < iterations; itr++){
        // STEP 1
        cufftExecC2C(plan,compAmp1,compAmp2,CUFFT_FORWARD);
        CuFFTshift<<<gridDatLen,block>>>(compAmp2,datLen);
        CuComplexMul<<<gridDatLen,block>>>(compAmp2,compAmp2,trans,datLen);
        CuFFTshift<<<gridDatLen,block>>>(compAmp2,datLen);
        cufftExecC2C(plan,compAmp2,compAmp2,CUFFT_INVERSE);
        CuInvFFTDiv<<<gridDatLen,block>>>(compAmp2,(float)(datLen*datLen),datLen);
        CuGetCompAngle<<<gridDatLen,block>>>(phi2,compAmp2, datLen);

        // STEP 2
        CuGetComplexArray<<<gridDatLen,block>>>(compAmp2,sqrtImg2,phi2,datLen);

        // STEP 3
        cufftExecC2C(plan,compAmp2,compAmp1,CUFFT_FORWARD);
        CuFFTshift<<<gridDatLen,block>>>(compAmp1,datLen);
        CuComplexMul<<<gridDatLen,block>>>(compAmp1,compAmp1,transInv,datLen);
        CuFFTshift<<<gridDatLen,block>>>(compAmp1,datLen);
        cufftExecC2C(plan,compAmp1,compAmp1,CUFFT_INVERSE);
        CuInvFFTDiv<<<gridDatLen,block>>>(compAmp1,(float)(datLen*datLen),datLen);
        CuGetCompAngle<<<gridDatLen,block>>>(phi1,compAmp1,datLen);

        // STEP 4
        CuGetComplexArray<<<gridDatLen,block>>>(compAmp1,sqrtImg1,phi1,datLen);
    }

    // CuGetComplexArray<<<gridDatLen,block>>>(compAmp1,sqrtImg1,phi1,datLen);
    
    CHECK(cudaMemcpy(dev_holo,compAmp1,sizeof(cufftComplex)*datLen*datLen,cudaMemcpyDeviceToDevice));

    cudaFree(compAmp1);
    cudaFree(compAmp2);
    cudaFree(phi1);
    cudaFree(phi2);
    cudaFree(sqrtImg1);
    cudaFree(sqrtImg2);
    cufftDestroy(plan);
}

__global__ void CuBackRem(float *out, float *back, float imgMode, int imgLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < imgLen) && (y < imgLen) ){
        // out[y*imgLen + x] = out[y*imgLen+x]/back[y*imgLen+x];
        // out[y*imgLen + x] = out[y*imgLen+x]-back[y*imgLen+x]+addConst;
        // out[y*imgLen + x] = out[y*imgLen+x]-back[y*imgLen+x]+imgMode;
        if (out[y*imgLen+x]-back[y*imgLen+x]+imgMode<0.0){
            out[y*imgLen + x] = 0.0;
        }else{
            out[y*imgLen + x] = out[y*imgLen+x]-back[y*imgLen+x]+imgMode;
        }
    }
}

__global__ void CuFloatDiv(float *out, float value, int imgLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < imgLen) && (y < imgLen) ){
        out[y*imgLen + x] /= value;
    }
}

void getPRImposed(float *floatout, unsigned char *charout, cufftComplex *outholo, unsigned char *in1, unsigned char *in2, float* backImg1, float* backImg2, cufftComplex *transF, cufftComplex *transInt, cufftComplex *transPR, cufftComplex *transInvPR, float back1, float back2, int imgLen, int loopCount, int PRloops, int blockSize=16){
    // dim3 Declaration
    int datLen = imgLen*2;
    dim3 gridImgLen((int)ceil((float)imgLen/(float)blockSize), (int)ceil((float)imgLen/(float)blockSize)), block(blockSize,blockSize);
    dim3 gridDatLen((int)ceil((float)datLen/(float)blockSize), (int)ceil((float)datLen/(float)blockSize));
    
    // Char to device float and get mean
    unsigned char *dev_in1, *dev_in2;
    CHECK(cudaMalloc((void**)&dev_in1,sizeof(unsigned char)*imgLen*imgLen));
    CHECK(cudaMalloc((void**)&dev_in2,sizeof(unsigned char)*imgLen*imgLen));
    float *dev_img1, *dev_img2;
    CHECK(cudaMalloc((void**)&dev_img1,sizeof(float)*imgLen*imgLen));
    CHECK(cudaMalloc((void**)&dev_img2,sizeof(float)*imgLen*imgLen));    
    CHECK(cudaMemcpy(dev_in1, in1, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_in2, in2, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyHostToDevice));
    CuCharToNormFloatArr<<<gridImgLen,block>>>(dev_img1,dev_in1,imgLen,1.0);
    CuCharToNormFloatArr<<<gridImgLen,block>>>(dev_img2,dev_in2,imgLen,1.0);
    // thrust::device_ptr<float> thimg1(dev_img1);
    // thrust::device_ptr<float> thimg2(dev_img2);
    // float meanImg1 = thrust::reduce(thimg1,thimg1+imgLen*imgLen, (float)0.0, thrust::plus<float>());
    // float meanImg2 = thrust::reduce(thimg2,thimg2+imgLen*imgLen, (float)0.0, thrust::plus<float>());
    // meanImg1 /= (float)(imgLen*imgLen);
    // meanImg2 /= (float)(imgLen*imgLen);
    // std::cout << "Cam1 mean: " << meanImg1 << std::endl;
    // std::cout << "Cam2 mean: " << meanImg2 << std::endl;

    // Background Subtraction. Means to be 0.5

    // float imgMode1 = 200;
    float imgMode1 = back1;
    // float imgMode1 = getImgMode(in1,imgLen);
    // float imgMode2 = 200;
    float imgMode2 = back2;
    // float imgMode2 = getImgMode(in2,imgLen);

    float *dev_bkg1,*dev_bkg2;
    CHECK(cudaMalloc((void**)&dev_bkg1,sizeof(float)*imgLen*imgLen));
    CHECK(cudaMalloc((void**)&dev_bkg2,sizeof(float)*imgLen*imgLen));
    CHECK(cudaMemcpy(dev_bkg1,backImg1,sizeof(float)*imgLen*imgLen,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_bkg2,backImg2,sizeof(float)*imgLen*imgLen,cudaMemcpyHostToDevice));
    CuBackRem<<<gridImgLen,block>>>(dev_img1,dev_bkg1,imgMode1,imgLen);
    // CuBackRem<<<gridImgLen,block>>>(dev_img1,dev_bkg1,0.5,imgLen);
    CuBackRem<<<gridImgLen,block>>>(dev_img2,dev_bkg2,imgMode2,imgLen);
    // CuBackRem<<<gridImgLen,block>>>(dev_img2,dev_bkg2,0.5,imgLen);  
    // CuFloatDiv<<<gridImgLen,block>>>(dev_img1,meanImg1,imgLen);
    // CuFloatDiv<<<gridImgLen,block>>>(dev_img2,meanImg2,imgLen);

    cudaFree(dev_bkg1);
    cudaFree(dev_bkg2);

    // !!!!!!!!!!!!! Cam1 img to dev_holo2, Cam2 img to dev_holo1 !!!!!!!!!!!!
    cufftComplex *dev_holo1, *dev_holo2;
    CHECK(cudaMalloc((void**)&dev_holo1,sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void**)&dev_holo2,sizeof(cufftComplex)*datLen*datLen));
    CuFillArrayComp<<<gridDatLen,block>>>(dev_holo1,imgMode2,datLen);
    // CuFillArrayComp<<<gridDatLen,block>>>(dev_holo1,0.5,datLen);
    CuFillArrayComp<<<gridDatLen,block>>>(dev_holo2,imgMode1,datLen);
    // CuFillArrayComp<<<gridDatLen,block>>>(dev_holo2,0.5,datLen);
    CuSetArrayCenterHalf<<<gridImgLen,block>>>(dev_holo1,dev_img2,imgLen);
    CuSetArrayCenterHalf<<<gridImgLen,block>>>(dev_holo2,dev_img1,imgLen);

    cufftComplex *dev_prholo;
    CHECK(cudaMalloc((void**)&dev_prholo,sizeof(cufftComplex)*datLen*datLen));

    getPRonGPU(dev_prholo,dev_holo1,dev_holo2,transPR,transInvPR,PRloops,datLen,blockSize);

    // cufftComplex *host;
    // host = (cufftComplex *)malloc(sizeof(cufftComplex)*datLen*datLen);
    // CHECK(cudaMemcpy(host,dev_prholo,sizeof(cufftComplex)*datLen*datLen,cudaMemcpyDeviceToHost));
    // unsigned char *inter;
    // inter = (unsigned char *)malloc(sizeof(unsigned char)*datLen*datLen);
    // for (int i = 0; i < datLen*datLen; i++){
    //     inter[i] = (unsigned char)((host[i].x*host[i].x+host[i].y*host[i].y));
    // }
    // cv::Mat interImg = cv::Mat(datLen,datLen,CV_8U,inter);
    // cv::imwrite("./inter.png",interImg);
    // free(host);
    // free(inter);

    CHECK(cudaMemcpy(outholo,dev_prholo,sizeof(cufftComplex)*datLen*datLen,cudaMemcpyDeviceToHost));

    float *dev_imp;
    CHECK(cudaMalloc((void**)&dev_imp,sizeof(float)*datLen*datLen));
    CuFillArrayFloat<<<gridDatLen,block>>>(dev_imp,255.0,datLen);

    cufftHandle plan;
    cufftPlan2d(&plan, datLen, datLen, CUFFT_C2C);

    cufftExecC2C(plan, dev_prholo, dev_prholo, CUFFT_FORWARD);
    CuFFTshift<<<gridDatLen,block>>>(dev_prholo, datLen);
    CuComplexMul<<<gridDatLen,block>>>(dev_prholo, dev_prholo, transF, datLen);

    cufftComplex *tmp_holo;
    float *tmp_imp;
    CHECK(cudaMalloc((void**)&tmp_holo,sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void**)&tmp_imp,sizeof(float)*datLen*datLen));
    
    for (int itr = 0; itr < loopCount; itr++){
        CuComplexMul<<<gridDatLen,block>>>(dev_prholo,dev_prholo,transInt,datLen);
        CHECK(cudaMemcpy(tmp_holo,dev_prholo,sizeof(cufftComplex)*datLen*datLen,cudaMemcpyDeviceToDevice));
        CuFFTshift<<<gridDatLen,block>>>(tmp_holo,datLen);
        cufftExecC2C(plan, tmp_holo, tmp_holo, CUFFT_INVERSE);
        CuInvFFTDiv<<<gridDatLen,block>>>(tmp_holo,(float)(datLen*datLen),datLen);
        CuGetAbs2FromComp<<<gridDatLen,block>>>(tmp_imp,tmp_holo,datLen);
        CuUpdateImposed<<<gridDatLen,block>>>(dev_imp,tmp_imp,datLen);
    }

    // cufftComplex *host;
    // host = (cufftComplex *)malloc(sizeof(cufftComplex)*datLen*datLen);
    // CHECK(cudaMemcpy(host,tmp_holo,sizeof(cufftComplex)*datLen*datLen,cudaMemcpyDeviceToHost));
    // unsigned char *inter;
    // inter = (unsigned char *)malloc(sizeof(unsigned char)*datLen*datLen);
    // for (int i = 0; i < datLen*datLen; i++){
    //     inter[i] = (unsigned char)((host[i].x*host[i].x+host[i].y*host[i].y));
    // }
    // cv::Mat interImg = cv::Mat(datLen,datLen,CV_8U,inter);
    // cv::imwrite("./inter.png",interImg);
    // free(host);
    // free(inter);

    float *dev_outImp;
    CHECK(cudaMalloc((void**)&dev_outImp,sizeof(float)*imgLen*imgLen));
    CuGetCenterHalf<<<gridImgLen,block>>>(dev_outImp,dev_imp,imgLen);

    CHECK(cudaMemcpy(floatout, dev_outImp, sizeof(float)*imgLen*imgLen, cudaMemcpyDeviceToHost));

    unsigned char *saveImp;
    CHECK(cudaMalloc((void**)&saveImp,sizeof(unsigned char)*imgLen*imgLen));
    CuNormFloatArrToChar<<<gridImgLen,block>>>(saveImp,dev_outImp,imgLen,1.0);
    // CuNormFloatArrToChar<<<gridImgLen,block>>>(saveImp,dev_outImp,imgLen,255.0);

    CHECK(cudaMemcpy(charout, saveImp, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyDeviceToHost));
    
    cufftDestroy(plan);
    CHECK(cudaFree(dev_in1));
    CHECK(cudaFree(dev_in2));
    CHECK(cudaFree(dev_img1));
    CHECK(cudaFree(dev_img2));
    CHECK(cudaFree(dev_holo1));
    CHECK(cudaFree(dev_holo2));
    CHECK(cudaFree(dev_prholo));
    CHECK(cudaFree(dev_imp));
    CHECK(cudaFree(tmp_holo));
    CHECK(cudaFree(tmp_imp));
    CHECK(cudaFree(dev_outImp));
    CHECK(cudaFree(saveImp));
}

void getBackGroundsWithoutBundle(float *backImg1,float *backImg2, unsigned char *cBackImg1, unsigned char *cBackImg2, Spinnaker::CameraPtr pCam[2], int imgLen, int loopCount){
    Spinnaker::ImagePtr pImg1, pImg2;
    unsigned char *cImg1, *cImg2;
    float *fImg1, *fImg2;
    fImg1 = (float *)calloc(imgLen*imgLen,sizeof(float));
    fImg2 = (float *)calloc(imgLen*imgLen,sizeof(float));
    for (int itr = 0; itr < loopCount; itr++)
    {
        std::cout << "iteration: " << itr << std::endl;
        pCam[0]->BeginAcquisition();
        pCam[1]->BeginAcquisition();
        pCam[0]->TriggerSoftware.Execute();
        pImg1 = pCam[0]->GetNextImage();
        pImg2 = pCam[1]->GetNextImage();
        pCam[0]->EndAcquisition();
        pCam[1]->EndAcquisition();
        cImg1 = (unsigned char *)pImg1->GetData();
        cImg2 = (unsigned char *)pImg2->GetData();
        for (int idx = 0; idx < imgLen*imgLen; idx++){
            fImg1[idx] += (float)((int)cImg1[idx]);
            fImg2[idx] += (float)((int)cImg2[idx]);
        }
    }
    std::cout << fImg1[0] << std::endl;

    // getNewFloatImage(fImg2,fImg2,coefa,imgLen);

    for (int idx = 0; idx < imgLen*imgLen; idx++){
        fImg1[idx] /= (float)(loopCount);
        fImg2[idx] /= (float)(loopCount);
    }

    for (int idx = 0; idx < imgLen*imgLen; idx++){
        backImg1[idx] = (float)fImg1[idx];
        backImg2[idx] = (float)fImg2[idx];
        cBackImg1[idx] = (unsigned char)fImg1[idx];
        cBackImg2[idx] = (unsigned char)fImg2[idx];
    }

    free(fImg1);
    free(fImg2);

}

void getBackGrounds(float *backImg1,float *backImg2, unsigned char *cBackImg1, unsigned char *cBackImg2, Spinnaker::CameraPtr pCam[2], int imgLen, int loopCount){
    float coefa[12];
    char *coefPath = "./coefa.dat";
    readCoef(coefPath,coefa);
    Spinnaker::ImagePtr pImg1, pImg2;
    unsigned char *cImg1, *cImg2;
    float *fImg1, *fImg2;
    fImg1 = (float *)calloc(imgLen*imgLen,sizeof(float));
    fImg2 = (float *)calloc(imgLen*imgLen,sizeof(float));
    for (int itr = 0; itr < loopCount; itr++)
    {
        std::cout << "iteration: " << itr << std::endl;
        pCam[0]->BeginAcquisition();
        pCam[1]->BeginAcquisition();
        pCam[0]->TriggerSoftware.Execute();
        pImg1 = pCam[0]->GetNextImage();
        pImg2 = pCam[1]->GetNextImage();
        pCam[0]->EndAcquisition();
        pCam[1]->EndAcquisition();
        cImg1 = (unsigned char *)pImg1->GetData();
        cImg2 = (unsigned char *)pImg2->GetData();
        for (int idx = 0; idx < imgLen*imgLen; idx++){
            fImg1[idx] += (float)((int)cImg1[idx]);
            fImg2[idx] += (float)((int)cImg2[idx]);
        }
    }

    // getNewFloatImage(fImg2,fImg2,coefa,imgLen);

    for (int idx = 0; idx < imgLen*imgLen; idx++){
        fImg1[idx] /= (float)(loopCount);
        fImg2[idx] /= (float)(loopCount);
    }

    getNewFloatImage(fImg2,fImg2,coefa,imgLen);

    for (int idx = 0; idx < imgLen*imgLen; idx++){
        backImg1[idx] = (float)fImg1[idx];
        backImg2[idx] = (float)fImg2[idx];
        cBackImg1[idx] = (unsigned char)(fImg1[idx]);
        cBackImg2[idx] = (unsigned char)(fImg2[idx]);
    }

    free(fImg1);
    free(fImg2);

}

void getSingleBackGrounds(float *backImg, unsigned char *cBackImg, Spinnaker::CameraPtr pCam, int imgLen, int loopCount){
    float coefa[12];
    char *coefPath = "./coefa.dat";
    readCoef(coefPath,coefa);
    Spinnaker::ImagePtr pImg;
    unsigned char *cImg;
    float *fImg;
    fImg = (float *)calloc(imgLen*imgLen,sizeof(float));
    for (int itr = 0; itr < loopCount; itr++)
    {
        std::cout << "iteration: " << itr << std::endl;
        pCam->BeginAcquisition();
        pCam->TriggerSoftware.Execute();
        pImg = pCam->GetNextImage();
        pCam->EndAcquisition();
        cImg = (unsigned char *)pImg->GetData();
        for (int idx = 0; idx < imgLen*imgLen; idx++){
            fImg[idx] += (float)((int)cImg[idx]);
        }
    }

    // getNewFloatImage(fImg2,fImg2,coefa,imgLen);

    for (int idx = 0; idx < imgLen*imgLen; idx++){
        fImg[idx] /= (float)(loopCount);
    }

    getNewFloatImage(fImg,fImg,coefa,imgLen);

    for (int idx = 0; idx < imgLen*imgLen; idx++){
        backImg[idx] = (float)fImg[idx];
        cBackImg[idx] = (unsigned char)(fImg[idx]);
    }

    free(fImg);
    free(cImg);

}

void getBackGroundsWithMode(float *backImg1,float *backImg2, unsigned char *cBackImg1, unsigned char *cBackImg2, Spinnaker::CameraPtr pCam[2], int imgLen, int loopCount){
    float coefa[12];
    char *coefPath = "./coefa.dat";
    readCoef(coefPath,coefa);
    Spinnaker::ImagePtr pImg1, pImg2;
    unsigned char *cImg1, *cImg2;
    int *vote1, *vote2;
    vote1 = (int *)calloc(imgLen*imgLen*256,sizeof(int));
    vote2 = (int *)calloc(imgLen*imgLen*256,sizeof(int));
    for (int itr = 0; itr < loopCount; itr++)
    {
        std::cout << "iteration: " << itr << std::endl;
        pCam[0]->BeginAcquisition();
        pCam[1]->BeginAcquisition();
        pCam[0]->TriggerSoftware.Execute();
        pImg1 = pCam[0]->GetNextImage();
        pImg2 = pCam[1]->GetNextImage();
        pCam[0]->EndAcquisition();
        pCam[1]->EndAcquisition();
        cImg1 = (unsigned char *)pImg1->GetData();
        cImg2 = (unsigned char *)pImg2->GetData();
        for (int idx = 0; idx < imgLen*imgLen; idx++){
            vote1[idx*256 + (int)cImg1[idx]] += 1;
            vote2[idx*256 + (int)cImg2[idx]] += 1;
        }
    }

    // getNewFloatImage(fImg2,fImg2,coefa,imgLen);

    for (int idx = 0; idx < imgLen*imgLen; idx++){
        int tmp1 = 0;
        int tmp2 = 0;
        int max1 = 0;
        int max2 = 0;
        for (int i = 0; i < 256; i++){
            if (vote1[idx*256 + i] > max1){
                tmp1 = i;
            }
            if (vote2[idx*256 + i] > max2){
                tmp2 = i;
            }
        }
        cImg1[idx] = tmp1;
        cImg2[idx] = tmp2;
    }

    getNewImage(cImg2,cImg2,coefa,imgLen);

    for (int idx = 0; idx < imgLen*imgLen; idx++){
        backImg1[idx] = (float)cImg1[idx];
        backImg2[idx] = (float)cImg2[idx];
        cBackImg1[idx] = (unsigned char)(cImg1[idx]);
        cBackImg2[idx] = (unsigned char)(cImg2[idx]);
    }

    free(vote1);
    free(vote2);
}

void getGaborReconstSlices(unsigned char *charin, float *backImg, float backMode, char *savepathheader, cufftComplex *transF, cufftComplex *transInt, int imgLen, int loopCount, int blockSize=16){
    int datLen = imgLen*2;
    dim3 gridImgLen((int)ceil((float)imgLen/(float)blockSize), (int)ceil((float)imgLen/(float)blockSize)), block(blockSize,blockSize);
    dim3 gridDatLen((int)ceil((float)datLen/(float)blockSize), (int)ceil((float)datLen/(float)blockSize));
    
    unsigned char *dev_in;
    CHECK(cudaMalloc((void**)&dev_in,sizeof(unsigned char)*imgLen*imgLen));
    float *dev_img;
    CHECK(cudaMalloc((void**)&dev_img,sizeof(float)*imgLen*imgLen));
    CHECK(cudaMemcpy(dev_in, charin, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyHostToDevice));
    CuCharToNormFloatArr<<<gridImgLen,block>>>(dev_img,dev_in,imgLen,1.0);
    // thrust::device_ptr<float> thimg(dev_img);
    // float meanImg = thrust::reduce(thimg,thimg+imgLen*imgLen, (float)0.0, thrust::plus<float>());
    // meanImg /= (float)(imgLen*imgLen);
    // std::cout << "Image mean: " << meanImg << std::endl;

    // Background Subtraction. Means to be 0.5
    float *dev_bkg;
    CHECK(cudaMalloc((void**)&dev_bkg,sizeof(float)*imgLen*imgLen));
    CHECK(cudaMemcpy(dev_bkg,backImg,sizeof(float)*imgLen*imgLen,cudaMemcpyHostToDevice));
    CuBackRem<<<gridImgLen,block>>>(dev_img,dev_bkg,backMode,imgLen);
    // CuBackRem<<<gridImgLen,block>>>(dev_img,dev_bkg,0.5,imgLen);
    cudaFree(dev_bkg);

    CuFloatSqrttoSqrt<<<gridImgLen,block>>>(dev_img,dev_img,imgLen);

    cufftComplex *dev_holo;
    CHECK(cudaMalloc((void**)&dev_holo,sizeof(cufftComplex)*datLen*datLen));
    CuFillArrayComp<<<gridDatLen,block>>>(dev_holo,sqrt(backMode),datLen);
    // CuFillArrayComp<<<gridDatLen,block>>>(dev_holo,0.5,datLen);
    CuSetArrayCenterHalf<<<gridImgLen,block>>>(dev_holo,dev_img,imgLen);

    // float *dev_imp;
    // CHECK(cudaMalloc((void**)&dev_imp,sizeof(float)*datLen*datLen));
    // CuFillArrayFloat<<<gridDatLen,block>>>(dev_imp,255.0,datLen);

    cufftHandle plan;
    cufftPlan2d(&plan, datLen, datLen, CUFFT_C2C);

    cufftExecC2C(plan, dev_holo, dev_holo, CUFFT_FORWARD);
    CuFFTshift<<<gridDatLen,block>>>(dev_holo, datLen);
    CuComplexMul<<<gridDatLen,block>>>(dev_holo, dev_holo, transF, datLen);

    cufftComplex *tmp_holo;
    // float *tmp_imp;
    unsigned char *dev_saveImg, *host_saveImg;
    CHECK(cudaMalloc((void**)&tmp_holo,sizeof(cufftComplex)*datLen*datLen));
    // CHECK(cudaMalloc((void**)&tmp_imp,sizeof(float)*datLen*datLen));
    CHECK(cudaMalloc((void**)&dev_saveImg,sizeof(unsigned char)*imgLen*imgLen));
    host_saveImg = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    Spinnaker::ImagePtr saveImg = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,host_saveImg);
    char savePath[200];

    for (int itr = 0; itr < loopCount; itr++){
        CuComplexMul<<<gridDatLen,block>>>(dev_holo,dev_holo,transInt,datLen);
        CHECK(cudaMemcpy(tmp_holo,dev_holo,sizeof(cufftComplex)*datLen*datLen,cudaMemcpyDeviceToDevice));
        CuFFTshift<<<gridDatLen,block>>>(tmp_holo,datLen);
        cufftExecC2C(plan, tmp_holo, tmp_holo, CUFFT_INVERSE);
        CuInvFFTDiv<<<gridDatLen,block>>>(tmp_holo,(float)(datLen*datLen),datLen);
        CuGetAbs2uintFromComp<<<gridImgLen,block>>>(dev_saveImg,tmp_holo,datLen);
        CHECK(cudaMemcpy(dev_saveImg, host_saveImg, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyDeviceToHost));
        saveImg->Convert(Spinnaker::PixelFormat_Mono8);
        sprintf(savePath,"%s/%05d.png",savepathheader,itr);
        saveImg->Save(savePath);
    }

    cufftDestroy(plan);
    CHECK(cudaFree(dev_in));
    CHECK(cudaFree(dev_img));
    CHECK(cudaFree(dev_holo));
    CHECK(cudaFree(tmp_holo));
    CHECK(cudaFree(dev_saveImg));
    free(host_saveImg);
    
}

void getBackRemGaborImposed(float *floatout, unsigned char *charout, unsigned char *in, float*backImg, cufftComplex *transF, cufftComplex *transInt, float backmode, int imgLen, int loopCount, int blockSize=16){
    int datLen = imgLen*2;
    dim3 gridImgLen((int)ceil((float)imgLen/(float)blockSize), (int)ceil((float)imgLen/(float)blockSize)), block(blockSize,blockSize);
    dim3 gridDatLen((int)ceil((float)datLen/(float)blockSize), (int)ceil((float)datLen/(float)blockSize));
    
    unsigned char *dev_in;
    CHECK(cudaMalloc((void**)&dev_in,sizeof(unsigned char)*imgLen*imgLen));
    float *dev_img;
    CHECK(cudaMalloc((void**)&dev_img,sizeof(float)*imgLen*imgLen));
    CHECK(cudaMemcpy(dev_in, in, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyHostToDevice));
    CuCharToNormFloatArr<<<gridImgLen,block>>>(dev_img,dev_in,imgLen,1.0);
    // thrust::device_ptr<float> thimg(dev_img);
    // float meanImg = thrust::reduce(thimg,thimg+imgLen*imgLen, (float)0.0, thrust::plus<float>());
    // meanImg /= (float)(imgLen*imgLen);
    // std::cout << "Image mean: " << meanImg << std::endl;

    // Background Subtraction. Means to be 0.5
    float imgMode = backmode;
    float *dev_bkg;
    CHECK(cudaMalloc((void**)&dev_bkg,sizeof(float)*imgLen*imgLen));
    CHECK(cudaMemcpy(dev_bkg,backImg,sizeof(float)*imgLen*imgLen,cudaMemcpyHostToDevice));
    CuBackRem<<<gridImgLen,block>>>(dev_img,dev_bkg,imgMode,imgLen);
    // CuBackRem<<<gridImgLen,block>>>(dev_img,dev_bkg,0.5,imgLen);
    cudaFree(dev_bkg);

    CuFloatSqrttoSqrt<<<gridImgLen,block>>>(dev_img,dev_img,imgLen);

    cufftComplex *dev_holo;
    CHECK(cudaMalloc((void**)&dev_holo,sizeof(cufftComplex)*datLen*datLen));
    CuFillArrayComp<<<gridDatLen,block>>>(dev_holo,sqrt(imgMode),datLen);
    // CuFillArrayComp<<<gridDatLen,block>>>(dev_holo,0.5,datLen);
    CuSetArrayCenterHalf<<<gridImgLen,block>>>(dev_holo,dev_img,imgLen);

    float *dev_imp;
    CHECK(cudaMalloc((void**)&dev_imp,sizeof(float)*datLen*datLen));
    CuFillArrayFloat<<<gridDatLen,block>>>(dev_imp,255.0,datLen);

    cufftHandle plan;
    cufftPlan2d(&plan, datLen, datLen, CUFFT_C2C);

    cufftExecC2C(plan, dev_holo, dev_holo, CUFFT_FORWARD);
    CuFFTshift<<<gridDatLen,block>>>(dev_holo, datLen);
    CuComplexMul<<<gridDatLen,block>>>(dev_holo, dev_holo, transF, datLen);

    cufftComplex *tmp_holo;
    float *tmp_imp;
    CHECK(cudaMalloc((void**)&tmp_holo,sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void**)&tmp_imp,sizeof(float)*datLen*datLen));
    
    for (int itr = 0; itr < loopCount; itr++){
        CuComplexMul<<<gridDatLen,block>>>(dev_holo,dev_holo,transInt,datLen);
        CHECK(cudaMemcpy(tmp_holo,dev_holo,sizeof(cufftComplex)*datLen*datLen,cudaMemcpyDeviceToDevice));
        CuFFTshift<<<gridDatLen,block>>>(tmp_holo,datLen);
        cufftExecC2C(plan, tmp_holo, tmp_holo, CUFFT_INVERSE);
        CuInvFFTDiv<<<gridDatLen,block>>>(tmp_holo,(float)(datLen*datLen),datLen);
        CuGetAbs2FromComp<<<gridDatLen,block>>>(tmp_imp,tmp_holo,datLen);
        CuUpdateImposed<<<gridDatLen,block>>>(dev_imp,tmp_imp,datLen);
    }

    float *dev_outImp;
    CHECK(cudaMalloc((void**)&dev_outImp,sizeof(float)*imgLen*imgLen));
    CuGetCenterHalf<<<gridImgLen,block>>>(dev_outImp,dev_imp,imgLen);

    CHECK(cudaMemcpy(floatout, dev_outImp, sizeof(float)*imgLen*imgLen, cudaMemcpyDeviceToHost));

    unsigned char *saveImp;
    CHECK(cudaMalloc((void**)&saveImp,sizeof(unsigned char)*imgLen*imgLen));
    CuNormFloatArrToChar<<<gridImgLen,block>>>(saveImp,dev_outImp,imgLen,1.0);
    // CuNormFloatArrToChar<<<gridImgLen,block>>>(saveImp,dev_outImp,imgLen,255.0);

    CHECK(cudaMemcpy(charout, saveImp, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    CHECK(cudaFree(dev_in));
    CHECK(cudaFree(dev_img));
    CHECK(cudaFree(dev_holo));
    CHECK(cudaFree(dev_imp));
    CHECK(cudaFree(tmp_holo));
    CHECK(cudaFree(tmp_imp));
    CHECK(cudaFree(dev_outImp));
    CHECK(cudaFree(saveImp));
}

void getImgAndPIV(Spinnaker::CameraPtr pCam[2],const int imgLen, const int gridSize, const int intrSize, const int srchSize, const float zF, const float dz, const float waveLen, const float dx,const int loopCount, const int blockSize){
    // Constant Declaration
    const int datLen = imgLen*2;
    dim3 grid((int)ceil((float)datLen/(float)blockSize),(int)ceil((float)datLen/(float)blockSize)), block(blockSize,blockSize);

    // Gabor Init
    float *d_sqr;
    cufftComplex *d_transF, *d_transF2, *d_transInt;
    CHECK(cudaMalloc((void **)&d_sqr, sizeof(float)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transF, sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transF2, sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transInt, sizeof(cufftComplex)*datLen*datLen));
    CuTransSqr<<<grid,block>>>(d_sqr,datLen,waveLen,dx);
    CuTransFunc<<<grid,block>>>(d_transF,d_sqr,-1.0*zF,waveLen,datLen,dx);
    CuTransFunc<<<grid,block>>>(d_transF2,d_sqr,-2.0*zF,waveLen,datLen,dx);
    // CuTransFunc<<<grid,block>>>(d_transF2,d_sqr,-2.0*zF,waveLen,datLen,dx);
    CuTransFunc<<<grid,block>>>(d_transInt,d_sqr,-1.0*dz,waveLen,datLen,dx);
    std::cout << "Gabor Init OK" << std::endl;

    // Camera Init
    Spinnaker::CameraPtr cam1 = pCam[0];
    Spinnaker::CameraPtr cam2 = pCam[1];
    cam1->BeginAcquisition();
    cam2->BeginAcquisition();
    cam1->TriggerSoftware.Execute();
    Spinnaker::ImagePtr pimg1 = cam1->GetNextImage();
    Spinnaker::ImagePtr pimg2 = cam2->GetNextImage();
    cam1->EndAcquisition();
    cam2->EndAcquisition();
    // unsigned char *charimg1 = (unsigned char *)pimg1->GetData();
    unsigned char *charimg1 = (unsigned char *)pimg1->GetData();
    // unsigned char *charimg2 = (unsigned char *)pimg2->GetData();
    unsigned char *charimg2 = (unsigned char *)pimg2->GetData();
    std::cout << "Cam 1 std: " << getImgSTD(charimg1,imgLen) << std::endl;
    std::cout << "Cam 2 std: " << getImgSTD(charimg2,imgLen) << std::endl;

    float *floatimp1, *floatimp2;
    floatimp1 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    floatimp2 = (float *)malloc(sizeof(float)*imgLen*imgLen);

    unsigned char *charimp1, *charimp2;
    charimp1 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    charimp2 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);



    //Save Imposed Image
    Spinnaker::ImagePtr saveImg1 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,charimp1);
    getGaborImposed(floatimp1,charimp1,charimg1,d_transF2,d_transInt,imgLen,loopCount);
    // getBackRemGaborImposed(floatimp1,charimp1,charimg1,backImg1,d_transF2,d_transInt,imgLen,loopCount);
    saveImg1->Convert(Spinnaker::PixelFormat_Mono8);
    
    Spinnaker::ImagePtr saveImg2 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,charimp2);
    getGaborImposed(floatimp2,charimp2,charimg2,d_transF,d_transInt,imgLen,loopCount);
    // getBackRemGaborImposed(floatimp2,charimp2,charimg2,backImg2,d_transF,d_transInt,imgLen,loopCount);
    saveImg2->Convert(Spinnaker::PixelFormat_Mono8);

    saveImg1->Save("./imposed1.jpg");
    saveImg2->Save("./imposed2.jpg");

    std::cout << "getGaborImposed OK" << std::endl;

    // Original image
    pimg1->Convert(Spinnaker::PixelFormat_Mono8);
    pimg2->Convert(Spinnaker::PixelFormat_Mono8);
    pimg1->Save("./outimg1.jpg");
    pimg2->Save("./outimg2.jpg");
    // pimg1->Release();
    // pimg2->Release();

    // Imshow visualization
    cv::Mat imp1 = cv::Mat(imgLen,imgLen,CV_8U,charimp1);
    cv::Mat imp2 = cv::Mat(imgLen,imgLen,CV_8U,charimp2);
    cv::namedWindow("Cam1",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Cam2",cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Cam1",0,0);
    cv::moveWindow("Cam2",520,0);
    cv::imshow("Cam1",imp1);
    cv::imshow("Cam2",imp2);
    cv::waitKey(10);
    // int key = cv::waitKey(1000);
    // if (key>=0) exit(0);

    // PIV
    int gridNum = imgLen/gridSize;
    float vecArrayX[(gridNum-1)*(gridNum-1)];
    float vecArrayY[(gridNum-1)*(gridNum-1)];
    float *pvecArrX = (float *)vecArrayX;
    float *pvecArrY = (float *)vecArrayY;
    getPIVMapOnGPU(pvecArrX,pvecArrY,floatimp1,floatimp2,imgLen,gridSize,intrSize,srchSize,blockSize);
    saveVecArray(pvecArrX,pvecArrY,gridSize,gridNum);
    plotVecFieldOnGnuplot(imgLen);

    // Finalize
    free(floatimp1);
    free(floatimp2);
    free(charimp1);
    free(charimp2);
    CHECK(cudaFree(d_sqr));
    CHECK(cudaFree(d_transF));
    CHECK(cudaFree(d_transF2));
    CHECK(cudaFree(d_transInt));

    // std::cout << "OK" << std::endl;
}

void getImgAndBundleAdjCheck(Spinnaker::CameraPtr pCam[2],const int imgLen, const int gridSize, const int intrSize, const int srchSize, const float zF, const float dz, const float waveLen, const float dx, const int blockSize){
    // Constant Declaretion
    const int datLen = imgLen*2;
    dim3 grid((int)ceil((float)datLen/(float)blockSize),(int)ceil((float)datLen/(float)blockSize)), block(blockSize,blockSize);

    // Gabor Init
    float *d_sqr;
    cufftComplex *d_transF, *d_transF2, *d_transInt;
    CHECK(cudaMalloc((void **)&d_sqr, sizeof(float)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transF, sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transF2, sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transInt, sizeof(cufftComplex)*datLen*datLen));
    CuTransSqr<<<grid,block>>>(d_sqr,datLen,waveLen,dx);
    CuTransFunc<<<grid,block>>>(d_transF,d_sqr,-1.0*zF,waveLen,datLen,dx);
    CuTransFunc<<<grid,block>>>(d_transF2,d_sqr,-2.0*zF,waveLen,datLen,dx);
    // CuTransFunc<<<grid,block>>>(d_transF2,d_sqr,-2.0*zF,waveLen,datLen,dx);
    CuTransFunc<<<grid,block>>>(d_transInt,d_sqr,-1.0*dz,waveLen,datLen,dx);
    std::cout << "Gabor Init OK" << std::endl;

    // Camera Init
    Spinnaker::CameraPtr cam1 = pCam[0];
    Spinnaker::CameraPtr cam2 = pCam[1];
    cam1->BeginAcquisition();
    cam2->BeginAcquisition();
    cam1->TriggerSoftware.Execute();
    Spinnaker::ImagePtr pimg1 = cam1->GetNextImage();
    Spinnaker::ImagePtr pimg2 = cam2->GetNextImage();
    cam1->EndAcquisition();
    cam2->EndAcquisition();
    // unsigned char *charimg1 = (unsigned char *)pimg1->GetData();
    unsigned char *charimg1 = (unsigned char *)pimg1->GetData();
    // unsigned char *charimg2 = (unsigned char *)pimg2->GetData();
    unsigned char *charimg2 = (unsigned char *)pimg2->GetData();

    // Bundle Adj Check
    float coefa[12];
    char *coefPath = "./coefa.dat";
    readCoef(coefPath,coefa);
    unsigned char *charimg3;
    charimg3 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    getNewImage(charimg3,charimg2,coefa,imgLen);

    float *floatimp1, *floatimp2;
    floatimp1 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    floatimp2 = (float *)malloc(sizeof(float)*imgLen*imgLen);

    unsigned char *charimp1, *charimp2;
    charimp1 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    charimp2 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);

    //Save Imposed Image
    Spinnaker::ImagePtr saveImg1 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,charimp1);
    getGaborImposed(floatimp1,charimp1,charimg1,d_transF2,d_transInt,imgLen,100);
    // getBackRemGaborImposed(floatimp1,charimp1,charimg1,backImg1,d_transF2,d_transInt,imgLen,100);
    saveImg1->Convert(Spinnaker::PixelFormat_Mono8);
    
    Spinnaker::ImagePtr saveImg2 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,charimp2);
    getGaborImposed(floatimp2,charimp2,charimg3,d_transF,d_transInt,imgLen,100);
    // getBackRemGaborImposed(floatimp2,charimp2,charimg3,backImg2,d_transF,d_transInt,imgLen,100);
    saveImg2->Convert(Spinnaker::PixelFormat_Mono8);

    saveImg1->Save("./bundle1.jpg");
    saveImg2->Save("./bundle2.jpg");

    std::cout << "getGaborImposed OK" << std::endl;

    // Original image
    // pimg1->Convert(Spinnaker::PixelFormat_Mono8);
    // pimg2->Convert(Spinnaker::PixelFormat_Mono8);
    // pimg1->Save("./outimg1.jpg");
    // pimg1->Save("./outimg2.jpg");

    // PIV
    int gridNum = imgLen/gridSize;
    float vecArrayX[(gridNum-1)*(gridNum-1)];
    float vecArrayY[(gridNum-1)*(gridNum-1)];
    float *pvecArrX = (float *)vecArrayX;
    float *pvecArrY = (float *)vecArrayY;
    getPIVMapOnGPU(pvecArrX,pvecArrY,floatimp1,floatimp2,imgLen,gridSize,intrSize,srchSize,blockSize);
    saveVecArray(pvecArrX,pvecArrY,gridSize,gridNum);
    plotVecFieldOnGnuplot(imgLen);

    // Finalize
    free(floatimp1);
    free(floatimp2);
    free(charimp1);
    free(charimp2);
    free(charimg3);
    CHECK(cudaFree(d_sqr));
    CHECK(cudaFree(d_transF));
    CHECK(cudaFree(d_transF2));
    CHECK(cudaFree(d_transInt));
}

void saveCufftComplex(cufftComplex *hostArr, char *filename, int datLen){
    FILE *fp;
    fp = fopen(filename, "w");
    if(fp == NULL){
        printf("%s seems not to be accesed! Quitting...\n",filename);
        exit(1);
    }
    for (int idx = 0; idx < datLen*datLen; idx++){
        fprintf(fp,"%f %f\n",hostArr[idx].x,hostArr[idx].y);
    }
    fclose(fp);
}