#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <string>

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

void getFloatimage(float *img, const int imgLen, const char* path){
    unsigned char *UIntImage;
    UIntImage = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);

    FILE *fp;
    fp = fopen(path,"rb");
    if(fp == NULL){
        printf("%s seems not to exist! Quitting...\n",path);
        exit(1);
    }
    fseek(fp,1078,0);
    fread(UIntImage,sizeof(unsigned char),imgLen*imgLen,fp);
    fclose(fp);
    
    for (int i = 0; i < imgLen*imgLen; i++){
        img[i] = (float)UIntImage[i]/128.0;
    }

    std::cout << (float)img[512*1024] << std::endl;
    free(UIntImage);
}

__global__ void CuFloatSqrt(float *dst, cufftComplex *src, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        // dst[y*datLen + x] = src[y*datLen+x].x;
        dst[y*datLen + x] = sqrt(src[y*datLen+x].x);
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

__global__ void CuFillCompArrayByFloatArray(cufftComplex *dst, float *src, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        dst[y*datLen + x].x = src[y*datLen+x];
        dst[y*datLen + x].y = 0.0;
        // printf("%lf ",dst[y*datLen + x].x);
    }
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

__global__ void CuNormFloatArrToChar(unsigned char *out, float *in, int datLen, float Norm){
    // dim3 grid((width+31)/32, (height+31)/32), block(32,32)
    // CHECH deviceQuery and make sure threads per block are 1024!!!!

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        out[y*datLen + x] = (unsigned char)(in[y*datLen + x]*Norm);
        // printf("%lf ",out[y*datLen+x]);
    }
}

__global__ void CuSetArrayCenter(cufftComplex *out, float *img, int imgLen){
    // dim3 grid((width+31)/32, (height+31)/32), block(32,32)
    // CHECH deviceQuery and make sure threads per block are 1024!!!!

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < imgLen) && (y < imgLen) ){
        out[y*imgLen + x].x = img[y*imgLen+x]; 
        out[y*imgLen + x].y = 0.0; 
        // printf("%lf ",out[y*imgLen + x].x);
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

__global__ void CuUpdateImposed(float *imp, float *tmp, int datLen){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x < datLen) && (y < datLen) ){
        if (tmp[y*datLen+x] < imp[y*datLen+x]){
            imp[y*datLen+x] = tmp[y*datLen+x];
        }
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

void getPRonGPU(cufftComplex *dev_holo, cufftComplex *holo1, cufftComplex *holo2, cufftComplex *trans, cufftComplex *transInv, int iterations, int datLen, int blockSize){
    dim3 gridDatLen((int)ceil((float)datLen/(float)blockSize), (int)ceil((float)datLen/(float)blockSize)), block(blockSize,blockSize);
    cufftComplex *compAmp1, *compAmp2;
    float *phi1, *phi2;
    CHECK(cudaMalloc((void**)&compAmp1,sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void**)&compAmp2,sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void**)&phi1,sizeof(float)*datLen*datLen));
    CHECK(cudaMalloc((void**)&phi2,sizeof(float)*datLen*datLen));
    CuFillArrayFloat<<<gridDatLen,block>>>(phi1,1.0,datLen);

    float *sqrtImg1, *sqrtImg2;
    CHECK(cudaMalloc((void**)&sqrtImg1,sizeof(float)*datLen*datLen));
    CHECK(cudaMalloc((void**)&sqrtImg2,sizeof(float)*datLen*datLen));
    CuFloatSqrt<<<gridDatLen,block>>>(sqrtImg1,holo1,datLen);
    CuFloatSqrt<<<gridDatLen,block>>>(sqrtImg2,holo2,datLen);
    CuFillCompArrayByFloatArray<<<gridDatLen,block>>>(compAmp1,sqrtImg1,datLen);

    cufftHandle plan;
    cufftPlan2d(&plan, datLen, datLen, CUFFT_C2C);

    for (int itr = 0; itr < iterations; itr++){
        std::cout << itr << std::endl;
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
    
    CHECK(cudaMemcpy(dev_holo,compAmp1,sizeof(cufftComplex)*datLen*datLen,cudaMemcpyDeviceToDevice));
    // CuFillArrayComp<<<gridDatLen,block>>>(dev_holo,1.0,datLen);

    CHECK(cudaFree(compAmp1));
    CHECK(cudaFree(compAmp2));
    CHECK(cudaFree(phi1));
    CHECK(cudaFree(phi2));
    CHECK(cudaFree(sqrtImg1));
    CHECK(cudaFree(sqrtImg2));
    cufftDestroy(plan);
}

int main(){
    const char * path1 = "./test60.bmp";
    const char * path2 = "./test120.bmp";

    const int imgLen = 1024;
    const float zF = 60.0*1000.0;
    const float dx = 6.9;
    const float dz = 25.0;
    const int loopCount = 20;
    const float prDist = 60.0*1000.0;
    const float waveLen = 0.532;
    const int blockSize = 16;
    dim3 gridImgLen((int)ceil((float)imgLen/(float)blockSize), (int)ceil((float)imgLen/(float)blockSize)), block(blockSize,blockSize);
    
    float *fImg1, *fImg2;
    fImg1 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    fImg2 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    
    getFloatimage(fImg1,imgLen,path1);
    getFloatimage(fImg2,imgLen,path2);
    
    // cImg1 = cvImg1.data;
    // cImg2 = cvImg2.data;

    // for (int i = 0; i < imgLen*imgLen; i++)
    // {
    //     std::cout << (int)cImg1[i] << " ";
    // }
    // std::cout << std::endl;
    
    float *dev_img1, *dev_img2;
    CHECK(cudaMalloc((void**)&dev_img1,sizeof(float)*imgLen*imgLen));
    CHECK(cudaMalloc((void**)&dev_img2,sizeof(float)*imgLen*imgLen));    
    CHECK(cudaMemcpy(dev_img1, fImg1, sizeof(float)*imgLen*imgLen, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_img2, fImg2, sizeof(float)*imgLen*imgLen, cudaMemcpyHostToDevice));

    cufftComplex *dev_holo1, *dev_holo2;
    CHECK(cudaMalloc((void**)&dev_holo1,sizeof(cufftComplex)*imgLen*imgLen));
    CHECK(cudaMalloc((void**)&dev_holo2,sizeof(cufftComplex)*imgLen*imgLen));
    CuSetArrayCenter<<<gridImgLen,block>>>(dev_holo1,dev_img1,imgLen);
    CuSetArrayCenter<<<gridImgLen,block>>>(dev_holo2,dev_img2,imgLen);

    cufftComplex *dev_prholo;
    CHECK(cudaMalloc((void**)&dev_prholo,sizeof(cufftComplex)*imgLen*imgLen));

    float *d_sqr;
    cufftComplex *d_transF, *d_transInt, *d_transPR, *d_transPRInv;
    CHECK(cudaMalloc((void **)&d_sqr, sizeof(float)*imgLen*imgLen));
    CHECK(cudaMalloc((void **)&d_transF, sizeof(cufftComplex)*imgLen*imgLen));
    CHECK(cudaMalloc((void **)&d_transInt, sizeof(cufftComplex)*imgLen*imgLen));
    CHECK(cudaMalloc((void **)&d_transPR, sizeof(cufftComplex)*imgLen*imgLen));
    CHECK(cudaMalloc((void **)&d_transPRInv, sizeof(cufftComplex)*imgLen*imgLen));
    CuTransSqr<<<gridImgLen,block>>>(d_sqr,imgLen,waveLen,dx);
    CuTransFunc<<<gridImgLen,block>>>(d_transF,d_sqr,-zF,waveLen,imgLen,dx);
    CuTransFunc<<<gridImgLen,block>>>(d_transInt,d_sqr,-dz,waveLen,imgLen,dx);
    CuTransFunc<<<gridImgLen,block>>>(d_transPR,d_sqr,prDist,waveLen,imgLen,dx);
    CuTransFunc<<<gridImgLen,block>>>(d_transPRInv,d_sqr,-1.0*prDist,waveLen,imgLen,dx);
    std::cout << "PR Init OK" << std::endl;

    getPRonGPU(dev_prholo,dev_holo1,dev_holo2,d_transPR,d_transPRInv,10,imgLen,blockSize);

    cufftComplex *host;
    host = (cufftComplex *)malloc(sizeof(cufftComplex)*imgLen*imgLen);
    CHECK(cudaMemcpy(host,dev_prholo,sizeof(cufftComplex)*imgLen*imgLen,cudaMemcpyDeviceToHost));

    unsigned char *inter;
    inter = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);

    for (int i = 0; i < imgLen*imgLen; i++)
    {
        inter[i] = (unsigned char)(128.0*(host[i].x*host[i].x+host[i].y*host[i].y));
    }
    cv::Mat interImg = cv::Mat(imgLen,imgLen,CV_8U,inter);
    cv::imwrite("./inter.bmp",interImg);


    float *dev_imp;
    CHECK(cudaMalloc((void**)&dev_imp,sizeof(float)*imgLen*imgLen));
    CuFillArrayFloat<<<gridImgLen,block>>>(dev_imp,1.0,imgLen);

    cufftHandle plan;
    cufftPlan2d(&plan, imgLen, imgLen, CUFFT_C2C);

    cufftExecC2C(plan, dev_prholo, dev_prholo, CUFFT_FORWARD);
    CuFFTshift<<<gridImgLen,block>>>(dev_prholo, imgLen);
    CuComplexMul<<<gridImgLen,block>>>(dev_prholo, dev_prholo, d_transF, imgLen);

    cufftComplex *tmp_holo;
    float *tmp_imp;
    CHECK(cudaMalloc((void**)&tmp_holo,sizeof(cufftComplex)*imgLen*imgLen));
    CHECK(cudaMalloc((void**)&tmp_imp,sizeof(float)*imgLen*imgLen));


    for (int itr = 0; itr < loopCount; itr++){
        CuComplexMul<<<gridImgLen,block>>>(dev_prholo,dev_prholo,d_transInt,imgLen);
        CHECK(cudaMemcpy(tmp_holo,dev_prholo,sizeof(cufftComplex)*imgLen*imgLen,cudaMemcpyDeviceToDevice));
        CuFFTshift<<<gridImgLen,block>>>(tmp_holo,imgLen);
        cufftExecC2C(plan, tmp_holo, tmp_holo, CUFFT_INVERSE);
        CuInvFFTDiv<<<gridImgLen,block>>>(tmp_holo,(float)(imgLen*imgLen),imgLen);
        CuGetAbsFromComp<<<gridImgLen,block>>>(tmp_imp,tmp_holo,imgLen);
        CuUpdateImposed<<<gridImgLen,block>>>(dev_imp,tmp_imp,imgLen);
    }

    // float *dev_outImp;
    // CHECK(cudaMalloc((void**)&dev_outImp,sizeof(float)*imgLen*imgLen));


    unsigned char *saveImp;
    CHECK(cudaMalloc((void**)&saveImp,sizeof(unsigned char)*imgLen*imgLen));
    CuNormFloatArrToChar<<<gridImgLen,block>>>(saveImp,dev_imp,imgLen,128.0);

    unsigned char *saveHost;
    saveHost = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    CHECK(cudaMemcpy(saveHost, saveImp, sizeof(unsigned char)*imgLen*imgLen, cudaMemcpyDeviceToHost));

    // for (int i = 0; i < imgLen*imgLen; i++)
    // {
    //     std::cout << (int)saveHost[i] << " ";
    // }
    // std::cout << std::endl;

    cv::Mat outImg = cv::Mat(imgLen,imgLen,CV_8U,saveHost);
    cv::imwrite("./imposed.bmp",outImg);

    cufftDestroy(plan);
    CHECK(cudaFree(dev_img1));
    CHECK(cudaFree(dev_img2));
    CHECK(cudaFree(dev_holo1));
    CHECK(cudaFree(dev_holo2));
    CHECK(cudaFree(dev_prholo));
    CHECK(cudaFree(dev_imp));
    CHECK(cudaFree(tmp_holo));
    CHECK(cudaFree(tmp_imp));
    // CHECK(cudaFree(dev_outImp));
    CHECK(cudaFree(saveImp));
    free(saveHost);
    free(fImg1);
    free(fImg2);

    return 0;
}