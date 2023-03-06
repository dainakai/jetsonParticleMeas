#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string>
#include <unistd.h>

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
 * @brief path の画像を読み込み正規化されたfloat配列 *img に格納する
 * @param path 画像のパス
 * @param imgLen 画像一辺の長さ
 * @param img (imgLen,imgLen)型配列のポインタ
 * @return なし
 */
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
        img[i] = (float)UIntImage[i]/255.0;
    }

    free(UIntImage);
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

/**
 * @fn
 * @brief vecArrayをvecArray.datに保存します
 * @param vecArrayX ベクトル場配列のポインタ
 * @param imgLen 画像一辺の長さ
 * @param gridNum gridNum-1 で各次元のベクトル個数
 * @return なし
 */
void saveVecArray(float *vecArrayX, float *vecArrayY, const int gridSize, const int gridNum){
    FILE *fp;
    if ((fp = fopen("vecArray.dat", "w")) == NULL) {
	    printf("File access not available! Quitting...!\n");
	    exit(1);
    }

    int n = gridNum-1;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fprintf(fp,"%d %d %lf %lf\n",(j+1)*gridSize,(i+1)*gridSize,vecArrayX[i*n+j],vecArrayY[i*n+j]);
        }
    }

    fclose(fp);
}

/**
 * @fn
 * @brief vecArrayをベクトル場表示します。
 * @param imgLen 画像一辺の長さ
 * @return なし
 */
void plotVecFieldOnGnuplot(const int imgLen){
    FILE *gp;
    const char* outputPath = "vecArrayPlot.pdf";
    const char* vecArrayDataPath = "vecArray.dat";
    const char* xxlabel = "{/Times-New-Roman:Italic=20 x} [pixel]";
    const char* yylabel = "{/Times-New-Roman:Italic=20 y} [pixel]";

    const int vecLenSclr = 10;
    
    if ((gp = popen("gnuplot", "w")) == NULL) {
	    printf("gnuplot is not available! Quitting...!\n");
	    exit(1);
    }

    fprintf(gp,"set terminal pdfcairo enhanced font 'Times New Roman,15' \n");
	fprintf(gp,"set output '%s'\n",outputPath);
	fprintf(gp,"set size ratio 1\n");
    fprintf(gp,"set xrange[0:%d]\n",imgLen);
    fprintf(gp,"set yrange[0:%d]\n",imgLen);
    // fprintf(gp,"set yrange[%d:0]\n",imgLen);
	fprintf(gp,"set palette rgb 33,13,10\n");

  // fprintf(gp,"set yrange reverse\n");

	fprintf(gp,"set xlabel '%s'offset 0.0,0.5\n",xxlabel);
	fprintf(gp,"set ylabel '%s'offset 0.5,0.0\n",yylabel);

	fprintf(gp,"plot '%s' using 1:2:(%d*$3):(%d*$4):(sqrt($3*$3+$4*$4))  w vector lc palette ti ''\n",vecArrayDataPath,vecLenSclr,vecLenSclr);

 	fflush(gp); //Clean up Data

	fprintf(gp, "exit\n"); // Quit gnuplot
	pclose(gp);
}


void readCoef(char *path, float a[12]){
    FILE *fp;
    fp = fopen(path,"r");
    if(fp == NULL){
        printf("%s seems not to exist! Quitting...\n",path);
        exit(1);
    }
    for (int i = 0; i < 12; i++)
    {
        char tmp[100];
        fgets(tmp,100,fp);
        a[i] = atof(tmp);
    }
    
    fclose(fp);
}

void getNewImage(float *out, float *img, float a[12], int imgLen){
    float *tmpImg;
    tmpImg = (float *)malloc(sizeof(float)*imgLen*imgLen);
    for (int i = 0; i < imgLen*imgLen; i++){
        tmpImg[i] = (float)img[i];
    }
    long int bkg = 0;
    for (int j = 0; j < imgLen*imgLen; j++){
        bkg += (float)tmpImg[j];
    }
    bkg = (float)bkg/(float)(imgLen*imgLen);


    for (int i = 0; i < imgLen; i++){
        for (int j = 0; j < imgLen; j++){
            int tmpX = (int)(round(a[0]+a[1]*j+a[2]*i+a[3]*j*j+a[4]*i*j+a[5]*i*i));
            int tmpY = (int)(round(a[6]+a[7]*j+a[8]*i+a[9]*j*j+a[10]*i*j+a[11]*i*i));
            if (tmpX>=0 && tmpX<imgLen && tmpY>=0 && tmpY <imgLen){
                out[i*imgLen+j] = (float)tmpImg[tmpY*imgLen+tmpX];
            }else{
                out[i*imgLen+j] = (float)bkg;
            }
        }
    }
    free(tmpImg);
}

int main(int argc, char** argv){    
    const int imgLen = 1024;
    const int intrSize = imgLen/8;
    const int gridSize = imgLen/8;
    const int srchSize = imgLen/4;
    const int gridNum = (int)(imgLen/gridSize);

    const int blockSize = 32;
    dim3 gridImgLen((int)ceil((float)imgLen/(float)blockSize), (int)ceil((float)imgLen/(float)blockSize)), block(blockSize,blockSize);

    const char * path1 = "./cam1.bmp";
    const char * path2 = "./cam2.bmp";

    float a[12];
    readCoef("./coefa.dat",a);

    float *fImg1, *fImg2;
    fImg1 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    fImg2 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    
    getFloatimage(fImg1,imgLen,path1);
    getFloatimage(fImg2,imgLen,path2);

    getNewImage(fImg2,fImg2,a,imgLen);
    
    float vecArrayX[(gridNum-1)*(gridNum-1)];
    float vecArrayY[(gridNum-1)*(gridNum-1)];
    getPIVMapOnGPU(vecArrayX,vecArrayY,fImg1,fImg2,imgLen,gridSize,intrSize,srchSize,blockSize);
    // saveVecArray(vecArrayX,vecArrayY,gridSize,gridNum);
    plotVecFieldOnGnuplot(imgLen);

    cudaDeviceReset();
    return 0;
}