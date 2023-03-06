#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <signal.h>
#include <time.h>
#include <sys/stat.h>
#include <fftw3.h>
#include "hostfunctions.h"
#include <chrono>

void transSqr(float *sqr, int datLen, float waveLen, float dx){
    for (int y = 0; y < datLen; y++)
    {
        for (int x = 0; x < datLen; x++)
        {
            // sqr[x+datLen*y] = 1.0;
            sqr[x+datLen*y] = 1.0 - (((float)x - (float)datLen/2.0)*waveLen/((float)datLen)/dx)*(((float)x - (float)datLen/2.0)*waveLen/((float)datLen)/dx) - (((float)y - (float)datLen/2.0)*waveLen/((float)datLen)/dx)*(((float)y - (float)datLen/2.0)*waveLen/((float)datLen)/dx);
        }
    }
}

void transFunc(fftwf_complex *output, float *sqr, float trans_z, float waveLen, int datLen, float dx){
    float tmp;
    float tmpx, tmpy;
    float uband = 1.0/waveLen/sqrt(2*trans_z/(float)datLen/dx + 1);

    for (int y = 0; y < datLen; y++){
        for (int x = 0; x < datLen; x++){
            tmp = 2.0*3.14159265358979*trans_z/waveLen*sqrt(sqr[x + datLen*y]);
            output[x + datLen*y][0] = cos(tmp);
            output[x + datLen*y][1] = sin(tmp);
            tmpx = abs(((float)x - (float)datLen/2.0)*waveLen/(float)datLen/dx);
            tmpy = abs(((float)y - (float)datLen/2.0)*waveLen/(float)datLen/dx);
            if (tmpx > uband || tmpy > uband){
                output[x + datLen*y][0] = 0.0;
                output[x + datLen*y][1] = 0.0;
            }
        }
    }
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

void fftShift(fftwf_complex *data, int datLen){
    fftwf_complex tmp1,tmp2;
    for (int y = 0; y < datLen/2; y++){
        for (int x = 0; x < datLen/2; x++){
            tmp1[0] = data[y*datLen/2+x][0];
            tmp1[1] = data[y*datLen/2+x][1];
            tmp2[0] = data[y*datLen/2+x+datLen/2][0];
            tmp2[1] = data[y*datLen/2+x+datLen/2][1];
            data[y*datLen/2+x][0] = data[(y+datLen/2)*datLen/2+x+datLen/2][0];
            data[y*datLen/2+x][1] = data[(y+datLen/2)*datLen/2+x+datLen/2][1];
            data[y*datLen/2+x+datLen/2][0] = data[(y+datLen/2)*datLen/2 + x][0];
            data[y*datLen/2+x+datLen/2][1] = data[(y+datLen/2)*datLen/2 + x][1];
            data[(y+datLen/2)*datLen/2+x+datLen/2][0] = tmp1[0];
            data[(y+datLen/2)*datLen/2+x+datLen/2][1] = tmp1[1];
            data[(y+datLen/2)*datLen/2 + x][0] = tmp2[0];
            data[(y+datLen/2)*datLen/2 + x][1] = tmp2[1];
        }
    }
}

void complexMul(fftwf_complex *out, fftwf_complex *in1, fftwf_complex *in2, int datLen){
    fftwf_complex tmp1, tmp2;
    for (int y = 0; y < datLen; y++){
        for (int x = 0; x < datLen; x++){
            tmp1[0] = in1[y*datLen+x][0];
            tmp1[1] = in1[y*datLen+x][1];
            tmp2[0] = in2[y*datLen+x][0];
            tmp2[1] = in2[y*datLen+x][1];
            out[y*datLen+x][0] = tmp1[0]*tmp2[0] - tmp1[1]*tmp2[1];
            out[y*datLen+x][1] = tmp1[0]*tmp2[1] + tmp1[1]*tmp2[0];
        }
    }
}

void fftInvDiv(fftwf_complex *data, int datLen){
    for (int idx = 0; idx < datLen*datLen; idx++){
        data[idx][0] /= (float)(datLen*datLen);
        data[idx][1] /= (float)(datLen*datLen);
    }
}

void getCompAngle(float *out, fftwf_complex *in, int datLen){
    for (int y = 0; y < datLen; y++){
        for (int  x = 0; x < datLen; x++){
            out[y*datLen+x] = atan2f(in[y*datLen+x][1],in[y*datLen+x][0]);
        }
    }
}

void getCompArray(fftwf_complex *out, float *magnitude, float *angle, int datLen){
    for (int y = 0; y < datLen; y++){
        for (int  x = 0; x < datLen; x++){
            out[y*datLen+x][0] = magnitude[y*datLen+x]*cosf(angle[y*datLen+x]);
            out[y*datLen+x][1] = magnitude[y*datLen+x]*sinf(angle[y*datLen+x]);
        }
    }
}

void getAbs2FromComp(float *out, fftwf_complex *in, int datLen){
    for (int idx = 0; idx < datLen*datLen; idx++)
    {
        out[idx] = in[idx][0]*in[idx][0] + in[idx][1]*in[idx][1];
    }
    
}

void updateImposed(float *imp, float *tmp, int datLen){
    for (int idx = 0; idx < datLen*datLen; idx++)
    {
        if (tmp[idx]<imp[idx]){
            imp[idx] = tmp[idx];
        }
    }
    
}

void getPRonCPU(fftwf_complex *prholo, fftwf_complex *holo1, fftwf_complex *holo2, fftwf_complex *trans, fftwf_complex *transInv, int iterations, int datLen){
    fftwf_complex *compAmp1, *compAmp2;
    float *phi1, *phi2;
    compAmp1 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    compAmp2 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    phi1 = (float *)malloc(sizeof(float)*datLen*datLen);
    phi2 = (float *)malloc(sizeof(float)*datLen*datLen);
    for (int idx = 0; idx < datLen*datLen; idx++){
        phi1[idx] = 0.0;
    }

    float *sqrtImg1, *sqrtImg2;
    sqrtImg1 = (float *)malloc(sizeof(float)*datLen*datLen);
    sqrtImg2 = (float *)malloc(sizeof(float)*datLen*datLen);
    for (int idx = 0; idx < datLen*datLen; idx++){
        sqrtImg1[idx] = sqrt(holo1[idx][0]);
    }
    for (int idx = 0; idx < datLen*datLen; idx++){
        sqrtImg2[idx] = sqrt(holo2[idx][0]);
    }
    for (int idx = 0; idx < datLen*datLen; idx++){
        compAmp1[idx][0] = sqrtImg1[idx];
    }

    fftwf_plan plan1to2, iplan2to2,plan2to1, iplan1to1;
    plan1to2 = fftwf_plan_dft_2d(datLen,datLen,compAmp1,compAmp2,FFTW_FORWARD,FFTW_ESTIMATE);
    iplan2to2 = fftwf_plan_dft_2d(datLen,datLen,compAmp2,compAmp2,FFTW_BACKWARD,FFTW_ESTIMATE);
    plan2to1 = fftwf_plan_dft_2d(datLen,datLen,compAmp2,compAmp1,FFTW_FORWARD,FFTW_ESTIMATE);
    iplan1to1 = fftwf_plan_dft_2d(datLen,datLen,compAmp1,compAmp1,FFTW_BACKWARD,FFTW_ESTIMATE);

    for (int itr = 0; itr < iterations; itr++)
    {
        fftwf_execute(plan1to2);
        fftShift(compAmp2,datLen);
        complexMul(compAmp2,compAmp2,trans,datLen);
        fftShift(compAmp2,datLen);
        fftwf_execute(iplan2to2);
        fftInvDiv(compAmp2,datLen);
        getCompAngle(phi2,compAmp2,datLen);

        getCompArray(compAmp2,sqrtImg2,phi2,datLen);

        fftwf_execute(plan2to1);
        fftShift(compAmp1,datLen);
        complexMul(compAmp1,compAmp1,transInv,datLen);
        fftShift(compAmp1,datLen);
        fftwf_execute(iplan1to1);
        fftInvDiv(compAmp1,datLen);
        getCompAngle(phi1,compAmp1,datLen);
        
        getCompArray(compAmp1,sqrtImg1,phi1,datLen);
    }
    
    for (int idx = 0; idx < datLen*datLen; idx++){
        prholo[idx][0] = compAmp1[idx][0];
        prholo[idx][1] = compAmp1[idx][1];
    }

    fftwf_free(compAmp1);
    fftwf_free(compAmp2);
    free(phi1);
    free(phi2);
    free(sqrtImg1);
    free(sqrtImg2);
    fftwf_destroy_plan(plan1to2);
    fftwf_destroy_plan(iplan1to1);
    fftwf_destroy_plan(plan2to1);
    fftwf_destroy_plan(iplan2to2);
}

void getPRImposed(float *floatout, unsigned char *charout, fftwf_complex *outholo, unsigned char *in1, unsigned char *in2, float *backImg1, float *backImg2, fftwf_complex *transF, fftwf_complex *transInt, fftwf_complex *transPR, fftwf_complex *transInvPR, int imgLen, int loopCount, int PRloops){
    int datLen = imgLen*2;
    float imgMode1 = getImgMode(in1,imgLen);
    float imgMode2 = getImgMode(in2,imgLen);

    fftwf_complex *holo1, *holo2;
    holo1 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    holo2 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);

    for (int idx = 0; idx < datLen*datLen; idx++){
        holo1[idx][0] = imgMode2;
        holo1[idx][1] = 0.0;
    }
    for (int idx = 0; idx < datLen*datLen; idx++){
        holo2[idx][0] = imgMode1;
        holo2[idx][1] = 0.0;
    }
    int xi,yi;
    for (int y = 0; y < imgLen; y++){
        for (int x = 0; x < imgLen; x++){
            xi = x + imgLen/2;
            yi = y + imgLen/2;
            if ((float)in1[y*imgLen+x]-backImg1[y*imgLen+x]+imgMode1<0.0){
                holo2[yi*datLen+xi][0] = 0.0;
            }else{
                holo2[yi*datLen+xi][0] = (float)in1[y*imgLen+x]-backImg1[y*imgLen+x]+imgMode1;
            }
            if ((float)in2[y*imgLen+x]-backImg2[y*imgLen+x]+imgMode2<0.0){
                holo1[yi*datLen+xi][0] = 0.0;
            }else{
                holo1[yi*datLen+xi][0] = (float)in2[y*imgLen+x]-backImg2[y*imgLen+x]+imgMode2;
            }
        }
    }

    fftwf_complex *prholo;
    prholo = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    getPRonCPU(prholo,holo1,holo2,transPR,transInvPR,PRloops,datLen);

    for (int idx = 0; idx < datLen*datLen; idx++){
        outholo[idx][0] = prholo[idx][0];
        outholo[idx][1] = prholo[idx][1];
    }
    
    float *imp;
    imp = (float *)malloc(sizeof(float)*datLen*datLen);
    for (int idx = 0; idx < datLen*datLen; idx++){
        imp[idx] = 255.0;
    }
    
    fftwf_complex *tmp_holo;
    tmp_holo = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    float *tmp_imp;
    tmp_imp = (float *)malloc(sizeof(float)*datLen*datLen);

    fftwf_plan plan, iplan;
    plan = fftwf_plan_dft_2d(datLen,datLen,prholo,prholo,FFTW_FORWARD,FFTW_ESTIMATE);
    iplan = fftwf_plan_dft_2d(datLen,datLen,tmp_holo,tmp_holo,FFTW_BACKWARD,FFTW_ESTIMATE);

    fftwf_execute(plan);
    fftShift(prholo,datLen);
    complexMul(prholo,prholo,transF,datLen);
    for (int itr = 0; itr < loopCount; itr++)
    {
        complexMul(prholo,prholo,transInt,datLen);
        for (int idx = 0; idx < datLen*datLen; idx++){
            tmp_holo[idx][0] = prholo[idx][0];
            tmp_holo[idx][1] = prholo[idx][1];
        }
        fftShift(tmp_holo,datLen);
        fftwf_execute(iplan);
        fftInvDiv(tmp_holo,datLen);
        getAbs2FromComp(tmp_imp,tmp_holo,datLen);
        updateImposed(imp,tmp_imp,datLen);
    }
    
    for (int y = 0; y < imgLen; y++){
        for (int x = 0; x < imgLen; x++){
            floatout[y*imgLen+x] = imp[(y+imgLen/2)*datLen+(x+imgLen/2)];
            charout[y*imgLen+x] = (unsigned char)imp[(y+imgLen/2)*datLen+(x+imgLen/2)];
        }
    }
    
    fftwf_destroy_plan(plan);
    fftwf_destroy_plan(iplan);
    fftwf_free(holo1);
    fftwf_free(holo2);
    fftwf_free(prholo);
    fftwf_free(tmp_holo);
    free(imp);
    free(tmp_imp);
}

void getBackRemGaborImposed(float *floatout, unsigned char *charout, unsigned char *in, float*backImg, fftwf_complex *transF, fftwf_complex *transInt, int imgLen, int loopCount){
    int datLen = imgLen*2;
    float imgMode = getImgMode(in,imgLen);

    fftwf_complex *holo;
    holo = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);

    for (int idx = 0; idx < datLen*datLen; idx++){
        holo[idx][0] = imgMode;
        holo[idx][1] = 0.0;
    }

    int xi,yi;
    for (int y = 0; y < imgLen; y++){
        for (int x = 0; x < imgLen; x++){
            xi = x + imgLen/2;
            yi = y + imgLen/2;
            if ((float)in[y*imgLen+x]-backImg[y*imgLen+x]+imgMode<0.0){
                holo[yi*datLen+xi][0] = 0.0;
            }else{
                holo[yi*datLen+xi][0] = (float)in[y*imgLen+x]-backImg[y*imgLen+x]+imgMode;
            }
        }
    }

    float *imp;
    imp = (float *)malloc(sizeof(float)*datLen*datLen);
    for (int idx = 0; idx < datLen*datLen; idx++){
        imp[idx] = 255.0;
    }
    
    fftwf_complex *tmp_holo;
    tmp_holo = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    float *tmp_imp;
    tmp_imp = (float *)malloc(sizeof(float)*datLen*datLen);

    fftwf_plan plan, iplan;
    plan = fftwf_plan_dft_2d(datLen,datLen,holo,holo,FFTW_FORWARD,FFTW_ESTIMATE);
    iplan = fftwf_plan_dft_2d(datLen,datLen,tmp_holo,tmp_holo,FFTW_BACKWARD,FFTW_ESTIMATE);

    fftwf_execute(plan);
    fftShift(holo,datLen);
    complexMul(holo,holo,transF,datLen);
    for (int itr = 0; itr < loopCount; itr++)
    {
        complexMul(holo,holo,transInt,datLen);
        for (int idx = 0; idx < datLen*datLen; idx++){
            tmp_holo[idx][0] = holo[idx][0];
            tmp_holo[idx][1] = holo[idx][1];
        }
        fftShift(tmp_holo,datLen);
        fftwf_execute(iplan);
        fftInvDiv(tmp_holo,datLen);
        getAbs2FromComp(tmp_imp,tmp_holo,datLen);
        updateImposed(imp,tmp_imp,datLen);
    }
    
    for (int y = 0; y < imgLen; y++){
        for (int x = 0; x < imgLen; x++){
            floatout[y*imgLen+x] = imp[(y+imgLen/2)*datLen+(x+imgLen/2)];
            charout[y*imgLen+x] = (unsigned char)imp[(y+imgLen/2)*datLen+(x+imgLen/2)];
        }
    }
    
    fftwf_destroy_plan(plan);
    fftwf_destroy_plan(iplan);
    fftwf_free(holo);
    fftwf_free(tmp_holo);
    free(imp);
    free(tmp_imp);
}

void savefftComplex(fftwf_complex *hostArr, char *filename, int datLen){
    FILE *fp;
    fp = fopen(filename, "w");
    if(fp == NULL){
        printf("%s seems not to be accesed! Quitting...\n",filename);
        exit(1);
    }
    for (int idx = 0; idx < datLen*datLen; idx++){
        fprintf(fp,"%f %f\n",hostArr[idx][0],hostArr[idx][1]);
    }
    fclose(fp);
}

volatile sig_atomic_t e_flag = 0;
void abrt_handler(int sig){
    e_flag = 1;
}

int main(int argc, char** argv){
    std::cout << argv[0] << " Starting..." << std::endl;

    // Key interrupt handling
    if ( signal(SIGTSTP, abrt_handler) == SIG_ERR ) {
        exit(1);
    }
    
    // Parameters
    const float camExposure = 200.0;
    const float gainInit = 1.0;

    const int OffsetX = atoi(argv[1]);
    // const int OffsetX = 592;
    const int OffsetY = atoi(argv[2]);
    // const int OffsetY = 514;

    float gain1,gain2;
    std::tie(gain1,gain2) = readGain("./gain.dat");
    
    const int imgLen = 512;
    const int intrSize = imgLen/8;
    const int gridSize = imgLen/8;
    const int srchSize = imgLen/4;
    const int gridNum = (int)(imgLen/gridSize);

    const int prLoop = 10;
    const int backgroundLoops = 10;
    const int ImposedLoop = 200;

    const float prDist = 5.0*1000.0; // 60 mm
    const float zF = 5.0*1000.0;
    const float dz = 200.0;
    const float waveLen = 0.532;
    const float dx = 3.45/0.5;

    const int blockSize = 16;

    // Camera Init
    Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
    Spinnaker::CameraList camList = system->GetCameras();
    unsigned int numCameras = camList.GetSize();
    if (numCameras==0){
        std::cout << "No Cameras are Connected! Quitting..." << std::endl;
        exit(1);
    }
    Spinnaker::CameraPtr pCam[numCameras];
    std::cout << "Camera" << "\t" << "ModelName" << "\t\t\t" << "SerialNumber" << std::endl;
    for (int i = 0; i < numCameras; i++){
        pCam[i] = camList.GetByIndex(i);
        pCam[i]->Init();
        Spinnaker::GenICam::gcstring modelName = pCam[i]->TLDevice.DeviceModelName.GetValue();
        Spinnaker::GenICam::gcstring serialNum = pCam[i]->TLDevice.DeviceSerialNumber.GetValue();
        std::cout << i << "\t" << modelName << "\t" << serialNum << std::endl;
    }
    if (numCameras != 2){
        std::cout << "Number of Connected Cameras is not 2. Quitting..." << std::endl;
        exit(0);
    }
    std::cout << "Camera Enum OK" << std::endl;

    // Camera Setup
    cameraSetup(pCam,imgLen,OffsetX,OffsetY,camExposure,gain1,gain2);

    // Processing

    // Constant Declaration
    const int datLen = imgLen*2;

    // Propagation Init
    float *sqr;
    fftwf_complex *transF, *transInt, *transPR, *transPRInv, *transZInterval;
    sqr = (float *)malloc(sizeof(float)*datLen*datLen);
    transF = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    transInt = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    transPR = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    transPRInv = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);
    transSqr(sqr,datLen,waveLen,dx);
    transFunc(transF,sqr,-1.0*zF,waveLen,datLen,dx);
    transFunc(transInt,sqr,-1.0*dz,waveLen,datLen,dx);
    transFunc(transPR,sqr,prDist,waveLen,datLen,dx);
    transFunc(transPRInv,sqr,-1.0*prDist,waveLen,datLen,dx);
    std::cout << "PR Init OK" << std::endl;

    // Background Processing
    pCam[0]->BeginAcquisition();
    pCam[1]->BeginAcquisition();
    pCam[0]->TriggerSoftware.Execute();
    Spinnaker::ImagePtr pImg1 = pCam[0]->GetNextImage();
    Spinnaker::ImagePtr pImg2 = pCam[1]->GetNextImage();
    pCam[0]->EndAcquisition();
    pCam[1]->EndAcquisition();

    std::cout << "tmp OK" << std::endl;

    unsigned char *charimg1 = (unsigned char *)pImg1->GetData();
    unsigned char *charimg2 = (unsigned char *)pImg2->GetData();


    float *bImg1, *bImg2;
    bImg1 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    bImg2 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    unsigned char *cBackImg1, *cBackImg2;
    cBackImg1 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    cBackImg2 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    Spinnaker::ImagePtr saveBack1 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,cBackImg1);
    Spinnaker::ImagePtr saveBack2 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,cBackImg2);
    getBackGrounds(bImg1,bImg2,cBackImg1,cBackImg2,pCam,imgLen,backgroundLoops);
    saveBack1->Convert(Spinnaker::PixelFormat_Mono8);
    saveBack2->Convert(Spinnaker::PixelFormat_Mono8);
    saveBack1->Save("./CPUmeanBkg1.png");
    saveBack2->Save("./CPUmeanBkg2.png");
    std::cout << "test" << std::endl;

    // Process Bridge
    std::cout << "Background Acquisition Completed. PR Reconstruction will be started in..." << std::endl;
    for (int i = 0; i <= 5; i++){
        int sec = 5-i;
        sleep(1);
        std::cout << sec << std::endl;
    }
    

    // Start Processing
    float coefa[12];
    char *coefPath = "./coefa.dat";
    readCoef(coefPath,coefa);
    float *outPrImp, *outGaborImp; 
    outPrImp = (float *)malloc(sizeof(float)*imgLen*imgLen);
    outGaborImp = (float *)malloc(sizeof(float)*imgLen*imgLen);
    unsigned char *outPrCImp, *outGaborCImp;
    outPrCImp = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    outGaborCImp = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    Spinnaker::ImagePtr savePrImp = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,outPrCImp);
    Spinnaker::ImagePtr saveGaborImp = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,outGaborCImp);

    struct tm tm;
    time_t t = time(NULL);
    localtime_r(&t,&tm);
    char dirPath[100];
    sprintf(dirPath,"./CPUImposedOutput/");
    mkdir(dirPath,0777);
    sprintf(dirPath,"./CPUImposedOutput/%02d%02d/",tm.tm_mon+1,tm.tm_mday);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./CPUImposedOutput/%02d%02d/%02d%02d%02d%02d",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./CPUImposedOutput/%02d%02d/%02d%02d%02d%02d/Gabor/",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./CPUImposedOutput/%02d%02d/%02d%02d%02d%02d/PR/",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./CPUImposedOutput/%02d%02d/%02d%02d%02d%02d/Holo/",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./CPUImposedOutput/%02d%02d/%02d%02d%02d%02d",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);

    char savePrPath[150];
    char saveGaborPath[150];
    char saveHoloPath[150];

    cv::Mat PrImp, GaborImp;
    cv::namedWindow("PR",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gabor",cv::WINDOW_AUTOSIZE);
    cv::moveWindow("PR",520,0);
    cv::moveWindow("Gabor",0,0);

    fftwf_complex *holoData;
    holoData = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*datLen*datLen);

    auto now = std::chrono::high_resolution_clock::now();

    int num = 0;
    while(!e_flag){
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - now);
        std::cout << duration.count() << std::endl;
        now = std::chrono::high_resolution_clock::now();

        // std::cout << "Recording Holograms" << std::endl;
        pCam[0]->BeginAcquisition();
        pCam[1]->BeginAcquisition();
        pCam[0]->TriggerSoftware.Execute();
        pImg1 = pCam[0]->GetNextImage();
        pImg2 = pCam[1]->GetNextImage();
        pCam[0]->EndAcquisition();
        pCam[1]->EndAcquisition();

        charimg1 = (unsigned char *)pImg1->GetData();
        charimg2 = (unsigned char *)pImg2->GetData();

        // std::cout << "Image Transforming" << std::endl;
        getNewImage(charimg2,charimg2,coefa,imgLen);

        // std::cout << "PR Imposing" << std::endl;
        // getPRImposed(outPrImp,outPrCImp,holoData,charimg1,charimg2,bImg1,bImg2,transF,transInt,transPR,transPRInv,imgLen,ImposedLoop,prLoop);
        // std::cout << "Gabor Imposing" << std::endl;
        getBackRemGaborImposed(outGaborImp,outGaborCImp,charimg2,bImg2,transF,transInt,imgLen,ImposedLoop);

        // saveGaborImp->Convert(Spinnaker::PixelFormat_Mono8);
        // savePrImp->Convert(Spinnaker::PixelFormat_Mono8);
        
        // sprintf(savePrPath,"%s/PR/%05d.png",dirPath,num);
        // sprintf(saveGaborPath,"%s/Gabor/%05d.png",dirPath,num);
        // sprintf(saveHoloPath,"%s/Holo/%05d.dat",dirPath,num);

        // saveGaborImp->Save(saveGaborPath);
        // savePrImp->Save(savePrPath);

        // savefftComplex(holoData,saveHoloPath,datLen);

        // PrImp = cv::Mat(imgLen,imgLen,CV_8U,outPrCImp);
        // GaborImp = cv::Mat(imgLen,imgLen,CV_8U,outGaborCImp);
        // cv::imshow("PR",PrImp);
        // cv::imshow("Gabor",GaborImp);
        // cv::waitKey(10);

        num += 1;
    }
    
    // // Finalize
    free(sqr);
    fftwf_free(transF);
    fftwf_free(transInt);
    fftwf_free(transPR);
    fftwf_free(transPRInv);
    free(bImg1);
    free(bImg2);
    free(cBackImg1);
    free(cBackImg2);
    free(outPrImp);
    free(outPrCImp);
    free(outGaborImp);
    free(outGaborCImp);
    free(holoData);

    camList.Clear();
    system->ReleaseInstance();

    return 0;
}


