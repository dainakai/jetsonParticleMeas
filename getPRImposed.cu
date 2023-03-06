#include "CUDAfunctions.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
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
#include <chrono>

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
    const float camExposure = 129.0;
    const float gainInit = 0.0;

    const int OffsetX = atoi(argv[1]);
    // const int OffsetX = 592;
    const int OffsetY = atoi(argv[2]);
    // const int OffsetY = 514;

    float gain1,gain2;
    std::tie(gain1,gain2) = readGain("./gain.dat");
    
    const int imgLen = 512;
    // const int intrSize = imgLen/8;
    // const int gridSize = imgLen/8;
    // const int srchSize = imgLen/4;
    // const int gridNum = (int)(imgLen/gridSize);

    const int prLoop = 10;
    const int backgroundLoops = 100;
    const int ImposedLoop = 200;

    const float prDist = 10.0*1000.0; // 10 mm
    const float zF = 10.0*1000.0;
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
    dim3 grid((int)ceil((float)datLen/(float)blockSize),(int)ceil((float)datLen/(float)blockSize)), block(blockSize,blockSize);

    // Propagation Init
    float *d_sqr;
    cufftComplex *d_transF, *d_transInt, *d_transPR, *d_transPRInv, *d_transZInterval;;
    CHECK(cudaMalloc((void **)&d_sqr, sizeof(float)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transF, sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transInt, sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transPR, sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transPRInv, sizeof(cufftComplex)*datLen*datLen));
    CuTransSqr<<<grid,block>>>(d_sqr,datLen,waveLen,dx);
    CuTransFunc<<<grid,block>>>(d_transF,d_sqr,-1.0*zF,waveLen,datLen,dx);
    CuTransFunc<<<grid,block>>>(d_transInt,d_sqr,-1.0*dz,waveLen,datLen,dx);
    CuTransFunc<<<grid,block>>>(d_transPR,d_sqr,prDist,waveLen,datLen,dx);
    CuTransFunc<<<grid,block>>>(d_transPRInv,d_sqr,-1.0*prDist,waveLen,datLen,dx);
    std::cout << "PR Init OK" << std::endl;

    // Background Processing
    pCam[0]->BeginAcquisition();
    pCam[1]->BeginAcquisition();
    pCam[0]->TriggerSoftware.Execute();
    Spinnaker::ImagePtr pImg1 = pCam[0]->GetNextImage();
    Spinnaker::ImagePtr pImg2 = pCam[1]->GetNextImage();
    pCam[0]->EndAcquisition();
    pCam[1]->EndAcquisition();

    unsigned char *charimg1 = (unsigned char *)pImg1->GetData();
    unsigned char *charimg2 = (unsigned char *)pImg2->GetData();


    float *bImg1, *bImg2;
    bImg1 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    bImg2 = (float *)malloc(sizeof(float)*imgLen*imgLen);

    // for (int i = 0; i < imgLen*imgLen; i++)
    // {
    //     bImg1[i] = 0.0;
    //     bImg2[i] = 0.0;
    // }
    // float backMode1 = 55.0; 
    // float backMode2 = 55.0; 
    
    unsigned char *cBackImg1, *cBackImg2;
    cBackImg1 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    cBackImg2 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    Spinnaker::ImagePtr saveBack1 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,cBackImg1);
    Spinnaker::ImagePtr saveBack2 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,cBackImg2);
    getBackGrounds(bImg1,bImg2,cBackImg1,cBackImg2,pCam,imgLen,backgroundLoops);
    saveBack1->Convert(Spinnaker::PixelFormat_Mono8);
    saveBack2->Convert(Spinnaker::PixelFormat_Mono8);
    // saveBack1->Save("./meanBkg1.png");
    // saveBack2->Save("./meanBkg2.png");
    std::cout << "test" << std::endl;

    float backMode1 = getImgMode(cBackImg1,imgLen);
    float backMode2 = getImgMode(cBackImg2,imgLen);
    std::cout << "background mode 1: " << backMode1 << std::endl;
    std::cout << "background mode 2: " << backMode2 << std::endl;

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
    sprintf(dirPath,"./ImposedOutput/");
    mkdir(dirPath,0777);
    sprintf(dirPath,"./ImposedOutput/%02d%02d/",tm.tm_mon+1,tm.tm_mday);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./ImposedOutput/%02d%02d/%02d%02d%02d%02d",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./ImposedOutput/%02d%02d/%02d%02d%02d%02d/Gabor/",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./ImposedOutput/%02d%02d/%02d%02d%02d%02d/PR/",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./ImposedOutput/%02d%02d/%02d%02d%02d%02d/Holo/",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./ImposedOutput/%02d%02d/%02d%02d%02d%02d",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);

    char savePrPath[150];
    char saveGaborPath[150];
    char saveHoloPath[150];

    cv::Mat PrImp, GaborImp;
    cv::namedWindow("PR",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Gabor",cv::WINDOW_AUTOSIZE);
    cv::moveWindow("PR",520,0);
    cv::moveWindow("Gabor",0,0);

    cufftComplex *holoData;
    holoData = (cufftComplex *)malloc(sizeof(cufftComplex)*datLen*datLen);
    auto now = std::chrono::high_resolution_clock::now();

    int num = 0;
    while(!e_flag){
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - now);
        std::cout << duration.count() << std::endl;
        now = std::chrono::high_resolution_clock::now();

        std::cout << "Recording Holograms" << std::endl;
        pCam[0]->BeginAcquisition();
        pCam[1]->BeginAcquisition();
        pCam[0]->TriggerSoftware.Execute();
        pImg1 = pCam[0]->GetNextImage();
        pImg2 = pCam[1]->GetNextImage();
        pCam[0]->EndAcquisition();
        pCam[1]->EndAcquisition();

        charimg1 = (unsigned char *)pImg1->GetData();
        charimg2 = (unsigned char *)pImg2->GetData();

        std::cout << "Image Transforming" << std::endl;
        getNewImage(charimg2,charimg2,coefa,imgLen);

        std::cout << "PR Imposing" << std::endl;
        getPRImposed(outPrImp,outPrCImp,holoData,charimg1,charimg2,bImg1,bImg2,d_transF,d_transInt,d_transPR,d_transPRInv,backMode1,backMode2,imgLen,ImposedLoop,prLoop,blockSize);
        std::cout << "Gabor Imposing" << std::endl;
        getBackRemGaborImposed(outGaborImp,outGaborCImp,charimg2,bImg2,d_transF,d_transInt,backMode2,imgLen,ImposedLoop,blockSize);

        saveGaborImp->Convert(Spinnaker::PixelFormat_Mono8);
        savePrImp->Convert(Spinnaker::PixelFormat_Mono8);
        
        sprintf(savePrPath,"%s/PR/%05d.png",dirPath,num);
        sprintf(saveGaborPath,"%s/Gabor/%05d.png",dirPath,num);
        sprintf(saveHoloPath,"%s/Holo/%05d.dat",dirPath,num);

        saveGaborImp->Save(saveGaborPath);
        savePrImp->Save(savePrPath);

        saveCufftComplex(holoData,saveHoloPath,datLen);

        PrImp = cv::Mat(imgLen,imgLen,CV_8U,outPrCImp);
        GaborImp = cv::Mat(imgLen,imgLen,CV_8U,outGaborCImp);
        cv::imshow("PR",PrImp);
        cv::imshow("Gabor",GaborImp);
        cv::waitKey(10);

        num += 1;
    }
    
    // Finalize
    cudaFree(d_sqr);
    cudaFree(d_transF);
    cudaFree(d_transInt);
    cudaFree(d_transPR);
    cudaFree(d_transPRInv);
    free(bImg1);
    free(bImg2);
    // free(cBackImg1);
    // free(cBackImg2);
    free(outPrImp);
    free(outPrCImp);
    free(outGaborImp);
    free(outGaborCImp);
    free(holoData);

    camList.Clear();
    system->ReleaseInstance();

    cudaDeviceReset();
    return 0;
}