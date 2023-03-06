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
    const int backgroundLoops = 100;
    const int imgLen = 512;
    const int blockSize = 16;
    const int loopCount = 200;
    const float waveLen = 0.532;
    const float zF = 10.0*1000.0;
    const float dz = 200.0;
    const float dx = 3.45/0.5;

    const int OffsetX = atoi(argv[1]);
    // const int OffsetX = 592;
    const int OffsetY = atoi(argv[2]);
    // const int OffsetY = 514;

    // Constant Declaration
    const int datLen = imgLen*2;
    dim3 grid((int)ceil((float)datLen/(float)blockSize),(int)ceil((float)datLen/(float)blockSize)), block(blockSize,blockSize);

    // Propagation Init
    float *d_sqr;
    cufftComplex *d_transF, *d_transInt;
    CHECK(cudaMalloc((void **)&d_sqr, sizeof(float)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transF, sizeof(cufftComplex)*datLen*datLen));
    CHECK(cudaMalloc((void **)&d_transInt, sizeof(cufftComplex)*datLen*datLen));
    CuTransSqr<<<grid,block>>>(d_sqr,datLen,waveLen,dx);
    CuTransFunc<<<grid,block>>>(d_transF,d_sqr,-1.0*zF,waveLen,datLen,dx);
    CuTransFunc<<<grid,block>>>(d_transInt,d_sqr,-1.0*dz,waveLen,datLen,dx);
    std::cout << "PR Init OK" << std::endl;


    float gain1,gain2;
    std::tie(gain1,gain2) = readGain("./gain.dat");
    
    // const int imgLen = 512;

    // Camera Init
    Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
    Spinnaker::CameraList camList = system->GetCameras();
    unsigned int numCameras = camList.GetSize();
    if (numCameras==0){
        std::cout << "No Cameras are Connected! Quitting..." << std::endl;
        camList.Clear();
        system->ReleaseInstance();
        exit(1);
    }else if(numCameras==2){
        std::cout << "Two Cameras are Connected! Requiring only SINGLE camera. Quitting..." << std::endl;
        camList.Clear();
        system->ReleaseInstance();
        exit(1);
    }
    Spinnaker::CameraPtr pCam;
    std::cout << "Camera" << "\t" << "ModelName" << "\t\t\t" << "SerialNumber" << std::endl;
    pCam = camList.GetByIndex(0);
    pCam->Init();
    Spinnaker::GenICam::gcstring modelName = pCam->TLDevice.DeviceModelName.GetValue();
    Spinnaker::GenICam::gcstring serialNum = pCam->TLDevice.DeviceSerialNumber.GetValue();
    std::cout << 1 << "\t" << modelName << "\t" << serialNum << std::endl;

    std::cout << "Camera Enum OK" << std::endl;

    // Camera Setup
    singleCameraSetup(pCam,imgLen,OffsetX,OffsetY,camExposure,gain1);

    // Processing
    pCam->BeginAcquisition();
    pCam->TriggerSoftware.Execute();
    Spinnaker::ImagePtr pImg = pCam->GetNextImage();
    pCam->EndAcquisition();

    unsigned char *charimg = (unsigned char *)pImg->GetData();

    // Start Processing
    float coefa[12];
    char *coefPath = "./coefa.dat";
    readCoef(coefPath,coefa);
    // unsigned char *saveholo1, *saveholo2;
    // saveholo1 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    // saveholo2 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    Spinnaker::ImagePtr holoptr = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,charimg);

    float *bImg;
    bImg = (float *)malloc(sizeof(float)*imgLen*imgLen);
    
    unsigned char *cBackImg;
    cBackImg = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    Spinnaker::ImagePtr saveBack = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,cBackImg);
    getSingleBackGrounds(bImg,cBackImg,pCam,imgLen,backgroundLoops);
    std::cout << "test" << std::endl;
    saveBack->Convert(Spinnaker::PixelFormat_Mono8);

    float backMode = getImgMode(cBackImg,imgLen);

    struct tm tm;
    time_t t = time(NULL);
    localtime_r(&t,&tm);
    char dirPath[100];
    sprintf(dirPath,"./holoOutput/");
    mkdir(dirPath,0777);
    sprintf(dirPath,"./holoOutput/%02d%02d/",tm.tm_mon+1,tm.tm_mday);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./holoOutput/%02d%02d/%02d%02d%02d%02d",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./holoOutput/%02d%02d/%02d%02d%02d%02d/cam1/",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./holoOutput/%02d%02d/%02d%02d%02d%02d/slices/",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./holoOutput/%02d%02d/%02d%02d%02d%02d",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);

    char savecamPath[150];

    cv::Mat cvMatcam;
    cv::namedWindow("cam",cv::WINDOW_AUTOSIZE);
    cv::moveWindow("cam",0,0);

    auto now = std::chrono::high_resolution_clock::now();

    int num = 0;
    while(!e_flag){
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - now);
        std::cout << duration.count() << std::endl;
        now = std::chrono::high_resolution_clock::now();

        std::cout << "Recording Holograms" << std::endl;
        pCam->BeginAcquisition();
        pCam->TriggerSoftware.Execute();
        pImg = pCam->GetNextImage();
        pCam->EndAcquisition();

        charimg = (unsigned char *)pImg->GetData();

        std::cout << "Image Transforming" << std::endl;
        getNewImage(charimg,charimg,coefa,imgLen);

        holoptr->Convert(Spinnaker::PixelFormat_Mono8);
        
        sprintf(savecamPath,"%s/cam1/%05d.png",dirPath,num);

        holoptr->Save(savecamPath);

        sprintf(dirPath,"./holoOutput/%02d%02d/%02d%02d%02d%02d/slices/%05d",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min,num);
        mkdir(dirPath,0777);

        getGaborReconstSlices(charimg,bImg,backMode, dirPath, d_transF, d_transInt, imgLen, loopCount, blockSize);

        sprintf(dirPath,"./holoOutput/%02d%02d/%02d%02d%02d%02d",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);

        cvMatcam = cv::Mat(imgLen,imgLen,CV_8U,charimg);
        cv::imshow("cam",cvMatcam);
        cv::waitKey(10);

        num += 1;
    }

    // Reset camera user sets and deinitialize all cameras
    if (pCam.IsValid())
    {
        std::cout << "Resetting configuration for device " << pCam->TLDevice.DeviceSerialNumber.GetValue() << std::endl;
        pCam->DeInit();
        pCam = nullptr;
    }
    camList.Clear();
    system->ReleaseInstance();

    return 0;
}