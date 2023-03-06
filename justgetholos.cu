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
    pCam[0]->BeginAcquisition();
    pCam[1]->BeginAcquisition();
    pCam[0]->TriggerSoftware.Execute();
    Spinnaker::ImagePtr pImg1 = pCam[0]->GetNextImage();
    Spinnaker::ImagePtr pImg2 = pCam[1]->GetNextImage();
    pCam[0]->EndAcquisition();
    pCam[1]->EndAcquisition();

    unsigned char *charimg1 = (unsigned char *)pImg1->GetData();
    unsigned char *charimg2 = (unsigned char *)pImg2->GetData(); 

    // Start Processing
    float coefa[12];
    char *coefPath = "./coefa.dat";
    readCoef(coefPath,coefa);
    // unsigned char *saveholo1, *saveholo2;
    // saveholo1 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    // saveholo2 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    Spinnaker::ImagePtr holoptr1 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,charimg1);
    Spinnaker::ImagePtr holoptr2 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,charimg2);

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
    sprintf(dirPath,"./holoOutput/%02d%02d/%02d%02d%02d%02d/cam2/",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
    mkdir(dirPath,0777);
    sprintf(dirPath,"./holoOutput/%02d%02d/%02d%02d%02d%02d",tm.tm_mon+1,tm.tm_mday,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);

    char savecam1Path[150];
    char savecam2Path[150];

    cv::Mat cvMatcam1, cvMatcam2;
    cv::namedWindow("cam1",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("cam2",cv::WINDOW_AUTOSIZE);
    cv::moveWindow("cam1",520,0);
    cv::moveWindow("cam2",0,0);

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

        holoptr1->Convert(Spinnaker::PixelFormat_Mono8);
        holoptr2->Convert(Spinnaker::PixelFormat_Mono8);
        
        sprintf(savecam1Path,"%s/cam1/%05d.png",dirPath,num);
        sprintf(savecam2Path,"%s/cam2/%05d.png",dirPath,num);

        holoptr1->Save(savecam1Path);
        holoptr2->Save(savecam2Path);

        cvMatcam1 = cv::Mat(imgLen,imgLen,CV_8U,charimg1);
        cvMatcam2 = cv::Mat(imgLen,imgLen,CV_8U,charimg2);
        cv::imshow("cam1",cvMatcam1);
        cv::imshow("cam2",cvMatcam2);
        cv::waitKey(10);

        num += 1;
    }

    camList.Clear();
    system->ReleaseInstance();

    return 0;
}