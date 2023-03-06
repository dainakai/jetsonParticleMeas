#include "CUDAfunctions.cuh"
// #include "hostfunctions.h"
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

int main(int argc, char** argv){    
    std::cout << argv[0] << " Starting..." << std::endl;
    
    // Parameters
    const float camExposure = 129.0;
    const float gainInit = 0.0;

    const int OffsetX = atoi(argv[1]);
    // const int OffsetX = 592;
    const int OffsetY = atoi(argv[2]);
    // const int OffsetY = 510;

    float gain1,gain2;
    std::tie(gain1,gain2) = readGain("./gain.dat");
    
    const int imgLen = 512;
    const int intrSize = imgLen/8;
    const int gridSize = imgLen/8;
    const int srchSize = imgLen/4;
    const int gridNum = (int)(imgLen/gridSize);

    const int backgroundLoops = 30;
    const int loopCount = 100;

    const float zFront = 1000.0*10.0;
    const float dz = 10.0;
    const float wavLen = 0.532;
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

    cameraSetup(pCam,imgLen,OffsetX,OffsetY,camExposure,gain1,gain2);

    // Background Processing
    // pCam[0]->BeginAcquisition();
    // pCam[1]->BeginAcquisition();
    // pCam[0]->TriggerSoftware.Execute();
    // Spinnaker::ImagePtr pImg1 = pCam[0]->GetNextImage();
    // Spinnaker::ImagePtr pImg2 = pCam[1]->GetNextImage();
    // pCam[0]->EndAcquisition();
    // pCam[1]->EndAcquisition();

    // char16_t *charimg1 = (char16_t *)pImg1->GetData();
    // char16_t *charimg2 = (char16_t *)pImg2->GetData();


    // float *bImg1, *bImg2;
    // bImg1 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    // bImg2 = (float *)malloc(sizeof(float)*imgLen*imgLen);
    // unsigned char *cBackImg1, *cBackImg2;
    // cBackImg1 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    // cBackImg2 = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    // Spinnaker::ImagePtr saveBack1 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,cBackImg1);
    // Spinnaker::ImagePtr saveBack2 = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,cBackImg2);
    // getBackGrounds(bImg1,bImg2,cBackImg1,cBackImg2,pCam,imgLen,backgroundLoops);
    // saveBack1->Convert(Spinnaker::PixelFormat_Mono8);
    // saveBack2->Convert(Spinnaker::PixelFormat_Mono8);
    // saveBack1->Save("./meanBkg1.png");
    // saveBack2->Save("./meanBkg2.png");
    // std::cout << "test" << std::endl;

    // // Process Bridge
    // std::cout << "Background Acquisition Completed. PR Reconstruction will be started in..." << std::endl;
    // for (int i = 0; i <= 5; i++){
    //     int sec = 5-i;
    //     sleep(1);
    //     std::cout << sec << std::endl;
    // }


    getImgAndBundleAdjCheck(pCam,imgLen,gridSize,intrSize,srchSize,zFront,dz,wavLen,dx,blockSize);

    
    pCam[0]->DeInit();
    pCam[1]->DeInit();
    system->ReleaseInstance();

    cudaDeviceReset();
    return 0;
}