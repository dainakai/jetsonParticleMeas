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

int main(int argc, char** argv){
    std::cout << argv[0] << " Starting..." << std::endl;
    
    // Parameters
    const float camExposure = 100.0;
    // const float gainInit = 1.0;

    float gain1,gain2;
    std::tie(gain1,gain2) = readGain("./gain.dat");

    const int OffsetX = atoi(argv[1]);
    // const int OffsetX = 584;
    const int OffsetY = atoi(argv[2]);
    // const int OffsetY = 506;
    
    const int imgLen = 512;
    const int intrSize = imgLen/8;
    const int gridSize = imgLen/8;
    const int srchSize = imgLen/4;
    const int gridNum = (int)(imgLen/gridSize);

    const float zFront = 1000*60.0;
    const float dz = 50.0;
    const float wavLen = 0.532;
    const float dx = 3.45/0.5;

    const int blockSize = 16; 

    const float targetMeanBright = 0.35;

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

    float mean1, mean2;
    gain1 = pCam[0]->Gain.GetValue();
    gain2 = pCam[1]->Gain.GetValue();
    // Processing
    while(1){
        std::tie(mean1,mean2) = getCamMean(pCam,imgLen);
        std::cout << "Cam1 mean: " << mean1 << std::endl; 
        std::cout << "Cam2 mean: " << mean2 << std::endl; 
        if (abs(mean2-targetMeanBright) <= 0.01){
            break;
        }else{
            gain2 += -(mean2-targetMeanBright);
            pCam[1]->Gain.SetValue((double)gain2);
        }

    }
    pCam[0]->Gain.SetValue((double)gain2);
    gain1 = pCam[0]->Gain.GetValue();
    gain2 = pCam[1]->Gain.GetValue();
    std::cout << "Cam1 Gain:" << gain1 << std::endl;
    std::cout << "Cam2 Gain:" << gain2 << std::endl;
    std::cout << "Gain ratio: " << gain2/gain1 << std::endl;

    FILE *fp;
    fp = fopen("gain.dat","w");
    fprintf(fp,"%lf\n%lf\n",gain1,gain2);
    fclose(fp);

    camList.Clear();
    system->ReleaseInstance();

    cudaDeviceReset();
    return 0;
}