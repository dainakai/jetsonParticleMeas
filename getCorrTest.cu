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

int main(int argc, char** argv){
    std::cout << argv[0] << " Starting..." << std::endl;

    float *img1;
    img1 = (float *)malloc(sizeof(float)*1024*1024);
    getFloatimage(img1,1024,"./cam1.bmp");

    std::cout << *img1 << std::endl;


    return 0;
}