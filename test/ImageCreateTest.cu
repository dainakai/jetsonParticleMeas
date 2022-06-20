#include "../include/Spinnaker.h"
#include "../include/SpinGenApi/SpinnakerGenApi.h"
#include <stdio.h>

void getImage(unsigned char *img, const int imgLen, const char* path){
    FILE *fp;
    fp = fopen(path,"rb");
    if(fp == NULL){
        printf("%s seems not to exist! Quitting...\n",path);
        exit(1);
    }
    fseek(fp,1078,0);
    fread(img,sizeof(unsigned char),imgLen*imgLen,fp);
    fclose(fp);
}

int main(){
    const char *path1 = "./cam1.bmp";
    unsigned char *img;
    const int imgLen = 1024;
    img = (unsigned char *)malloc(sizeof(unsigned char)*imgLen*imgLen);
    Spinnaker::ImagePtr saveImg = Spinnaker::Image::Create(imgLen,imgLen,0,0,Spinnaker::PixelFormatEnums::PixelFormat_Mono8,img);
    getImage(img,imgLen,path1);
    saveImg->Convert(Spinnaker::PixelFormat_Mono8);
    saveImg->Save("YReverseCam1.bmp");

    return 0;
}