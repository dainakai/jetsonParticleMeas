/**
 * @file hostfunctions.h
 * @brief Jetson上の位相回復ホログラフィによる流刑分布取得実験用関数群
 * @author Dai Nakai
 * @date May, 2022.
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
// #include <gtk/gtk.h>
#include <string>

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
    fprintf(gp,"set yrange[%d:0]\n",imgLen);
	fprintf(gp,"set palette rgb 33,13,10\n");

  // fprintf(gp,"set yrange reverse\n");

	fprintf(gp,"set xlabel '%s'offset 0.0,0.5\n",xxlabel);
	fprintf(gp,"set ylabel '%s'offset 0.5,0.0\n",yylabel);

	fprintf(gp,"plot '%s' using 1:2:(%d*$3):(%d*$4):(sqrt($3*$3+$4*$4))  w vector lc palette ti ''\n",vecArrayDataPath,vecLenSclr,vecLenSclr);

 	fflush(gp); //Clean up Data

	fprintf(gp, "exit\n"); // Quit gnuplot
	pclose(gp);
}

/**
 * @fn
 * @brief 2カメラの設定
 * @param cam2OffSetX カメラ2のオフセット．事前にSpinViewで調整
 * @return なし
 */
void cameraSetup(Spinnaker::CameraPtr pCam[2], int imgLen, int cam2OffSetX, int cam2OffSetY, float exposure, float gain1, float gain2){

    // Settings common to all cameras
    for (int i = 0; i < 2; i++){
        pCam[i]->GammaEnable.SetValue(false);
        pCam[i]->AdcBitDepth.SetValue(Spinnaker::AdcBitDepth_Bit12);
        pCam[i]->AcquisitionFrameRateEnable.SetValue(false);
        pCam[i]->PixelFormat.SetValue(Spinnaker::PixelFormat_Mono8);
        // pCam[i]->Width.SetValue(imgLen);
        // pCam[i]->Height.SetValue(imgLen);
        pCam[i]->ExposureAuto.SetValue(Spinnaker::ExposureAutoEnums::ExposureAuto_Off);
        pCam[i]->ExposureMode.SetValue(Spinnaker::ExposureModeEnums::ExposureMode_Timed);
        pCam[i]->ExposureTime.SetValue(exposure-33.0*i);
        pCam[i]->GainAuto.SetValue(Spinnaker::GainAutoEnums::GainAuto_Off);
        // pCam[i]->Gain.SetValue(gain);
        pCam[i]->AcquisitionMode.SetValue(Spinnaker::AcquisitionModeEnums::AcquisitionMode_Continuous);
    }



    // Settings for Camera 1
    pCam[0]->Width.SetValue(imgLen);
    pCam[0]->Height.SetValue(imgLen);
    pCam[0]->Gain.SetValue(gain1);
    pCam[0]->OffsetX.SetValue((int)((2048-imgLen)/2));
    pCam[0]->OffsetY.SetValue((int)((1536-imgLen)/2));
    pCam[0]->ReverseX.SetValue(true);
    pCam[0]->ReverseY.SetValue(false);
    pCam[0]->TriggerMode.SetValue(Spinnaker::TriggerModeEnums::TriggerMode_On);
    pCam[0]->TriggerSource.SetValue(Spinnaker::TriggerSourceEnums::TriggerSource_Software);
    pCam[0]->TriggerSelector.SetValue(Spinnaker::TriggerSelectorEnums::TriggerSelector_FrameStart);
    pCam[0]->LineSelector.SetValue(Spinnaker::LineSelectorEnums::LineSelector_Line1);
    pCam[0]->LineMode.SetValue(Spinnaker::LineModeEnums::LineMode_Output);
    pCam[0]->LineSelector.SetValue(Spinnaker::LineSelectorEnums::LineSelector_Line2);
    pCam[0]->V3_3Enable.SetValue(true);
    pCam[0]->TriggerOverlap.SetValue(Spinnaker::TriggerOverlapEnums::TriggerOverlap_Off);

    // Settings for Camera 2
    pCam[1]->Width.SetValue(imgLen);
    pCam[1]->Height.SetValue(imgLen);
    pCam[1]->OffsetX.SetValue(cam2OffSetX);
    pCam[1]->OffsetY.SetValue(cam2OffSetY);
    pCam[1]->Gain.SetValue(gain2);
    pCam[1]->ReverseX.SetValue(false);
    pCam[1]->ReverseY.SetValue(false);
    pCam[1]->TriggerMode.SetValue(Spinnaker::TriggerModeEnums::TriggerMode_On);
    pCam[1]->TriggerSource.SetValue(Spinnaker::TriggerSourceEnums::TriggerSource_Line3);
    pCam[1]->TriggerSelector.SetValue(Spinnaker::TriggerSelectorEnums::TriggerSelector_FrameStart);
    pCam[1]->TriggerOverlap.SetValue(Spinnaker::TriggerOverlapEnums::TriggerOverlap_ReadOut);

    printf("Camera Setup Completed.\n\n");
}

/**
 * @fn
 * @brief 1カメラ(inline)の設定
 * @param camOffSetX カメラのオフセット．事前にSpinViewで調整
 * @return なし
 */
void singleCameraSetup(Spinnaker::CameraPtr pCam, int imgLen, int camOffSetX, int camOffSetY, float exposure, float gain){
    pCam->GammaEnable.SetValue(false);
    pCam->AdcBitDepth.SetValue(Spinnaker::AdcBitDepth_Bit12);
    pCam->AcquisitionFrameRateEnable.SetValue(false);
    pCam->PixelFormat.SetValue(Spinnaker::PixelFormat_Mono8);
    pCam->ExposureAuto.SetValue(Spinnaker::ExposureAutoEnums::ExposureAuto_Off);
    pCam->ExposureMode.SetValue(Spinnaker::ExposureModeEnums::ExposureMode_Timed);
    pCam->ExposureTime.SetValue(exposure);
    pCam->GainAuto.SetValue(Spinnaker::GainAutoEnums::GainAuto_Off);
    pCam->AcquisitionMode.SetValue(Spinnaker::AcquisitionModeEnums::AcquisitionMode_Continuous);

    // Settings for Camera 1
    pCam->Width.SetValue(imgLen);
    pCam->Height.SetValue(imgLen);
    pCam->OffsetX.SetValue(camOffSetX);
    pCam->OffsetY.SetValue(camOffSetY);
    pCam->Gain.SetValue(gain);
    pCam->ReverseX.SetValue(false);
    pCam->ReverseY.SetValue(false);
    pCam->TriggerMode.SetValue(Spinnaker::TriggerModeEnums::TriggerMode_On);
    pCam->TriggerSource.SetValue(Spinnaker::TriggerSourceEnums::TriggerSource_Software);
    pCam->TriggerSelector.SetValue(Spinnaker::TriggerSelectorEnums::TriggerSelector_FrameStart);
    pCam->LineSelector.SetValue(Spinnaker::LineSelectorEnums::LineSelector_Line1);
    pCam->LineMode.SetValue(Spinnaker::LineModeEnums::LineMode_Output);
    pCam->LineSelector.SetValue(Spinnaker::LineSelectorEnums::LineSelector_Line2);
    pCam->V3_3Enable.SetValue(true);
    pCam->TriggerOverlap.SetValue(Spinnaker::TriggerOverlapEnums::TriggerOverlap_Off);

    printf("Camera Setup Completed.\n\n");
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

std::tuple<float, float> readGain(char *path){
    float gain1,gain2;
    FILE *fp;
    fp = fopen(path,"r");
    if(fp == NULL){
        printf("%s seems not to exist! Quitting...\n",path);
        exit(1);
    }
    char tmp1[30], tmp2[30];
    fgets(tmp1,100,fp);
    fgets(tmp2,100,fp);
    gain1 = atof(tmp1);
    gain2 = atof(tmp2);
    fclose(fp);

    return std::forward_as_tuple(gain1,gain2);
}

std::tuple<float,float> getCamMean(Spinnaker::CameraPtr pCam[2],const int imgLen){
    // Camera Init
    Spinnaker::CameraPtr cam1 = pCam[0];
    Spinnaker::CameraPtr cam2 = pCam[1];
    cam1->BeginAcquisition();
    cam2->BeginAcquisition();
    cam1->TriggerSoftware.Execute();
    Spinnaker::ImagePtr pimg1 = cam1->GetNextImage();
    Spinnaker::ImagePtr pimg2 = cam2->GetNextImage();
    cam1->EndAcquisition();
    cam2->EndAcquisition();
    unsigned char *charimg1 = (unsigned char *)pimg1->GetData();
    unsigned char *charimg2 = (unsigned char *)pimg2->GetData();

    // Get Mean
    float mean1 = 0.0;
    float mean2 = 0.0;
    for (int i = 0; i < imgLen*imgLen; i++){
        mean1 += (float)((int)charimg1[i])/255.0;
        mean2 += (float)((int)charimg2[i])/255.0;
    }
    mean1 /= (float)(imgLen*imgLen);
    mean2 /= (float)(imgLen*imgLen);

    return std::forward_as_tuple(mean1,mean2);
}