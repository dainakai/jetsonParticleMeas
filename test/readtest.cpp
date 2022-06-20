#include <stdio.h>
#include <stdlib.h>

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

int main(){
    char *path = "../coefa.dat";
    float a[12];
    readCoef(path,a);
    for (int i = 0; i < 12; i++)
    {
        printf("%lf\n",a[i]);
    }
    
}