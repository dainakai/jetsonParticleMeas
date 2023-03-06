# nvcc -O3 getPRImposed.cu -o getPRImposed -lcufft

# nvcc -Xcompiler "-std=c++11" -I/opt/spinnaker/include -I/opt/spinnaker/include -D LINUX -c -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -L/usr/lib -L/usr/local/lib -lfftw3f getPRImposedonCPU.cu -o .obj/build/getPRImposedonCPU.o

# nvcc -o getPRImposedonCPU .obj/build/getPRImposedonCPU.o -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -L/opt/spinnaker/lib -lSpinnaker -L/usr/lib -L/usr/local/lib -lfftw3f -Xlinker=-rpath,/opt/spinnaker/lib

# nvcc -Xcompiler "-std=c++11" -I/opt/spinnaker/include -I/opt/spinnaker/include -D LINUX -c -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lcufft -lopencv_highgui -Xcompiler="`pkg-config --cflags --libs gtk+-3.0`" getPRImposed.cu -o .obj/build/getPRImposed.o `pkg-config --libs gtk+-3.0`

# nvcc -o getPRImposed .obj/build/getPRImposed.o -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -L/opt/spinnaker/lib -lSpinnaker -lcufft -Xlinker=-rpath,/opt/spinnaker/lib -Xcompiler="`pkg-config --cflags --libs gtk+-3.0`" `pkg-config --libs gtk+-3.0`

g++ -I/opt/spinnaker/include -I/opt/spinnaker/include -D LINUX -c -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -L/usr/lib -L/usr/local/lib -lfftw3f getPRImposedonCPU.cpp -o .obj/build/getPRImposedonCPU.o

g++ -o getPRImposedonCPU .obj/build/getPRImposedonCPU.o -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -L/opt/spinnaker/lib -lSpinnaker -L/usr/lib -L/usr/local/lib -lfftw3f -L/opt/spinnaker/lib
