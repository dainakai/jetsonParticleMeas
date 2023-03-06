# nvcc -O3 ImageCreateTest.cu -o ImageCreateTest -lcufft

nvcc -Xcompiler "-std=c++11" -I/opt/spinnaker/include -I/opt/spinnaker/include -D LINUX -c -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lcufft -lopencv_highgui ImageCreateTest.cu -o .obj/build/ImageCreateTest.o

nvcc -o ImageCreateTest .obj/build/ImageCreateTest.o -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -L/opt/spinnaker/lib -lSpinnaker -lcufft -Xlinker=-rpath,/opt/spinnaker/lib

# nvcc -Xcompiler "-std=c++11" -I/opt/spinnaker/include -I/opt/spinnaker/include -D LINUX -c -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lcufft -lopencv_highgui -Xcompiler="`pkg-config --cflags --libs gtk+-3.0`" ImageCreateTest.cu -o .obj/build/ImageCreateTest.o `pkg-config --libs gtk+-3.0`

# nvcc -o ImageCreateTest .obj/build/ImageCreateTest.o -I/usr/local/include/opencv4/ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -L/opt/spinnaker/lib -lSpinnaker -lcufft -Xlinker=-rpath,/opt/spinnaker/lib -Xcompiler="`pkg-config --cflags --libs gtk+-3.0`" `pkg-config --libs gtk+-3.0`