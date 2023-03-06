using ImageView
using Images
using CUDA
using CUDA.CUFFT
using Random
using Plots

function CuObjPlane(rad, x0, y0, dx, datLen, Plane)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        if (x0-Float32(x))^2 + (y0-Float32(y))^2 < (rad/dx)^2
            Plane[x,y] = 0
        else
            Plane[x,y] = 1
        end
    end
    return
end

function CuTransSqr(datLen, wavLen, dx, Plane)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        Plane[x,y] = 1.0 - ((x-datLen/2)*wavLen/datLen/dx)^2 - ((y-datLen/2)*wavLen/datLen/dx)^2
    end
    return
end

function CuTransFunc(z0, wavLen, datLen, d_sqrPart, Plane)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        Plane[x,y] = exp(2im*pi*z0/wavLen*sqrt(d_sqrPart[x,y]))
    end
    return
end

function main()
    dx = 6.9
    z0 = (120.0)*1000.0
    datLen = 1024
    wavLen = 0.532
    threads = (32,32)
    blocks = (cld(1024,32),cld(1024,32))

    meanDiam = 50.0
    sdDiam = 10.0
    # rad = sdDiam*randn() + meanDiam
    rad = 35.0
    print(rad)

    obj = CuArray{Float32}(undef,(datLen,datLen))
    sqr = CuArray{Float32}(undef,(datLen,datLen))
    trans = CuArray{ComplexF32}(undef,(datLen,datLen))
    holo = CuArray{ComplexF32}(undef,(datLen,datLen))
    @cuda threads = threads blocks = blocks CuObjPlane(rad,512,512,dx,datLen,obj)
    @cuda threads = threads blocks = blocks CuTransSqr(datLen,wavLen,dx,sqr)
    @cuda threads = threads blocks = blocks CuTransFunc(z0,wavLen,datLen,sqr,trans)
    holo = fftshift(fft(obj)) .* trans
    holo = ifft(fftshift(holo))
    img = Array(real.(holo .* conj.(holo)))
    save("./test120.png",float.(img)./255.0.*128.0)
    # imshow(float.(img)/2)
    # graph = img[512,512:1024]
    # plot(graph,title="rad = $rad", xlims=(0,200),ylims=(0.5,1.5))
end

@time main()