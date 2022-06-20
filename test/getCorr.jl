"""
ベクトル場datを読み込んでcorr_aを計算
"""

using StatsBase
using DelimitedFiles

"""
修正コレスキー分解

対象行列Aを A = LDL^T に変換します。

戻り値行列の対角成分は D の逆数、左下成分は L に一致します。
"""
function choleskyDecom(Matrix)
    ndims(Matrix) == 2 ? 1 : error("choleskyDecom : Input Matrix is not in 2 dims.")
    n = size(Matrix)[1]
    n == size(Matrix)[2] ? 1 : error("choleskyDecom : Input Matrix is not square.")
    # println("Input : $(n)*$(n) matrix")
    matA = copy(Matrix)
    vecw = Array{Float64}(undef,n)
    for j in 1:n
        for i in 1:j-1
            vecw[i] = matA[j,i]
            for k in 1:i-1
                vecw[i] -= matA[i,k] * vecw[k]
            end
            matA[j,i] = vecw[i] * matA[i,i]
        end
        t = matA[j,j]
        for k in 1:j-1
            t -= matA[j,k] * vecw[k]
        end
        matA[j,j] = 1.0/t
    end
    # matA の対角成分は matD の逆数。matA_ji (i<j) は matL_ji。
    return matA
end

"""
連立方程式ソルバー

修正コレスキー分解で取得した行列 `chlskyMat` およびヤコビアン `yacob` 、エラーベクトル `errorArray`から

`chlskyMat` x = - `yacob` `errorArray`

を解き、戻り値として返します。
"""
function simEqSolver(chlskyMat,yacob,errorArray)
    vecB = -transpose(yacob)*errorArray
    n = size(vecB)[1]

    matL = chlskyMat
    

    vecX = zeros(n)
    vecY = zeros(n)

    for k in 1:n
        vecY[k] = vecB[k]
        for i in 1:k-1
            vecY[k] -= matL[k,i]*vecY[i]
        end
    end
    for mink in 1:n
        k = n+1-mink
        vecX[k] = vecY[k]*matL[k,k]
        for i in k+1:n
            vecX[k] -= matL[i,k]*vecX[i]
        end
    end
    return vecX
end

"""
ヤコビアン計算

パラメータ a で与えた二次の画像変換について、ターゲット座標との差のベクトル e の a に関するヤコビアンを計算し、戻り値として返します。
"""
function getYacobian(imgSize, gridSize)
    # x = collect(gridSize:gridSize:imgSize-1)
    x = collect(gridSize+0.5:gridSize:imgSize)
    # y = collect(gridSize:gridSize:imgSize-1)
    y = collect(gridSize+0.5:gridSize:imgSize)
    n = size(x)[1]
    na = 12

    yacob = Array{Float64}(undef,2*n*n,na)
    for j in 1:n
        for i in 1:n
            idx = i + (j-1)*n
            yacob[2idx-1,1] = -1.0    
            yacob[2idx,1] = 0.0

            yacob[2idx-1,2] = -x[i]    
            yacob[2idx,2] = 0.0
        
            yacob[2idx-1,3] = -y[j]    
            yacob[2idx,3] = 0.0

            yacob[2idx-1,4] = -x[i]^2    
            yacob[2idx,4] = 0.0

            yacob[2idx-1,5] = -x[i]*y[j]    
            yacob[2idx,5] = 0.0

            yacob[2idx-1,6] = -y[j]^2    
            yacob[2idx,6] = 0.0

            yacob[2idx-1,7] = 0.0
            yacob[2idx,7] = -1.0
        
            yacob[2idx-1,8] = 0.0
            yacob[2idx,8] = -x[i]
        
            yacob[2idx-1,9] = 0.0
            yacob[2idx,9] = -y[j]
        
            yacob[2idx-1,10] = 0.0
            yacob[2idx,10] = -x[i]^2
        
            yacob[2idx-1,11] = 0.0
            yacob[2idx,11] = -x[i]*y[j]
        
            yacob[2idx-1,12] = 0.0
            yacob[2idx,12] = -y[j]^2
        end
    end
    return yacob
end

"""
エラーベクトルの計算

1枚目の画像で設定する `gridx` `gridy` から2枚目の画像の `targetX` `targetY` を目的に変換して得た `procX` `procY` を計算し、`targetX` `targetY` と `procX` `procY` の差を計算したベクトル `errorVec` を返します。ヤコビアン取得のため `gridx` `gridy` も同時に返します。
"""
function getErrorVec(vecMap, coefa, imgSize)
    gridSize = div(imgSize,8)
    n = 7
    # gridx = collect(gridSize:gridSize:imgSize-1)
    gridx = collect(gridSize+0.5:gridSize:imgSize)
    # gridy = collect(gridSize:gridSize:imgSize-1)
    gridy = collect(gridSize+0.5:gridSize:imgSize)
    targetX = Array{Float64}(undef,n*n)
    targetY = Array{Float64}(undef,n*n)
    procX = Array{Float64}(undef,n*n)
    procY = Array{Float64}(undef,n*n)

    for y in 1:n
        for x in 1:n
            targetX[x + n*(y-1)] = gridx[x] + vecMap[y,x,1]
            targetY[x + n*(y-1)] = gridy[y] + vecMap[y,x,2]
            procX[x + n*(y-1)] = coefa[1] + coefa[2]*gridx[x] + coefa[3]*gridy[y] + coefa[4]*gridx[x]^2 + coefa[5]*gridx[x]*gridy[y] + coefa[6]*gridy[y]^2
            procY[x + n*(y-1)] = coefa[7] + coefa[8]*gridx[x] + coefa[9]*gridy[y] + coefa[10]*gridx[x]^2 + coefa[11]*gridx[x]*gridy[y] + coefa[12]*gridy[y]^2
        end
    end

    errorVec = Array{Float64}(undef,2*n*n)

    for idx in 1:n*n
        errorVec[2*idx-1] = targetX[idx] - procX[idx]
        errorVec[2*idx] = targetY[idx] - procY[idx]
    end

    return errorVec
end

function main()
    n = 7
    imgLen = 1024
    gridSize = div(imgLen,8)
    array = readdlm("./vecArray.dat")
    vecArray = Array{Float32}(undef,n,n,2)
    for j in 1:n
        for i in 1:n
            vecArray[i,j,1] = array[(i-1)*7+j,3]
            vecArray[i,j,2] = array[(i-1)*7+j,4]
        end
    end

    coefa = fill(1.0,12)
    itr = 1
    yacobian = getYacobian(imgLen,gridSize)
    hMat = transpose(yacobian)*yacobian

    while itr <= 10
        println("Iteration: ",itr)
        errorVec= getErrorVec(vecArray,coefa,imgLen)
        println("Error vec mean norm: ",sqrt(mean(errorVec.*errorVec)))
        println()
        deltaCoefa = simEqSolver(choleskyDecom(hMat),yacobian,errorVec)
        coefa += deltaCoefa
        itr += 1
    end
    display(coefa)
    println("")
    writedlm("./coefa.dat",coefa)
end

main()