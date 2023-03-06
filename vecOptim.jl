using Glob
using DelimitedFiles
using StatsBase
using Dates

function main()
    d = monthday(today())
    dstring = lpad(d[1],2,"0")*lpad(d[2],2,"0")
    vecPath = "./vectorResult/"*dstring*"/*/*.dat"
    vecArrPath = glob(vecPath)
    vecArray = readdlm(vecArrPath[1])
    meanVal = mean(vecArray[:,3].^2+vecArray[:,4].^2)
    for i in 2:length(vecArrPath)
        tmp = readdlm(vecArrPath[i])
        for idx in 1:7^2
            if tmp[idx,3]^2+tmp[idx,4]^2 < vecArray[idx,3]^2+vecArray[idx,4]^2 && (vecArray[idx,3]^2+vecArray[idx,4]^2)/meanVal > 3.0
                vecArray[idx,3] = tmp[idx,3]
                vecArray[idx,4] = tmp[idx,4]
            end
        end
    end
    display(vecArray)
    println("")
    writedlm("./vecArray.dat",vecArray)
end

main()