using Dates

function main()
    d = monthday(today())
    dstring = lpad(d[1],2,"0")*lpad(d[2],2,"0")
    cp("./vecArray.dat","./vectorResult/"*dstring*"/coefa.dat")
end

main()