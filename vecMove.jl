using Dates

function main()
    d = monthday(today())
    dstring = lpad(d[1],2,"0")*lpad(d[2],2,"0")
    h = hour(now())
    m = minute(now())
    tstring = lpad(h,2,"0")*lpad(m,2,"0")
    mkpath("./vectorResult/"*dstring*"/"*dstring*tstring*"/")
    cp("./vecArray.dat","./vectorResult/"*dstring*"/"*dstring*tstring*"/vecArray.dat")
    cp("./vecArrayPlot.pdf","./vectorResult/"*dstring*"/"*dstring*tstring*"/vecArrayPlot.pdf")
end

main()