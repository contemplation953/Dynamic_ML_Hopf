using Parsers
using Plots

global y,i
y = []
i = 0
open("./num/outputdata/log.log") do file
# Read the contents of a file line by line
    for ln in eachline(file)
        global i,y
        i = i + 1
        s = split(ln,",")
        val = Parsers.parse(Float32, s[2])
        #Remove large errors caused by parameters that are out of bounds or do not satisfy monotonicity
        if i>7000 && val > 200
            continue
        end
        y = vcat(y,val)
    end
end

step1 = 1:7000
step2 = 7001:length(y)-3000
step3 = length(y)-3000+1:length(y)
plot(step1, y[step1], color=:red, linewidth=1.5, label="step1",xlabel="setp",ylabel = "error")
plot!(step2, y[step2], color=:blue, linewidth=1.5, label="step2")
plot!(step3, y[step3], color=:black, linewidth=1.5, label="step3")
savefig("./num/outputdata/train_error.pdf")
