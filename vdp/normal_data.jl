using LinearAlgebra
using LaTeXStrings
using JLD
using HDF5


pwd()
include("../Global_Parameters.jl")
import .Global_Parameters

include("../Differentiate_Function.jl")
import .Differentiate_Function

nh = Global_Parameters.nh
θ_l = 150
u0 = Float32[0.01,0.01]
tl = 15.0;
tol = 2e-7
stol = 1e-8
st = 0.02;
st2 = 0.05
dig = Global_Parameters.normal_digits

#规范型
function normal_form(du, u, p, t)
    #speed
    α =  p.α
    du[1] = α*u[1]-u[2]-u[1]*(u[1]^2+u[2]^2);
    du[2] = u[1]+α*u[2]-u[2]*(u[1]^2+u[2]^2);

    return du
end

bif_params = 0:10.0^(-1*dig):0.04
ts_dict = Dict()
ff_dict = Dict()
for p in bif_params
    normal_t_series = Differentiate_Function.generate_LCO_ts(normal_form, p, nh, u0, tl, tol, stol, st, st2, 1, 2)

    t_lco_series = normal_t_series[1][:,end-149:end]
    index = Int64(round(p * 10.0^dig))
    if index % 100 == 0
        println(index)
    end
    ts_dict["$(index)"] = t_lco_series
end

save("./vdp/inputdata/ts.jld","ts",ts_dict)