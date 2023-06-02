using DifferentialEquations
using LinearAlgebra
using LaTeXStrings
using JLD
using HDF5
using Statistics

pwd()
include("../Global_Parameters.jl")
import .Global_Parameters


include("basic_function.jl")

tl = 35.0;
st = 0.05;

u0 = Float32[0.1,0,0.1];
st2 = 0.02;
nh = Global_Parameters.nh
tol = 1e-7
stol = 1e-8


vel_l = Global_Parameters.vel_l
Vel = Global_Parameters.vdp_Vel

dat = generate_data(vel_l, Vel, nh, u0, tol, stol, tl, st, st2)
save("./vdp/inputdata/AA.jld","AA",dat.data)
save("./vdp/inputdata/t_series.jld","t_series",dat.ts)

Vel2 = Global_Parameters.vdp_Vel2
dat2 = generate_data(length(Vel2), Vel2, nh, u0, tol, stol)
save("./vdp/inputdata/test_AA.jld","AA",dat2.data)
save("./vdp/inputdata/test_t_series.jld","test_t_series",dat2.ts)

dat2_ts = dat2.ts
amp =zeros(length(dat2_ts))
for i=1:length(dat2_ts)
    amp[i] = maximum(dat2_ts[i][1,:])
end
save("./vdp/inputdata/amp.jld","amp",amp)
