using LinearAlgebra
using DiffEqFlux
using LaTeXStrings
using PGFPlotsX
using Statistics
using JLD
using HDF5

pwd()
include("../Global_Parameters.jl")
import .Global_Parameters

include("../Differentiate_Function.jl")
import .Differentiate_Function


## parameter

Vel = Global_Parameters.Vel
vel_l = Global_Parameters.vel_l
nh = Global_Parameters.nh
V₀ = Global_Parameters.V₀
dig = Global_Parameters.normal_digits

u0 = Float32[0.01,0.01]
tl = 30;
tol = 1e-7
stol = 1e-8
st = 0.02;
st2 = 0.02
θ_l = 900


Vel_num = Global_Parameters.vel_l
Vel2 = Global_Parameters.Vel2


global output_path,AA,t_series,ts_dict
for snr in [10 20 30 40]
    global AA = load("./noisy/inputdata/$(snr)_AA.jld","AA")
    global t_series = load("./noisy/inputdata/$(snr)_t_series.jld","t_series")
    global output_path = "./noisy/outputdata/$(snr)"
    #Pre-calculated normal data
    global ts_dict = load("./noisy/inputdata/ts.jld","ts")
    include("trainProcess.jl")
end

global amp_fig
for snr in [10 20 30 40]
    global AA = load("./noisy/inputdata/$(snr)_AA.jld","AA")
    global t_series = load("./noisy/inputdata/$(snr)_t_series.jld","t_series")
    global output_path = "./noisy/outputdata/$(snr)"
    #Pre-calculated normal data
    global ts_dict = load("./noisy/inputdata/ts.jld","ts")
    global amp_fig_path = "$(output_path)/noise_amp_$(snr).pdf"
    include("polt_figure.jl")
end
