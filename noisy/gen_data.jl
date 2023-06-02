using LinearAlgebra
using LaTeXStrings
using PGFPlotsX
using JLD
using HDF5
using Statistics
using Random

pwd()
include("../Global_Parameters.jl")
import .Global_Parameters

t_s = load("./num/inputdata/t_series.jld","t_series")
##
function awgn(X,SNR)
    #Assumes X to be a matrix and SNR a signal-to-noise ratio specified in decibel (dB)
    #Implented by author, inspired by https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/
    N=length(X) #Number of elements in X
    signalPower = sum(X[:].^2)/N
    linearSNR = 10^(SNR/10)
    len=length(X)
    noiseMat = sqrt(signalPower/linearSNR)*randn(len) #Random gaussian noise, scaled according to the signal power of the entire matrix (!) and specified SNR

    return solution = X + noiseMat
end

function get_noisy_LCO(ts, SNR)
    AA = zeros(vel_l, Int(nh * 2 + 1))
    u₀ = 0
    v₀ = 0
    res_ts = []
    for i in 1:length(ts)
        t = Array{Float64}(undef, length(ts[1][1,:]))
        r = Array{Float64}(undef, length(ts[1][1,:]))
        u = awgn(ts[i][1, :]/1000, SNR)*1000
        v = awgn(ts[i][2, :]/1000, SNR)*1000
        push!(res_ts,transpose([u v]))

        for i in 1:length(u)
            t[i] = atan(v[i] - v₀, u[i] - u₀)
            r[i] = sqrt((u[i] - u₀)^2 + (v[i] - v₀)^2)
        end
        c = LS_harmonics(r, t, 1, nh).coeff
        AA[i, :] = c
    end
    AA = transpose(AA)
    return (ts=res_ts, AA=AA)
end

SNRs = [10 15 20 30 40]
for i in 1:length(SNRs)
    res = get_noisy_LCO(t_s, SNRs[i])
    save("./noisy/inputdata/$(SNRs[i])_t_series.jld","t_series",res.ts)
    save("./noisy/inputdata/$(SNRs[i])_AA.jld","AA",res.AA)
end
