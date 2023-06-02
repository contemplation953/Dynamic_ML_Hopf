using LinearAlgebra
using DiffEqFlux
using LaTeXStrings
using PGFPlotsX
using Statistics
using JLD
using HDF5
using Random
using Polynomials
using ComponentArrays


pwd()
include("../Global_Parameters.jl")
import .Global_Parameters

include("../Differentiate_Function.jl")
import .Differentiate_Function

include("../GetTestData.jl")
import .GetTestData

Vel = Global_Parameters.vdp_Vel
vel_l = Global_Parameters.vel_l
nh = Global_Parameters.nh
θ_l = 400
dig = Global_Parameters.normal_digits

u0 = Float32[0.01,0.01]
tl = 30.0;
tol = 1e-7
stol = 1e-8
st = 0.02;
st2 = 0.05

#规范型
function normal_form(du, u, p, t)
    #speed
    α =  p.α
    du[1] = α*u[1]-u[2]-u[1]*(u[1]^2+u[2]^2);
    du[2] = u[1]+α*u[2]-u[2]*(u[1]^2+u[2]^2);

    return du
end

AA = load("./vdp/inputdata/AA.jld","AA")
t_series = load("./vdp/inputdata/t_series.jld","t_series")

function f_coeff(vlT, Vel, u₀, v₀)
    Pr = zeros(2 * nh + 1, 0)
    for k in 1:length(Vel)
        z1 = vlT[k][1, :] - u₀ * ones(θ_l)
        z2 = vlT[k][2, :] - v₀ * ones(θ_l)
        theta = atan.(z2, z1)
        r = sqrt.(z1 .^ 2 + z2 .^ 2)
        tM = Array{Float64}(undef, 0, 2 * nh + 1)
        rr = Array{Float64}(undef, θ_l)
        for j in 1:θ_l
            tM1 = Array{Float64}(undef, 0, nh + 1)
            tM2 = Array{Float64}(undef, 0, nh)
            tM1_ = [cos(theta[j] * i) for i in 1:nh]
            tM2_ = [sin(theta[j] * i) for i in 1:nh]
            tM1_ = vcat(1, tM1_)
            tM1 = vcat(tM1, Transpose(tM1_))
            tM2 = vcat(tM2, Transpose(tM2_))
            tM_ = hcat(tM1, tM2)
            tM = vcat(tM, tM_)
        end
        MM = Transpose(tM) * tM
        rN = Transpose(tM) * r
        c = inv(MM) * rN
        Pr = hcat(Pr, c)
        Pr
    end
    return Pr
end

##Training T_L

scpp = 1e2
function predict_nt(θ1, θ2)
    for p in θ2
        if p <= 0
            return 1000
        end
    end

    p1, p2, p3, p4 = θ1
    T = [p1 p3; p2 p4]
    bif_params = θ2

    normal_t_series = [
       ts_dict["$(Int32(round(round(p; digits = dig) * 10^dig)))"] for p in θ2
    ]

    t_lco_series = [
       normal_t_series[i][:,end-θ_l+1:end] for i in 1:length(normal_t_series)
    ]

    vlT = [T * (t_lco_series[i]) for i in 1:length(Vel)]
    Pr = f_coeff(vlT, Vel, 0, 0)

    error = sum(abs2, AA .- Pr)
    return error
end

function predict_param(params, θn)
    AC = zeros(1, 0)
    pol_params = reshape(θn,1,length(θn))
    for x in params
        input = [x]
        output = pol_params * [1 x x^2 x^3 x^4 x^5]'
        AC = hcat(AC, output)
    end
    normalize_Vel_matx = reshape(collect(Vel),1,Global_Parameters.vel_l)
    pred_error = norm(normalize_Vel_matx - AC) * 10
    monotony_error = scpp * (monotony(params) + monotony(AC))
    return pred_error + monotony_error
end

#monotonically increasing
function monotony(x)
    if 0 in x
        return 1
    end
    for i in 1 : length(x)-1
        if x[i+1] <= x[i]
            return 1
        end
    end
    return 0
end

function loss_nt(θ)
    pred_param_error = predict_param(θ[5:12], θ[13:end])
    pred_nt_error = predict_nt(θ[1:4], θ[5:12])
    error = pred_param_error + pred_nt_error
    return error
end

iter = 0
function callbackf(t, l)
    global iter += 1
    if iter % 10 == 1
        process = iter / 20000 * 100;
        println("train $iter,error $(l),time $(time()), process $(process)% \n")
    end
    return false
end

#Pre-calculated normal data
ts_dict = load("./vdp/inputdata/ts.jld","ts")

#=

(approximate range given for efficiency) T_L
[2.4174520608279244 -1.529276429330012 1.567301583640992 2.4422721041917335;Vel;0,1,0,0,0,0]

θₜ = vec([1.0 0.0; 0.0 1.0])
=#
θₜ = vec([2.4174520608279244 -1.529276429330012 1.567301583640992 2.4422721041917335])

#Predicting bifurcation parameters for the normal form
norm_bif_params = vec(Vel);

#Polynomial coefficients
pol_params = vec([0.01 1 0.01 0.01 0.01 0.01]);
θ = vcat(θₜ,norm_bif_params);
θ = vcat(θ,pol_params);
res1 = DiffEqFlux.sciml_train(loss_nt, θ, ADAM(0.0001); maxiters=20000,cb=callbackf)
θ_ = res1.minimizer
save("./vdp/outputdata/θ_.jld","θ_",θ_)


norm_params = θ_[5]:0.04:θ_[12]
coeff = Polynomial(θ_[13:end])
vp = [coeff(p) for p in norm_params]

pol_params_res = θ_[13:end]
pred_V = [reshape(pol_params_res,1,length(pol_params_res)) * [1 x x^2 x^3 x^4 x^5]' for x in θ_[5:12]]
println(pred_V)

train_α_list = θ_[5:12]
pre_lco_series = Differentiate_Function.generate_LCO_ts(normal_form, train_α_list, nh, u0, tl, tol, stol, st, st2, 1, 2)
##

# NNT
function Array_chain(gu, ann, p)
    al = length(gu[1, :])
    AC = zeros(2, 0)
    for i in 1:al
        AC = hcat(AC, ann(gu[:, i], p))
    end
    return AC
end

#Predicted value
function trans_lt(θ_t, θnn)
    p1, p2, p3, p4 = θ_t[1:4]
    norm_p = θ_t[5:12]
    T = [p1 p3; p2 p4]
    vlT = [
        T * (pre_lco_series[i][:,end-θ_l+1:end]) +
        Array_chain([pre_lco_series[i][:,end-θ_l+1:end]; norm_p[i] * ones(1, θ_l)], ann, θnn)
        for i in 1:length(norm_p)
    ]
    return vlT
end

function loss_nn(θ)
    vlT = trans_lt(θ_, θ)
    Pr = f_coeff(vlT, Vel, 0, 0)
    error = sum(abs2, AA .- Pr)
    return error
end

hidden = 23
ann = FastChain(
    FastDense(3, hidden, tanh), FastDense(hidden, hidden, tanh), FastDense(hidden, 2)
)
θn = initial_params(ann)
res2 = DiffEqFlux.sciml_train(loss_nn, θn, ADAM(0.01); maxiters=2000,cb=callbackf)
θn_ = res2.minimizer
save("./vdp/outputdata/θn_.jld","θn_",θn_);
