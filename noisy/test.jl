using Random
using Polynomials
using ComponentArrays
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

include("../GetTestData.jl")
import .GetTestData

## parameter

#运算所需参数
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
snr = 30

AA = load("./noisy/inputdata/$(snr)_AA.jld","AA")
t_series = load("./noisy/inputdata/$(snr)_t_series.jld","t_series")
output_path = "./noisy/outputdata/$(snr)"
#提前计算好的normal ode
ts_dict = load("./noisy/inputdata/ts.jld","ts")

#规范型
function normal_form(du, u, p, t)
    #speed
    α =  p.α
    du[1] = α*u[1]-u[2]-u[1]*(u[1]^2+u[2]^2);
    du[2] = u[1]+α*u[2]-u[2]*(u[1]^2+u[2]^2);

    return du
end


function f_coeff(vlT, Vel, u₀, v₀)
    Pr = zeros(2 * nh + 1, 0)
    θ_l = length(vlT[1][1, :])
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
##训练旋转拉伸 T_L和系数的大致范围

function getPr(norm_bif_params)
    train_t_series = [
       ts_dict["$(Int32(round(round(p; digits = dig) * 10^dig)))"] for p in norm_bif_params
    ]

    train = [
       train_t_series[i][:,end-θ_l+1:end] for i in 1:length(norm_bif_params)
    ]

    return train
end

function loss1(θ)
    p1, p2, p3, p4 = θ[1:4]
    T = [p1 p3; p2 p4]

    vlT = [T * (train_t_series[i]) for i in 1:length(Vel)]
    Pr = f_coeff(vlT, Vel, 0, 0)
    pr_error = sum(abs2, AA .- Pr)

    AC = zeros(1, 0)
    θn = θ[5:end]
    θn = vcat(init_V₀,θn)
    pol_coeffs = reshape(θn,1,length(θn))
    for x in norm_bif_params
        input = [x]
        output = pol_coeffs * [1 x x^2 x^3 x^4 x^5]'
        AC = hcat(AC, output)
    end
    normalize_Vel_matx = reshape(collect(Vel),1,vel_l)
    pred_error = norm(normalize_Vel_matx - AC) * 10
    return pred_error + pr_error
end

#判断单调性递增
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

iter = 0
function callbackf(t, l)
    global iter += 1
    if iter % 100 == 1
        process = iter / 13000 * 100;
        println("train $iter,error $(l),time $(time()), process $(process)% \n")
    end
    return false
end

init_V₀ = 98
#旋转拉伸T_L
θₜ = vec([1.0 0.0; 0.0 1.0])
#预测标准型的分岔参数
norm_bif_params = Global_Parameters.hunting_Vel;
train_t_series = getPr(norm_bif_params)
#多项式系数
pol_coeffs = vec([1 1 1 1 1]);
θ = vcat(θₜ,pol_coeffs);
res = DiffEqFlux.sciml_train(loss1, θ, ADAM(0.1); maxiters=1000,cb=callbackf)
θ1_ = res.minimizer
save("$(output_path)/θ1_.jld","θ1_",θ1_);
# θ1_ = load("$(output_path)/θ1_.jld","θ1_");
θₜ = θ1_[1:4]
save("$(output_path)/θₜ.jld","θₜ",θₜ);
pol_coeffs = vcat(init_V₀,θ1_[5:end])
##

#精细的搜索
function predict_nt(bif_params)
    for p in bif_params
        if p <= 0
            return 1000
        end
    end

    p1, p2, p3, p4 = θₜ
    T = [p1 p3; p2 p4]

    normal_t_series = [
       ts_dict["$(Int32(round(round(p; digits = dig) * 10^dig)))"] for p in bif_params
    ]

    t_lco_series = [
       normal_t_series[i][:,end-θ_l+1:end] for i in 1:length(normal_t_series)
    ]

    vlT = [T * (t_lco_series[i]) for i in 1:length(Vel)]
    Pr = f_coeff(vlT, Vel, 0, 0)
    return sum(abs2, AA .- Pr)
end

function predict_param(params, coeffs)
    AC = zeros(1, 0)
    pol_coeffs = reshape(coeffs,1,length(coeffs))
    for x in params
        input = [x]
        output = pol_coeffs * [1 x x^2 x^3 x^4 x^5]'
        AC = hcat(AC, output)
    end
    normalize_Vel_matx = reshape(collect(Vel),1,vel_l)
    pred_error = norm(normalize_Vel_matx - AC) * 10
    monotony_error = 100 * (monotony(params) + monotony(AC))
    return pred_error + monotony_error
end

function loss_norm_param(θ)
    pred_param_error = predict_param(θ, pol_coeffs)
    pred_nt_error = predict_nt(θ)
    return pred_param_error + pred_nt_error
end
θ = collect(norm_bif_params)
res2 = DiffEqFlux.sciml_train(loss_norm_param, θ, Adam(0.0001); maxiters=2500,cb=callbackf)
norm_bif_params = res2.minimizer
save("$(output_path)/norm_bif_params.jld","norm_bif_params",norm_bif_params);
# norm_bif_params = load("$(output_path)/norm_bif_params.jld","norm_bif_params");

function loss_pol_coeffs(θ)
    pred_param_error = predict_param(norm_bif_params, θ)
    pred_nt_error = predict_nt(norm_bif_params)
    return pred_param_error + pred_nt_error
end

pol_coeffs=load("$(output_path)/pol_coeffs.jld","pol_coeffs");
θ = vec(pol_coeffs);
res1 = DiffEqFlux.sciml_train(loss_pol_coeffs, θ, Adam(0.01); maxiters=2500,cb=callbackf)
θ_ = res1.minimizer
pol_coeffs = θ_[1:end]
save("$(output_path)/pol_coeffs.jld","pol_coeffs",pol_coeffs);


pre_lco_series = Differentiate_Function.generate_LCO_ts(normal_form, norm_bif_params, nh, u0, tl, tol, stol, st, st2, 1, 2)
##

function Array_chain(gu, ann, p) # vectorized input-> vectorized neural net
    al = length(gu[1, :])
    AC = zeros(2, 0)
    for i in 1:al
        AC = hcat(AC, ann(gu[:, i], p))
    end
    return AC
end

function trans_lt(θnn) #predict the linear transformation
    p1, p2, p3, p4 = θₜ
    T = [p1 p3; p2 p4]
    vlT = [
        T * (pre_lco_series[i][:,end-θ_l+1:end]) +
        Array_chain([pre_lco_series[i][:,end-θ_l+1:end]; norm_bif_params[i] * ones(1, θ_l)], ann, θnn)
        for i in 1:length(norm_bif_params)
    ]
    return vlT
end

function loss_nn(θ)
    vlT = trans_lt(θ)
    Pr = f_coeff(vlT, Vel, 0, 0)
    error = sum(abs2, AA .- Pr)
    return error
end

hidden = 23
ann = FastChain(
    FastDense(3, hidden, tanh), FastDense(hidden, hidden, tanh), FastDense(hidden, 2)
)

# θn = initial_params(ann)
θn = load("$(output_path)/θn_.jld","θn_");
res2 = DiffEqFlux.sciml_train(loss_nn, θn, ADAM(0.01); maxiters=4000,cb=callbackf)
θn_ = res2.minimizer
save("$(output_path)/θn_.jld","θn_",θn_);


Vel_num = vel_l
PLOT_FLAG = true

if PLOT_FLAG
    learnt_model = trans_lt(θn_)
    figures = Array{PGFPlotsX.Axis}(undef, Vel_num)
    for i in 1:Vel_num
        ind = i
        a = @pgf PGFPlotsX.Axis(
            {
                xlabel = L"$y_{w1}$",
                ylabel = L"$\varphi_{w1}$",
                legend_pos = "north west",
                height = "8cm",
                width = "8cm",
                xmin = -15,
                xmax = 15,
                ymin = -10,
                ymax = 10,
                ylabel_shift="-10pt",
                xlabel_shift="-3pt"
            },
            Plot({color = "red", no_marks}, Coordinates(learnt_model[ind][1, :], learnt_model[ind][2, :])),
            LegendEntry("Training results"),
            Plot({color = "blue", only_marks, mark_size = "2pt"}, Coordinates(t_series[ind][1, end-150:2:end], t_series[ind][2, end-150:2:end])),
            LegendEntry("Observational data"),
        )
        figures[i]=a
    end

    g = @pgf GroupPlot(
    { group_style = { group_size="4 by 2" },
      no_markers
    },
    figures[1],figures[2],figures[3],figures[4],figures[5],figures[6],figures[7],figures[8])
    pgfsave("$(output_path)/noise_phase.pdf",g)
end



##

Vel2 = Global_Parameters.hunting_Vel2
coeff = Polynomial(pol_coeffs)
vp = [coeff(p) for p in Vel2]


function lt_pp_n(θnn)
    t_series = Differentiate_Function.generate_LCO_ts(normal_form, Vel2, nh, u0, tl, tol, stol, st, st2, 1, 2)

    p1, p2, p3, p4 = θₜ
    T = [p1 p3; p2 p4]

    vlT = [
        T * (t_series[i][:,end-θ_l+1:end]) +
        Array_chain([t_series[i][:,end-θ_l+1:end]; Vel2[i] * ones(1, θ_l)], ann, θnn)
        for i in 1:length(Vel2)
    ]
    return vlT
end


#观测幅值
amp = GetTestData.getTestData(vp)

#预测值
Ap2 = lt_pp_n(θn_)
amp2=zeros(length(Ap2))
for i=1:length(amp)
    amp2[i]=maximum(Ap2[i][1,:])
end

#参数0.1:1观测值的原始数据
amp_t=zeros(length(Vel))
for i=1:length(t_series)
    amp_t[i]=maximum(t_series[i][1,:])
end

using Plots
# 绘图
plot(vp, amp, color=:blue, linewidth=1.5, label="Observational data",xlabel=L"$v$(m/s)",ylabel = L"$\|y_{w1}\|_{\infty}$(mm)")
plot!(vp, amp2, color=:red, seriestype=:scatter, label="Training results")
plot!(vp, amp2, color=:lightblue, fillrange=amp, fillalpha=0.3, label="error band")
savefig("$(output_path)/noise_amp_$(snr).pdf")
