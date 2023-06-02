using LinearAlgebra
using LaTeXStrings
using PGFPlotsX
using JLD
using HDF5
using Polynomials
using DiffEqFlux


pwd()
include("../Global_Parameters.jl")
import .Global_Parameters

include("../GetTestData.jl")
import .GetTestData

include("../Differentiate_Function.jl")
import .Differentiate_Function

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

t_series = load("./vdp/inputdata/t_series.jld","t_series")
θₜ = vec([2.4174520608279244 -1.529276429330012 1.567301583640992 2.4422721041917335])
θ_ = load("./vdp/outputdata/θ_.jld","θ_")

norm_params = θ_[5]:0.04:θ_[12]
coeff = Polynomial(θ_[13:end])

train_α_list = θ_[5:12]
pre_lco_series = Differentiate_Function.generate_LCO_ts(normal_form, train_α_list, nh, u0, tl, tol, stol, st, st2, 1, 2)
##

function Array_chain(gu, ann, p) # vectorized input-> vectorized neural net
    al = length(gu[1, :])
    AC = zeros(2, 0)
    for i in 1:al
        AC = hcat(AC, ann(gu[:, i], p))
    end
    return AC
end

function trans_lt(θ_t, θnn) #predict the linear transformation
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

hidden = 23
ann = FastChain(
    FastDense(3, hidden, tanh), FastDense(hidden, hidden, tanh), FastDense(hidden, 2)
)

θn_=load("./vdp/outputdata/θn_.jld","θn_")

Vel_num = Global_Parameters.vel_l


learnt_model = trans_lt(θ_, θn_)
figures = Array{PGFPlotsX.Axis}(undef, Vel_num)
for i in 1:Vel_num
    ind = i
    mu = round(train_α_list[i]; digits = 3)
    a = @pgf PGFPlotsX.Axis(
        {
            xlabel = "y",
            ylabel = "x",
            legend_pos = "north west",
            legend_style = "{font=\\fontsize{8}{10}\\selectfont}",
            height = "8cm",
            width = "8cm",
            xmin = -0.45,
            xmax = 0.45,
            ymin = -0.45,
            ymax = 0.45,
            ylabel_shift="-10pt",
            xlabel_shift="-3pt",
            title=string(L"$\mu$","=$(mu)")
        },
        Plot({color = "red", mark="square"}, Coordinates(learnt_model[ind][1, :], learnt_model[ind][2, :])),
        LegendEntry("Training results"),
        Plot({color = "blue", only_marks, mark_size = "2pt"}, Coordinates(t_series[ind][1, end-θ_l:5:end], t_series[ind][2, end-θ_l:5:end])),
        LegendEntry("Observational data"),
    )
    figures[i]=a
end

g = @pgf GroupPlot(
{ group_style = { group_size="4 by 2","vertical sep=1.5cm" },
  no_markers
},
figures[1],figures[2],figures[3],figures[4],figures[5],figures[6],figures[7],figures[8])
pgfsave("./vdp/outputdata/vdp_phase.pdf",g)


##
norm_params = range(θ_[5],θ_[12],30)
coeff = Polynomial(θ_[13:end])
pv = [coeff(p) for p in norm_params]


function lt_pp_n(θ_, θnn)
    t_series = Differentiate_Function.generate_LCO_ts(normal_form, norm_params, nh, u0, tl, tol, stol, st, st2, 1, 2)

    p1, p2, p3, p4 = θ_[1:4]
    T = [p1 p3; p2 p4]

    vlT = [
        T * (t_series[i][:,end-θ_l+1:end]) +
        Array_chain([t_series[i][:,end-θ_l+1:end]; norm_params[i] * ones(1, θ_l)], ann, θnn)
        for i in 1:length(norm_params)
    ]
    return vlT
end

#Observed amplitude
include("basic_function.jl")
tl = 35.0;
st = 0.05;
st2 = 0.02;
nh = Global_Parameters.nh
tol = 1e-7
stol = 1e-8
amp_dat = generate_data(length(pv), pv, nh, u0, tol, stol, tl, st, st2).ts
amp=zeros(length(amp_dat))
for i=1:length(amp)
    amp[i]=maximum(amp_dat[i][1,end-1000:end])
end

#Predicted value
Ap2 = lt_pp_n(θ_, θn_)
amp2=zeros(length(Ap2))
for i=1:length(amp)
    amp2[i]=maximum(Ap2[i][1,:])
end

using Plots

plot(pv, amp, color=:blue, linewidth=1.5, label="Observational data",xlabel=L"$\mu$",ylabel = L"$y$")
plot!(pv, amp2, color=:red, seriestype=:scatter, label="Training results")
plot!(pv, amp, color=:transparent, fillrange=amp2, fillalpha=0.3, label="error band")
savefig("./vdp/outputdata/vdp_amp.pdf")


normal_form_tran = norm_params./sqrt.(ones(length(norm_params),1)-norm_params.^2);
plot(norm_params, vec(normal_form_tran), color=:blue, linewidth=1.5, label="Ground truth",xlabel=L"$\beta$",ylabel = L"$\Phi(\beta)$")
plot!(norm_params, pv, color=:red, seriestype=:scatter, label="Training results")
plot!(norm_params, vec(normal_form_tran), color=:transparent, fillrange=pv, fillalpha=0.3, label="error band")
savefig("./vdp/outputdata/vdp_params.pdf")
