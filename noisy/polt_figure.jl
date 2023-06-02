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

#normal form
function normal_form(du, u, p, t)
    #speed
    α =  p.α
    du[1] = α*u[1]-u[2]-u[1]*(u[1]^2+u[2]^2);
    du[2] = u[1]+α*u[2]-u[2]*(u[1]^2+u[2]^2);

    return du
end



θ1_ = load("./num/outputdata/θ1_.jld","θ1_");
θₜ = θ1_[1:4]
norm_bif_params = load("$(output_path)/norm_bif_params.jld","norm_bif_params");
pol_coeffs=load("$(output_path)/pol_coeffs.jld","pol_coeffs");


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

hidden = 23
ann = FastChain(
    FastDense(3, hidden, tanh), FastDense(hidden, hidden, tanh), FastDense(hidden, 2)
)
θn_=load("$(output_path)/θn_.jld","θn_");


learnt_model = trans_lt(θn_)
figures = Array{PGFPlotsX.Axis}(undef, vel_l)
for i in 1:vel_l
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


#Observed amplitude
amp = GetTestData.getTestData(vp)

#Predicted value
Ap2 = lt_pp_n(θn_)
amp2=zeros(length(Ap2))
for i=1:length(amp)
    amp2[i]=maximum(Ap2[i][1,:])
end


using Plots
plot(vp, amp, color=:blue, linewidth=1.5, label="Observational data",xlabel=L"$v$(m/s)",ylabel = L"$\|y_{w1}\|_{\infty}$(mm)")
plot!(vp, amp2, color=:red, seriestype=:scatter, label="Training results")
plot!(vp, amp2, color=:lightblue, fillrange=amp, fillalpha=0.3, label="error band")
savefig(amp_fig_path)
