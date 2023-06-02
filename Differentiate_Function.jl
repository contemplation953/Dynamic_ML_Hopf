module Differentiate_Function

using DifferentialEquations
using LinearAlgebra
using LaTeXStrings
using Statistics
using ComponentArrays

export generate_LCO,test

##to avoid array mutation

function test()
    println("v4")
end

function trans2Matrix(sol)
    lu = length(sol.u)
    u1 = [sol.u[i][1] for i in 1: lu]
    u2 = [sol.u[i][2] for i in 1: lu]
    u = vcat(u1',u2')
    return u
end

function get_sol2(sol, ind) # Convert solution of ODE solver to array solution
    lu = length(sol.u)
    u = Vector{Float64}(undef, lu)
    for i in 1:lu
        uv = sol.u[i]
        u[i] = uv[ind]
    end
    return u
end

# Measuring the time of zero crossing points from numerical continuation
function zero_measure(u, ind, t)
    l = length(u)
    low_p = Array{Float64}(undef, 0)
    high_p = Array{Float64}(undef, 0)
    for i in 2:(l - 1)
        sign_con2 = u[i][ind + 1] * u[i - 1][ind + 1]
        if sign_con2 < 0
            if (u[i][ind] + u[i - 1][ind]) / 2 < 0
                low_p = vcat(low_p, (u[i][ind] + u[i - 1][ind]) / 2)
            else
                high_p = vcat(high_p, (u[i][ind] + u[i - 1][ind]) / 2)
            end
        end
    end
    return high_p
end

function get_stable_LCO_ts(p, u0, tl, tol, eq, stol, rp, ind1, ind2, u₀, v₀, st) # Get a stable LCO from numerical integration
    p = ComponentArray(α=p)
    prob = ODEProblem(eq, u0, (0, tl * rp), p)
    sol = DifferentialEquations.concrete_solve(prob, Tsit5(), u0, p; reltol=stol, abstol=stol, saveat=st)
    vP = 1
    count_index = 0
    while vP > tol
        u = sol.u[end]
        prob = ODEProblem(eq, u, (0, tl), p)
        sol = DifferentialEquations.solve(
            prob, Tsit5(); reltol=stol, abstol=stol, saveat=st
        )
        z = zero_measure(sol.u, 1, sol.t)
        vP = Statistics.var(z)
        count_index = count_index + 1
        if count_index > 40
            println("More than 40 times without limit cycle,p=$p,vP=$vP,tol=$tol")
            break
        end
    end
    return trans2Matrix(sol)
end


#Generate normal form ,time series
function generate_LCO_ts(eq, Vel, nh, u0, tl, tol, stol, st, st2, ind1, ind2)
    rp = 2
    vel_l = length(Vel)
    t_series = [
        get_stable_LCO_ts(Vel[i], u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st2) for i in 1:vel_l
    ]
    return t_series
end


#Finding the cycle from the data
function searchCycle(t_data, cycle_num)
    data_lengen = length(t_data[1,:])
    po_t = Array{Float64}(undef, data_lengen)
    po_r = Array{Float64}(undef, data_lengen)
    v = t_data[1,:]
    u = t_data[2,:]
    for i in 1:data_lengen
        po_t[i] = atan(u[i], v[i])
        po_r[i] = sqrt((u[i])^2 + (v[i])^2)
    end
    search_num = 2 * cycle_num
    index_sers = Array{Int32}(undef,0)
    for i  = 1 : length(po_t) - 1
        if(search_num < 0 )
            break
        end

        if po_t[i] * po_t[i+1] > 0
            continue
        end

        if length(index_sers) == 0
            index_sers = vcat(index_sers,i+1)
        else
            index_sers = vcat(index_sers,i,i+1)
        end

        search_num = search_num -1
    end
    #time series
    t_data_cycle_1=t_data[1,first(index_sers):last(index_sers)]
    t_data_cycle_2=t_data[2,first(index_sers):last(index_sers)]
    time_sol = hcat(t_data_cycle_1,t_data_cycle_2)

    #phase angle
    θ_series = po_t[first(index_sers):last(index_sers)]
    return (t_series=time_sol,θ_series=θ_series)
end

function f_coeff(vlT, num, u₀, v₀, nh, θ_l)
    Pr = zeros(2 * nh + 1, 0)
    for k in 1:num
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

end
