module Global_Functions

using DifferentialEquations
using LinearAlgebra
using LaTeXStrings
using Statistics
using ComponentArrays

export generate_LCO,test

function test()
    println("v4")
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
    zero = Array{Float64}(undef, 0)
    T = Array{Float64}(undef, 0)
    low_p = Array{Float64}(undef, 0)
    high_p = Array{Float64}(undef, 0)
    Ti = Array{Int64}(undef, 0)
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
    h₀ = mean(high_p) + mean(low_p)
    h₀ = h₀ / 2
    for i in 2:(l - 1)
        sign_con = (u[i][ind] - h₀) * (u[i + 1][ind] - h₀)
        if sign_con < 0
            if (u[i][ind + 1] + u[i + 1][ind + 1]) / 2 > 0
                zero = vcat(zero, (u[i][ind] + u[i - 1][ind]) / 2)
                Ti = vcat(Ti, i)
                T = vcat(T, t[i])
            end
        end
    end
    t_l = length(T)
    P = Array{Float64}(undef, t_l - 1)
    for j in 2:t_l
        P[j - 1] = T[j] - T[j - 1]
    end
    return (T=Ti, P=P, hp=high_p, lp=low_p, h₀=h₀)
end

function get_stable_LCO(p, u0, tl, tol, eq, stol, rp, ind1, ind2, u₀, v₀, st) # Get a stable LCO from numerical integration
    u = u0
    dim = length(u0)
    p = ComponentArray(α=p)
    prob = ODEProblem(eq, u, (0, tl * rp), p)
    sol = DifferentialEquations.concrete_solve(prob, Tsit5(), u0, p; reltol=stol, abstol=stol, saveat=st)
    vP = 1
    P = 0
    T = 0
    count_index = 0
    while vP > tol
        u = sol.u[end]
        prob = ODEProblem(eq, u, (0, tl), p)
        sol = DifferentialEquations.solve(
            prob, Tsit5(); reltol=stol, abstol=stol, saveat=st
        )
        z = zero_measure(sol.u, 1, sol.t)
        vP = Statistics.var(z.hp)
        P = z.P[1]
        T = z.T
        count_index = count_index + 1
        if count_index > 20
            println("More than 40 times without limit cycle,v=$p,vP=$vP,tol=$tol")
            break
        end
    end
    tl = length(sol)
    uu = Array{Float64}(undef, tl, dim)
    for i in 1:dim
        uu[:, i] = get_sol2(sol, i)
    end
    t = Array{Float64}(undef, length(sol))
    r = Array{Float64}(undef, length(sol))
    u = uu[:, ind1]
    v = uu[:, ind2]
    for i in 1:length(u)
        t[i] = atan(v[i] - v₀, u[i] - u₀)
        r[i] = sqrt((u[i] - u₀)^2 + (v[i] - v₀)^2)
    end
    return (u=uu, t=t, r=r, P=P, T=T)
end


function LS_harmonics(r, t, ω, N)
    # Computing Fourier coefficients of the amplitude in the measued state-variable coordinates
    # Fourier coefficients are computed in least square sence
    c = Array{Float64}(undef, 2 * N + 1)
    M = Array{Float64}(undef, 1, 2 * N + 1)
    tM = Array{Float64}(undef, 0, 2 * N + 1)
    tl = length(t)
    rr = Array{Float64}(undef, tl)
    M[1] = 1
    for j in 1:tl
        for i in 1:N
            M[1 + i] = cos(ω * t[j] * i)
            M[1 + N + i] = sin(ω * t[j] * i)
        end
        tM = vcat(tM, M)
    end
    MM = transpose(tM) * tM
    rN = transpose(tM) * r
    MM
    rN
    c = inv(MM) * rN
    for j in 1:tl
        rr[j] = c[1]
        for i in 1:N
            rr[j] += c[i + 1] * cos(ω * t[j] * i)
            rr[j] += c[i + 1 + N] * sin(ω * t[j] * i)
        end
    end
    return (coeff=c, rr=rr)
end

#Generate normal form data
function generate_LCO(eq, Vel, nh, u0, tl, tol, stol, st, st2, ind1, ind2)
    rp = 5
    vel_l = length(Vel)
    AA = zeros(vel_l, Int(nh * 2 + 1))
    p_ = Vel[1]

    g = get_stable_LCO(p_, u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st)
    u₀ = mean(g.u[:, ind1])
    v₀ = mean(g.u[:, ind2])
    for i in 1:vel_l
        p_ = Vel[i]
        g = get_stable_LCO(p_, u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st)
        r = g.r
        t = g.t
        c = LS_harmonics(r, t, 1, nh).coeff
        AA[i, :] = c
    end
    AA = transpose(AA)
    t_series = [
        Transpose(
            get_stable_LCO(Vel[i], u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st2).u[
                :, [ind1, ind2]
            ],
        ) for i in 1:vel_l
    ]
    θ_series = [
        get_stable_LCO(Vel[i], u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st2).t for
        i in 1:vel_l
    ]
    return (data=AA, ts=t_series, theta_s=θ_series)
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
    #时序解
    t_data_cycle_1=t_data[1,first(index_sers):last(index_sers)]
    t_data_cycle_2=t_data[2,first(index_sers):last(index_sers)]
    time_sol = hcat(t_data_cycle_1,t_data_cycle_2)

    #相位角
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
