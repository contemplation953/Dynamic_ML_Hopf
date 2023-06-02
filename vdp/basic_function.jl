using DifferentialEquations
using LinearAlgebra
using LaTeXStrings
using JLD
using HDF5
using Statistics


function vdp(du, u, p, t)
    #speed
    a = p[1]

    du[1] = u[2];
    du[2] = 2*a*u[2]-u[1]^2*u[2]-u[1];

    return du
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

function zero_measure(u, ind, t) # Measuring the time of zero crossing points from numerical continuation
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
    prob = ODEProblem(eq, u, (0, tl * rp), p)
    sol = DifferentialEquations.solve(prob, Tsit5(); reltol=stol, abstol=stol, saveat=st)
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

function LS_harmonics(r, t, ω, N) # Computing Fourier coefficients of the amplitude in the measued state-variable coordinates
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

function generate_data(vel_l, Vel, nh, u0, tol, stol, tl, st, st2)
    #Generate training data
    u0 = u0
    eq = vdp
    rp = 5
    ind1 = 1
    ind2 = 2
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
