using LinearAlgebra, JuMP, Gurobi, Distributions, Statistics, Mosek, MosekTools
using Ipopt, DelimitedFiles, HDF5, Random, CSV, DataFrames


###################     Global functions   ########################

function generate_hyperplane_eigen(x_opt,X_opt)
    n_x = size(X_opt,1)
    Y = X_opt .- x_opt*x_opt'
    (λ,U) = eigen(Y)
    f = U[:,n_x]
    l = f'*x_opt
    return false, f, l
end

function calculate_mc_input(x,y,U)
    X = [x]
    for j in 1:size(y)[1]
        if y[j] == 0
            push!(X,x)
        else
            push!(X,U[:,j]/y[j])
        end
    end
    return X
end

function calculate_candidate_vectors(x,X)
    Y = [x]
    for j in 1:size(x)[1]
        if x[j] == 0
            push!(Y,x)
        else
            push!(Y,X[:,j]/x[j])
        end
    end
    return Y
end

function calculate_different_vectors(X,x)
    Y = x
    for i in 1:length(X)
        if norm(X[i] .- x) > 1e-3
            Y = hcat(Y,X[i])
        end
    end
    return Y
end

function is_point_feasible(C,d,x)
    res = (sum(C*x .<= d)  == size(C,1))
    return res
end

function poly3_init(c3,c2,c1,c0,x)
    n = size(x,1)
    res_1 = sum(c3[i,j,k]*x[i]*x[j]*x[k] for i in 1:n for j in 1:n for k in 1:n)
    res_2 = x'*c2*x + x'*c1 + c0
    return res_1 + res_2
end


function generate_degree3_polynomial(n, density=0.5)
    # Initialize coefficients with zeros
    c3 = zeros(Float64, n, n, n)  # For cubic terms
    c2 = zeros(Float64, n, n)     # For quadratic terms
    c1 = zeros(Float64, n)        # For linear terms
    c0 = 0.0                      # Constant term

    # Populate cubic coefficients sparsely
    num_cubic_terms = Int(density * n^3)
    for _ in 1:num_cubic_terms
        i, j, k = rand(1:n, 3)  # Random indices
        c3[i, j, k] = rand(-5.0:0.1:5.0)  # Random coefficient in range [-5, 5]
    end

    # Populate quadratic coefficients sparsely
    num_quadratic_terms = Int(density * n^2)
    for _ in 1:num_quadratic_terms
        i, j = rand(1:n, 2)  # Random indices
        c2[i, j] = rand(-5.0:0.1:5.0)  # Random coefficient in range [-5, 5]
    end

    # Populate linear coefficients sparsely
    num_linear_terms = Int(density * n)
    for _ in 1:num_linear_terms
        i = rand(1:n)  # Random index
        c1[i] = rand(-5.0:0.1:5.0)  # Random coefficient in range [-5, 5]
    end

    # Set a random constant term
    c0 = rand(-5.0:0.1:5.0)

    return c3, c2, c1, c0
end

############################   X_1   ############################

function poly3_slc_obj_X1(p2_list,q2_list,x)
    n = size(x,1)
    P_list, r_list, w_list = p2_list[1], p2_list[2], p2_list[3]
    Q_list, f_list, g_list = q2_list[1], q2_list[2], q2_list[3]
    res_1 = sum( x[i]*(x'*P_list[i]*x + x'*r_list[i] + w_list[i]) for i in 1:n)
    res_2 = sum( (1-x[i])*(x'*Q_list[i]*x + x'*f_list[i] + g_list[i]) for i in 1:n)
    return res_1 + res_2
end

function get_poly3_slc_decomposition_X1_sdp(c3,c2,c1,c0)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, P[1:n,1:n,1:n])
    @variable(m1, r[1:n,1:n])
    @variable(m1, w[1:n])
    @variable(m1, Q[1:n,1:n,1:n])
    @variable(m1, f[1:n,1:n])
    @variable(m1, g[1:n])
    # zero degree terms
    @constraint(m1, sum(g[i] for i in 1:n) == c0)
    # first degree terms
    @constraint(m1, [i in 1:n], w[i]-g[i]+sum(f[i,l] for l in 1:n) == c1[i])
    # second degree terms
    for i in 1:n
        for j in i:n
            if i != j
                @constraint(m1, r[j,i]+r[i,j]-f[j,i]-f[i,j]+sum(Q[i,j,l]+Q[j,i,l] for l in 1:n) == c2[i,j]+c2[j,i])
            else
                @constraint(m1, r[i,i]-f[i,i]+sum(Q[i,i,l] for l in 1:n) == c2[i,i])
            end
        end
    end
    # third degree terms
    for i in 1:n
        for j in i:n
            for k in j:n
                if i != j && i != k && j != k
                    @constraint(m1, P[j,k,i]+P[k,j,i]+P[i,k,j]+P[k,i,j]+P[i,j,k]+P[j,i,k]
                                   -Q[j,k,i]-Q[k,j,i]-Q[i,k,j]-Q[k,i,j]-Q[i,j,k]-Q[j,i,k]
                                   == c3[i,j,k]+c3[i,k,j]+c3[j,i,k]+c3[j,k,i]+c3[k,i,j]+c3[k,j,i])
                elseif i == j && k != j
                    @constraint(m1, P[i,i,k]+P[i,k,i]+P[k,i,i]-Q[i,i,k]-Q[i,k,i]-Q[k,i,i] == c3[k,i,i]+c3[i,k,i]+c3[i,i,k])
                elseif i == k && j != k
                    @constraint(m1, P[i,i,j]+P[j,i,i]+P[i,j,i]-Q[i,i,j]-Q[j,i,i]-Q[i,j,i] == c3[j,i,i]+c3[i,j,i]+c3[i,i,j])
                elseif j == k && i != j
                    @constraint(m1, P[j,j,i]+P[i,j,j]+P[j,i,j]-Q[j,j,i]-Q[i,j,j]-Q[j,i,j] == c3[i,j,j]+c3[j,i,j]+c3[j,j,i])
                else
                    @constraint(m1, P[i,i,i]-Q[i,i,i] == c3[i,i,i])
                end

            end
        end
    end
    # Constraints for convexity
    @constraint(m1, [i in 1:n], P[:,:,i] in PSDCone())
    @constraint(m1, [i in 1:n], Q[:,:,i] in PSDCone())
    @objective(m1, Min, 0)
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL
        P_list = [JuMP.value.(P)[:,:,i] for i in 1:n]
        r_list = [JuMP.value.(r)[:,i] for i in 1:n]
        w_list = JuMP.value.(w)
        p2_list = [P_list, r_list, w_list]
        Q_list = [JuMP.value.(Q)[:,:,i] for i in 1:n]
        f_list = [JuMP.value.(f)[:,i] for i in 1:n]
        g_list = JuMP.value.(g)
        q2_list = [Q_list, f_list, g_list]
    else
        println("Error")
        p2_list, q2_list = [], []
    end
    return p2_list, q2_list
end

function get_poly3_slc_decomposition_X1_lp(c3,c2,c1,c0)
    m1 = Model(Mosek.Optimizer)
    @variable(m1, P[1:n,1:n,1:n])
    @variable(m1, r[1:n,1:n])
    @variable(m1, w[1:n])
    @variable(m1, Q[1:n,1:n,1:n])
    @variable(m1, f[1:n,1:n])
    @variable(m1, g[1:n])
    @variable(m1, P_aux[1:n,1:n,1:n])
    @variable(m1, Q_aux[1:n,1:n,1:n])
    # zero degree terms
    @constraint(m1, sum(g[i] for i in 1:n) == c0)
    # first degree terms
    @constraint(m1, [i in 1:n], w[i]-g[i]+sum(f[i,l] for l in 1:n) == c1[i])
    # second degree terms
    for i in 1:n
        for j in i:n
            if i != j
                @constraint(m1, r[j,i]+r[i,j]-f[j,i]-f[i,j]+sum(Q[i,j,l]+Q[j,i,l] for l in 1:n) == c2[i,j]+c2[j,i])
            else
                @constraint(m1, r[i,i]-f[i,i]+sum(Q[i,i,l] for l in 1:n) == c2[i,i])
            end
        end
    end
    # third degree terms
    for i in 1:n
        for j in i:n
            for k in j:n
                if i != j && i != k && j != k
                    @constraint(m1, P[j,k,i]+P[k,j,i]+P[i,k,j]+P[k,i,j]+P[i,j,k]+P[j,i,k]
                                   -Q[j,k,i]-Q[k,j,i]-Q[i,k,j]-Q[k,i,j]-Q[i,j,k]-Q[j,i,k]
                                   == c3[i,j,k]+c3[i,k,j]+c3[j,i,k]+c3[j,k,i]+c3[k,i,j]+c3[k,j,i])
                elseif i == j && k != j
                    @constraint(m1, P[i,i,k]+P[i,k,i]+P[k,i,i]-Q[i,i,k]-Q[i,k,i]-Q[k,i,i] == c3[k,i,i]+c3[i,k,i]+c3[i,i,k])
                elseif i == k && j != k
                    @constraint(m1, P[i,i,j]+P[j,i,i]+P[i,j,i]-Q[i,i,j]-Q[j,i,i]-Q[i,j,i] == c3[j,i,i]+c3[i,j,i]+c3[i,i,j])
                elseif j == k && i != j
                    @constraint(m1, P[j,j,i]+P[i,j,j]+P[j,i,j]-Q[j,j,i]-Q[i,j,j]-Q[j,i,j] == c3[i,j,j]+c3[j,i,j]+c3[j,j,i])
                else
                    @constraint(m1, P[i,i,i]-Q[i,i,i] == c3[i,i,i])
                end

            end
        end
    end
    # Constraints for convexity
    for i in 1:n
        for j in 1:n
            ind_list = [k for k in 1:n if k != j]
            @constraint(m1, sum(P_aux[j,k,i] for k in ind_list) <= P[j,j,i])
            @constraint(m1, sum(Q_aux[j,k,i] for k in ind_list) <= Q[j,j,i])
        end
    end
    @constraint(m1, P .<= P_aux)
    @constraint(m1, -P .<= P_aux)
    @constraint(m1, Q .<= Q_aux)
    @constraint(m1, -Q .<= Q_aux)
    @objective(m1, Min, 0)
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL
        P_list = [JuMP.value.(P)[:,:,i] for i in 1:n]
        r_list = [JuMP.value.(r)[:,i] for i in 1:n]
        w_list = JuMP.value.(w)
        p2_list = [P_list, r_list, w_list]
        Q_list = [JuMP.value.(Q)[:,:,i] for i in 1:n]
        f_list = [JuMP.value.(f)[:,i] for i in 1:n]
        g_list = JuMP.value.(g)
        q2_list = [Q_list, f_list, g_list]
    else
        println("Error")
        p2_list, q2_list = [], []
    end
    return p2_list, q2_list
end

function solve_rpt_relaxation_X1_random_slc(C,d,p2_list,q2_list,use_lmi)
    L, n = size(C)
    P_list, r_list, w_list = p2_list[1], p2_list[2], p2_list[3]
    Q_list, f_list, g_list = q2_list[1], q2_list[2], q2_list[3]

    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, t[1:n])
    @variable(m1, v[1:n])
    @variable(m1, X[1:n,1:n]>=0, Symmetric)

    @constraint(m1, [i in 1:n], [t[i]+x[i];
                                vcat(2*P_list[i]^(0.5)*X[:,i], t[i]-x[i])]
                                in SecondOrderCone())

    @constraint(m1, [i in 1:n], [v[i]+1-x[i];
                                vcat(2*Q_list[i]^(0.5)*(x.-X[:,i]), v[i]-1+x[i])]
                                in SecondOrderCone())

    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], C*X[:,i] .<= x[i]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    if use_lmi
        @constraint(m1, [X  x;  x' 1] in PSDCone())
    end
    @objective(m1, Min,sum(t[i] + r_list[i]'*X[:,i] + w_list[i]*x[i] +
                           v[i] + f_list[i]'*(x .- X[:,i]) + g_list[i]*(1 - x[i]) for i in 1:n))
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL
        return true, JuMP.value.(x), JuMP.value.(X), objective_value(m1)
    else
        return false, zeros(n), zeros(n,n), 1e6
    end
end

function get_uncertainty_set_z1(c3,c2,c1,c0)
    n = size(c1,1)
    A_list = []
    B_list = []
    c_list = []
    d_list = []
    μ_list = []
    ν_list = []
    Ξ_list = []
    ω_list = []
    γ_list = []
    s1 = []
    # third degree terms
    for i in 1:n
        for j in i:n
            for k in j:n
                A = zeros(n,n,n)
                B = zeros(n,n,n)
                c = zeros(n,n)
                d = zeros(n,n)
                μ = zeros(n)
                ν = zeros(n)
                Ξ = zeros(n,n)
                ω = zeros(n)
                γ = 0
                if i != j && i != k && j != k
                    A[j,k,i] = 1
                    A[k,j,i] = 1
                    A[i,k,j] = 1
                    A[k,i,j] = 1
                    A[i,j,k] = 1
                    A[j,i,k] = 1
                    B[j,k,i] = -1
                    B[k,j,i] = -1
                    B[i,k,j] = -1
                    B[k,i,j] = -1
                    B[i,j,k] = -1
                    B[j,i,k] = -1
                    push!(s1, c3[i,j,k]+c3[i,k,j]+c3[j,i,k]+c3[j,k,i]+c3[k,i,j]+c3[k,j,i])
                elseif i == j && j != k
                    A[i,i,k] = 1
                    A[i,k,i] = 1
                    A[k,i,i] = 1
                    B[i,i,k] = -1
                    B[i,k,i] = -1
                    B[k,i,i] = -1
                    push!(s1, c3[k,i,i]+c3[i,k,i]+c3[i,i,k])
                elseif i == k && j != i
                    A[i,i,j] = 1
                    A[j,i,i] = 1
                    A[i,j,i] = 1
                    B[i,i,j] = -1
                    B[j,i,i] = -1
                    B[i,j,i] = -1
                    push!(s1, c3[j,i,i]+c3[i,j,i]+c3[i,i,j])
                elseif j == k && i != j
                    A[j,j,i] = 1
                    A[i,j,j] = 1
                    A[j,i,j] = 1
                    B[j,j,i] = -1
                    B[i,j,j] = -1
                    B[j,i,j] = -1
                    push!(s1, c3[i,j,j]+c3[j,i,j]+c3[j,j,i])
                elseif i == j && j == k
                    A[i,i,i] = 1
                    B[i,i,i] = -1
                    push!(s1, c3[i,i,i])
                end
                push!(A_list, A)
                push!(B_list, B)
                push!(c_list, c)
                push!(d_list, d)
                push!(μ_list, μ)
                push!(ν_list, ν)
                push!(Ξ_list, Ξ)
                push!(ω_list, ω)
                push!(γ_list, γ)
            end
        end
    end
    # second degree terms
    for i in 1:n
        for j in i:n
            A = zeros(n,n,n)
            B = zeros(n,n,n)
            c = zeros(n,n)
            d = zeros(n,n)
            μ = zeros(n)
            ν = zeros(n)
            Ξ = zeros(n,n)
            ω = zeros(n)
            γ = 0
            if i != j
                c[j,i] = 1
                c[i,j] = 1
                d[j,i] = -1
                d[i,j] = -1
                for l in 1:n
                    B[i,j,l] = 1
                    B[j,i,l] = 1
                end
                Ξ[i,j] = 1
                Ξ[j,i] = 1
                push!(s1, c2[i,j]+c2[j,i])
            else
                c[i,i] = 1
                d[i,i] = -1
                for l in 1:n
                    B[i,i,l] = 1
                end
                Ξ[i,i] = 1
                push!(s1, c2[i,i])
            end
            push!(A_list, A)
            push!(B_list, B)
            push!(c_list, c)
            push!(d_list, d)
            push!(μ_list, μ)
            push!(ν_list, ν)
            push!(Ξ_list, Ξ)
            push!(ω_list, ω)
            push!(γ_list, γ)
        end
    end
    # first degree terms
    for i in 1:n
        A = zeros(n,n,n)
        B = zeros(n,n,n)
        c = zeros(n,n)
        d = zeros(n,n)
        μ = zeros(n)
        ν = zeros(n)
        Ξ = zeros(n,n)
        ω = zeros(n)
        γ = 0
        for l in 1:n
            d[i,l] = 1
        end
        μ[i] = 1
        ν[i] = -1
        ω[i] = 1
        push!(A_list, A)
        push!(B_list, B)
        push!(c_list, c)
        push!(d_list, d)
        push!(μ_list, μ)
        push!(ν_list, ν)
        push!(Ξ_list, Ξ)
        push!(ω_list, ω)
        push!(γ_list, γ)
        push!(s1, c1[i])
    end
    # zero degree terms
    A = zeros(n,n,n)
    B = zeros(n,n,n)
    c = zeros(n,n)
    d = zeros(n,n)
    μ = zeros(n)
    ν = zeros(n)
    Ξ = zeros(n,n)
    ω = zeros(n)
    γ = 1
    for l in 1:n
        ν[l] = 1
    end
    push!(A_list, A)
    push!(B_list, B)
    push!(c_list, c)
    push!(d_list, d)
    push!(μ_list, μ)
    push!(ν_list, ν)
    push!(Ξ_list, Ξ)
    push!(ω_list, ω)
    push!(γ_list, γ)
    push!(s1, c0)

    unc_set_list = [A_list, B_list, c_list, d_list, μ_list, ν_list,
                    Ξ_list, ω_list, γ_list, s1]
    return unc_set_list
end

function solve_rpt_relaxation_X1_best_slc(unc_set_list,C,d,use_lmi)
    L, n = size(C)
    A_list, B_list, c_list = unc_set_list[1], unc_set_list[2], unc_set_list[3]
    d_list, μ_list, ν_list = unc_set_list[4], unc_set_list[5], unc_set_list[6]
    Ξ_list, ω_list, γ_list = unc_set_list[7], unc_set_list[8], unc_set_list[9]
    s1 = unc_set_list[10]
    L1 = length(s1)

    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, Y[1:n,1:n,1:n+1])
    @variable(m1, V[1:n,1:n,1:n])
    @variable(m1, λ[1:L1])

    @constraint(m1, [i in 1:n], -Y[:,:,i] .- sum(λ[j]*A_list[j][:,:,i] for j in 1:L1) in PSDCone())
    @constraint(m1, [i in 1:n], -V[:,:,i] .- sum(λ[j]*B_list[j][:,:,i] for j in 1:L1) in PSDCone())
    @constraint(m1, -Y[:,:,n+1] .- sum(λ[j]*Ξ_list[j] for j in 1:L1) in PSDCone())

    @constraint(m1, [i in 1:n], X[:,i] .+ sum(λ[j]*c_list[j][:,i] for j in 1:L1) .== 0)
    @constraint(m1, [i in 1:n], x .- X[:,i] .+ sum(λ[j]*d_list[j][:,i] for j in 1:L1) .== 0)
    @constraint(m1, x .+ sum(λ[j]*ω_list[j] for j in 1:L1) .== 0)

    @constraint(m1, [i in 1:n], x[i] + sum(λ[j]*μ_list[j][i] for j in 1:L1) == 0)
    @constraint(m1, [i in 1:n], 1 - x[i] + sum(λ[j]*ν_list[j][i] for j in 1:L1) == 0)
    @constraint(m1, 1 + sum(λ[j]*γ_list[j] for j in 1:L1) == 0)

    @constraint(m1, [i in 1:n], [Y[:,:,i]  X[:,i]; (X[:,i])' x[i]] in PSDCone())
    @constraint(m1, [i in 1:n], [V[:,:,i]  (x.-X[:,i]); (x.-X[:,i])' (1-x[i])] in PSDCone())
    @constraint(m1, [Y[:,:,n+1] x; x' 1] in PSDCone())

    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], C*X[:,i] .<= x[i]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    if use_lmi
        @constraint(m1, [X  x;  x' 1] in PSDCone())
    end
    @objective(m1, Min, -sum(s1[j]*λ[j] for j in 1:L1))
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL
        return true, JuMP.value.(x), JuMP.value.(X), objective_value(m1)
    else
        return false, zeros(n), zeros(n,n), 1e6
    end
end

function get_uncertainty_set_z2(c3,c2,c1,c0)
    n = size(c1,1)

    ############ Equality constraints ######################

    unc_set_list_aux = get_uncertainty_set_z1(c3,c2,c1,c0)
    A_list, B_list, c_list = unc_set_list_aux[1], unc_set_list_aux[2], unc_set_list_aux[3]
    d_list, μ_list, ν_list = unc_set_list_aux[4], unc_set_list_aux[5], unc_set_list_aux[6]
    Ξ_list, ω_list, γ_list = unc_set_list_aux[7], unc_set_list_aux[8], unc_set_list_aux[9]
    s1 = unc_set_list_aux[10]

    ############ Inequality constraints ######################
    A_hat_list = []
    B_hat_list = []
    c_hat_list = []
    d_hat_list = []
    μ_hat_list = []
    ν_hat_list = []
    Ξ_hat_list = []
    ω_hat_list = []
    γ_hat_list = []
    H_list = []
    G_list = []
    Ψ_list = []
    s2 = []
    # Constraints: P_{jj}^i >= sum_{k # j} T_{kj}^i, i,j in [n]
    #              Q_{jj}^i >= sum_{k # j} S_{kj}^i, i,j in [n]
    for i in 1:n
        for j in 1:n
            # P_{jj}^i >= sum_{k # j} T_{kj}^i
            A_hat = zeros(n,n,n)
            B_hat = zeros(n,n,n)
            c_hat = zeros(n,n)
            d_hat = zeros(n,n)
            μ_hat = zeros(n)
            ν_hat = zeros(n)
            Ξ_hat = zeros(n,n)
            ω_hat = zeros(n)
            γ_hat = 0
            H = zeros(n,n,n)
            G = zeros(n,n,n)
            Ψ = zeros(n,n)
            A_hat[j,j,i] = -1
            for k in 1:n
                if k != j
                    H[k,j,i] = 1
                end
            end
            push!(A_hat_list, A_hat)
            push!(B_hat_list, B_hat)
            push!(c_hat_list, c_hat)
            push!(d_hat_list, d_hat)
            push!(μ_hat_list, μ_hat)
            push!(ν_hat_list, ν_hat)
            push!(Ξ_hat_list, Ξ_hat)
            push!(ω_hat_list, ω_hat)
            push!(γ_hat_list, γ_hat)
            push!(H_list, H)
            push!(G_list, G)
            push!(Ψ_list, Ψ)
            push!(s2, 0)

            # Q_{jj}^i >= sum_{k # j} S_{kj}^i
            A_hat = zeros(n,n,n)
            B_hat = zeros(n,n,n)
            c_hat = zeros(n,n)
            d_hat = zeros(n,n)
            μ_hat = zeros(n)
            ν_hat = zeros(n)
            Ξ_hat = zeros(n,n)
            ω_hat = zeros(n)
            γ_hat = 0
            H = zeros(n,n,n)
            G = zeros(n,n,n)
            Ψ = zeros(n,n)
            B_hat[j,j,i] = -1
            for k in 1:n
                if k != j
                    G[k,j,i] = 1
                end
            end
            push!(A_hat_list, A_hat)
            push!(B_hat_list, B_hat)
            push!(c_hat_list, c_hat)
            push!(d_hat_list, d_hat)
            push!(μ_hat_list, μ_hat)
            push!(ν_hat_list, ν_hat)
            push!(Ξ_hat_list, Ξ_hat)
            push!(ω_hat_list, ω_hat)
            push!(γ_hat_list, γ_hat)
            push!(H_list, H)
            push!(G_list, G)
            push!(Ψ_list, Ψ)
            push!(s2, 0)
        end
    end
    #Constraint: P_{jj}' >= sum_{k # j} T_{kj}', j in [n]
    for j in 1:n
        A_hat = zeros(n,n,n)
        B_hat = zeros(n,n,n)
        c_hat = zeros(n,n)
        d_hat = zeros(n,n)
        μ_hat = zeros(n)
        ν_hat = zeros(n)
        Ξ_hat = zeros(n,n)
        ω_hat = zeros(n)
        γ_hat = 0
        H = zeros(n,n,n)
        G = zeros(n,n,n)
        Ψ = zeros(n,n)
        Ξ_hat[j,j] = -1
        for k in 1:n
            if k != j
                Ψ[k,j] = 1
            end
        end
        push!(A_hat_list, A_hat)
        push!(B_hat_list, B_hat)
        push!(c_hat_list, c_hat)
        push!(d_hat_list, d_hat)
        push!(μ_hat_list, μ_hat)
        push!(ν_hat_list, ν_hat)
        push!(Ξ_hat_list, Ξ_hat)
        push!(ω_hat_list, ω_hat)
        push!(γ_hat_list, γ_hat)
        push!(H_list, H)
        push!(G_list, G)
        push!(Ψ_list, Ψ)
        push!(s2, 0)
    end
    # Constraints: -T^i <= P^i <= T^i,  -S^i <= Q^i <= S^i
    for i in 1:n
        for j in 1:n
            for k in 1:n
                # P_{jk}^i <= T_{jk}^i
                A_hat = zeros(n,n,n)
                B_hat = zeros(n,n,n)
                c_hat = zeros(n,n)
                d_hat = zeros(n,n)
                μ_hat = zeros(n)
                ν_hat = zeros(n)
                Ξ_hat = zeros(n,n)
                ω_hat = zeros(n)
                γ_hat = 0
                H = zeros(n,n,n)
                G = zeros(n,n,n)
                Ψ = zeros(n,n)
                A_hat[j,k,i] = 1
                H[j,k,i] = -1
                push!(A_hat_list, A_hat)
                push!(B_hat_list, B_hat)
                push!(c_hat_list, c_hat)
                push!(d_hat_list, d_hat)
                push!(μ_hat_list, μ_hat)
                push!(ν_hat_list, ν_hat)
                push!(Ξ_hat_list, Ξ_hat)
                push!(ω_hat_list, ω_hat)
                push!(γ_hat_list, γ_hat)
                push!(H_list, H)
                push!(G_list, G)
                push!(Ψ_list, Ψ)
                push!(s2, 0)

                # -T_{jk}^i <= P_{jk}^i
                A_hat = zeros(n,n,n)
                B_hat = zeros(n,n,n)
                c_hat = zeros(n,n)
                d_hat = zeros(n,n)
                μ_hat = zeros(n)
                ν_hat = zeros(n)
                Ξ_hat = zeros(n,n)
                ω_hat = zeros(n)
                γ_hat = 0
                H = zeros(n,n,n)
                G = zeros(n,n,n)
                Ψ = zeros(n,n)
                A_hat[j,k,i] = -1
                H[j,k,i] = -1
                push!(A_hat_list, A_hat)
                push!(B_hat_list, B_hat)
                push!(c_hat_list, c_hat)
                push!(d_hat_list, d_hat)
                push!(μ_hat_list, μ_hat)
                push!(ν_hat_list, ν_hat)
                push!(Ξ_hat_list, Ξ_hat)
                push!(ω_hat_list, ω_hat)
                push!(γ_hat_list, γ_hat)
                push!(H_list, H)
                push!(G_list, G)
                push!(Ψ_list, Ψ)
                push!(s2, 0)

                # Q_{jk}^i <= S_{jk}^i
                A_hat = zeros(n,n,n)
                B_hat = zeros(n,n,n)
                c_hat = zeros(n,n)
                d_hat = zeros(n,n)
                μ_hat = zeros(n)
                ν_hat = zeros(n)
                Ξ_hat = zeros(n,n)
                ω_hat = zeros(n)
                γ_hat = 0
                H = zeros(n,n,n)
                G = zeros(n,n,n)
                Ψ = zeros(n,n)
                B_hat[j,k,i] = 1
                G[j,k,i] = -1
                push!(A_hat_list, A_hat)
                push!(B_hat_list, B_hat)
                push!(c_hat_list, c_hat)
                push!(d_hat_list, d_hat)
                push!(μ_hat_list, μ_hat)
                push!(ν_hat_list, ν_hat)
                push!(Ξ_hat_list, Ξ_hat)
                push!(ω_hat_list, ω_hat)
                push!(γ_hat_list, γ_hat)
                push!(H_list, H)
                push!(G_list, G)
                push!(Ψ_list, Ψ)
                push!(s2, 0)

                # -S_{jk}^i <= Q_{jk}^i
                A_hat = zeros(n,n,n)
                B_hat = zeros(n,n,n)
                c_hat = zeros(n,n)
                d_hat = zeros(n,n)
                μ_hat = zeros(n)
                ν_hat = zeros(n)
                Ξ_hat = zeros(n,n)
                ω_hat = zeros(n)
                γ_hat = 0
                H = zeros(n,n,n)
                G = zeros(n,n,n)
                Ψ = zeros(n,n)
                B_hat[j,k,i] = -1
                G[j,k,i] = -1
                push!(A_hat_list, A_hat)
                push!(B_hat_list, B_hat)
                push!(c_hat_list, c_hat)
                push!(d_hat_list, d_hat)
                push!(μ_hat_list, μ_hat)
                push!(ν_hat_list, ν_hat)
                push!(Ξ_hat_list, Ξ_hat)
                push!(ω_hat_list, ω_hat)
                push!(γ_hat_list, γ_hat)
                push!(H_list, H)
                push!(G_list, G)
                push!(Ψ_list, Ψ)
                push!(s2, 0)

            end
        end
    end
    # Constraints: -T' <= P' <= T'
    for j in 1:n
        for k in 1:n
            # P_{jk}' <= T_{jk}'
            A_hat = zeros(n,n,n)
            B_hat = zeros(n,n,n)
            c_hat = zeros(n,n)
            d_hat = zeros(n,n)
            μ_hat = zeros(n)
            ν_hat = zeros(n)
            Ξ_hat = zeros(n,n)
            ω_hat = zeros(n)
            γ_hat = 0
            H = zeros(n,n,n)
            G = zeros(n,n,n)
            Ψ = zeros(n,n)
            Ξ_hat[j,k] = 1
            Ψ[j,k] = -1
            push!(A_hat_list, A_hat)
            push!(B_hat_list, B_hat)
            push!(c_hat_list, c_hat)
            push!(d_hat_list, d_hat)
            push!(μ_hat_list, μ_hat)
            push!(ν_hat_list, ν_hat)
            push!(Ξ_hat_list, Ξ_hat)
            push!(ω_hat_list, ω_hat)
            push!(γ_hat_list, γ_hat)
            push!(H_list, H)
            push!(G_list, G)
            push!(Ψ_list, Ψ)
            push!(s2, 0)

            # -T_{jk}' <= P_{jk}'
            A_hat = zeros(n,n,n)
            B_hat = zeros(n,n,n)
            c_hat = zeros(n,n)
            d_hat = zeros(n,n)
            μ_hat = zeros(n)
            ν_hat = zeros(n)
            Ξ_hat = zeros(n,n)
            ω_hat = zeros(n)
            γ_hat = 0
            H = zeros(n,n,n)
            G = zeros(n,n,n)
            Ψ = zeros(n,n)
            Ξ_hat[j,k] = -1
            Ψ[j,k] = -1
            push!(A_hat_list, A_hat)
            push!(B_hat_list, B_hat)
            push!(c_hat_list, c_hat)
            push!(d_hat_list, d_hat)
            push!(μ_hat_list, μ_hat)
            push!(ν_hat_list, ν_hat)
            push!(Ξ_hat_list, Ξ_hat)
            push!(ω_hat_list, ω_hat)
            push!(γ_hat_list, γ_hat)
            push!(H_list, H)
            push!(G_list, G)
            push!(Ψ_list, Ψ)
            push!(s2, 0)
        end
    end


    eq_cons_list = [A_list, B_list, c_list, d_list, μ_list, ν_list, Ξ_list, ω_list, γ_list, s1]
    ineq_cons_list = [A_hat_list, B_hat_list, c_hat_list, d_hat_list, μ_hat_list, ν_hat_list,
                      Ξ_hat_list, ω_hat_list, γ_hat_list, H_list, G_list, Ψ_list, s2]
    unc_set_list = [eq_cons_list, ineq_cons_list]
    return unc_set_list
end

function solve_rpt_relaxation_X1_best_slc_gersh(unc_set_list,C,d,use_lmi)
    L, n = size(C)
    eq_cons_list, ineq_cons_list = unc_set_list[1], unc_set_list[2]

    A_list, B_list, c_list = eq_cons_list[1], eq_cons_list[2], eq_cons_list[3]
    d_list, μ_list, ν_list = eq_cons_list[4], eq_cons_list[5], eq_cons_list[6]
    Ξ_list, ω_list, γ_list = eq_cons_list[7], eq_cons_list[8], eq_cons_list[9]
    s1 = eq_cons_list[10]

    A_hat_list, B_hat_list, c_hat_list = ineq_cons_list[1], ineq_cons_list[2], ineq_cons_list[3]
    d_hat_list, μ_hat_list, ν_hat_list = ineq_cons_list[4], ineq_cons_list[5], ineq_cons_list[6]
    Ξ_hat_list, ω_hat_list, γ_hat_list = ineq_cons_list[7], ineq_cons_list[8], ineq_cons_list[9]
    H_list, G_list, Ψ_list = ineq_cons_list[10], ineq_cons_list[11], ineq_cons_list[12]
    s2 = ineq_cons_list[13]

    L1 = length(s1)
    L2 = length(s2)

    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, Y[1:n,1:n,1:n+1])
    @variable(m1, V[1:n,1:n,1:n])
    @variable(m1, λ[1:L1])
    @variable(m1, θ[1:L2]>=0)

    @constraint(m1, [i in 1:n], Y[:,:,i] .+ sum(λ[j]*A_list[j][:,:,i] for j in 1:L1) .- sum(θ[j]*A_hat_list[j][:,:,i] for j in 1:L2) .== 0)
    @constraint(m1, [i in 1:n], V[:,:,i] .+ sum(λ[j]*B_list[j][:,:,i] for j in 1:L1) .- sum(θ[j]*B_hat_list[j][:,:,i] for j in 1:L2) .== 0)
    @constraint(m1, Y[:,:,n+1] .+ sum(λ[j]*Ξ_list[j] for j in 1:L1) .- sum(θ[j]*Ξ_hat_list[j] for j in 1:L1) .== 0)

    @constraint(m1, [i in 1:n], X[:,i] .+ sum(λ[j]*c_list[j][:,i] for j in 1:L1) .- sum(θ[j]*c_hat_list[j][:,i] for j in 1:L2) .== 0)
    @constraint(m1, [i in 1:n], x .- X[:,i] .+ sum(λ[j]*d_list[j][:,i] for j in 1:L1) .- sum(θ[j]*d_hat_list[j][:,i] for j in 1:L2) .== 0)
    @constraint(m1, x .+ sum(λ[j]*ω_list[j] for j in 1:L1) - sum(θ[j]*ω_hat_list[j] for j in 1:L2) .== 0)

    @constraint(m1, [i in 1:n], x[i] + sum(λ[j]*μ_list[j][i] for j in 1:L1) - sum(θ[j]*μ_hat_list[j][i] for j in 1:L2) == 0)
    @constraint(m1, [i in 1:n], 1 - x[i] + sum(λ[j]*ν_list[j][i] for j in 1:L1) - sum(λ[j]*ν_hat_list[j][i] for j in 1:L2) == 0)
    @constraint(m1, 1 + sum(λ[j]*γ_list[j] for j in 1:L1) - sum(θ[j]*γ_hat_list[j] for j in 1:L2) == 0)

    @constraint(m1, [i in 1:n], sum(θ[j]*H_list[j][:,:,i] for j in 1:L2) .== 0)
    @constraint(m1, [i in 1:n], sum(θ[j]*G_list[j][:,:,i] for j in 1:L2) .== 0)
    @constraint(m1, sum(θ[j]*Ψ_list[j] for j in 1:L2) .== 0)

    @constraint(m1, [i in 1:n], [Y[:,:,i]  X[:,i]; (X[:,i])' x[i]] in PSDCone())
    @constraint(m1, [i in 1:n], [V[:,:,i]  (x.-X[:,i]); (x.-X[:,i])' (1-x[i])] in PSDCone())
    @constraint(m1, [Y[:,:,n+1] x; x' 1] in PSDCone())

    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], C*X[:,i] .<= x[i]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    if use_lmi
        @constraint(m1, [X  x;  x' 1] in PSDCone())
    end
    @objective(m1, Min, sum(s2[j]*θ[j] for j in 1:L2) - sum(s1[j]*λ[j] for j in 1:L1))
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL
        return true, JuMP.value.(x), JuMP.value.(X), objective_value(m1)
    else
        return false, zeros(n), zeros(n,n), 1e6
    end
end

function ub_ipopt_X1(C,d,c3,c2,c1,c0,x0)
    n = size(c1,1)
    m1 = Model(Ipopt.Optimizer)
    @variable(m1, x[1:n]>=0)
    for j in 1:n
        JuMP.set_start_value(x[j], x0[j])
    end
    @constraint(m1, C*x .<= d)
    # Objective function
    @NLexpression(m1, obj_term, sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n) +
                                sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n) +
                                sum(c1[i]*x[i] for i=1:n) + c0)
    @NLobjective(m1, Min, obj_term)
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.LOCALLY_SOLVED
        return objective_value(m1)
    else
        return 1e6
    end
end

function get_ub_X1(X,C,d,c3,c2,c1,c0)
    best_ub = 1e6
    for i in 1:length(X)
        x = X[i]
        if is_point_feasible(C,d,x)
            cur_ub_1 = poly3_init(c3,c2,c1,c0,x)
            cur_ub_2 = ub_ipopt_X1(C,d,c3,c2,c1,c0,x)
            best_ub = min(best_ub,cur_ub_1,cur_ub_2)
        end
    end
    return best_ub
end

function rpt_bb_X1(C_init,d_init,c3,c2,c1,c0,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    unc_set_list = get_uncertainty_set_z1(c3,c2,c1,c0)

    # Root Node
    res, x_lb, X_lb, lb = solve_rpt_relaxation_X1_best_slc(unc_set_list,C,d,use_lmi)
    X = calculate_candidate_vectors(x_lb,X_lb)
    ub = get_ub_X1(X,C,d,c3,c2,c1,c0)

    x_cur, X_cur = x_lb, X_lb

    UB, LB, opt_sol, opt_val = ub, lb, x_lb, ub

    nodes_list = []
    ub_list, lb_list, hyp_list = [], [], []
    push!(ub_list, ub)
    push!(lb_list, lb)

    t0_1 = time_ns()
    total_time = 0.0

    while UB - LB > δ && total_time < 3600
        res, f_opt, l_opt = generate_hyperplane_eigen(x_cur,X_cur)

        f_r, l_r, f_l, l_l  = f_opt, l_opt, -f_opt, -l_opt
        C_r, d_r, C_l, d_l = vcat(C,f_r'), vcat(d,l_r), vcat(C,f_l'), vcat(d,l_l)
        gen_hyper += 1
        push!(hyp_list, [f_opt, l_opt])

        # Right child
        res_r, x_lb_r, X_lb_r, lb_r = solve_rpt_relaxation_X1_best_slc(unc_set_list,C_r,d_r,use_lmi)
        if res_r
            X_r = calculate_candidate_vectors(x_lb_r,X_lb_r)
            ub_r = get_ub_X1(X_r,C_r,d_r,c3,c2,c1,c0)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r])
            end
        end
        # Left child
        res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_X1_best_slc(unc_set_list,C_l,d_l,use_lmi)
        if res_l
            X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
            ub_l = get_ub_X1(X_l,C_l,d_l,c3,c2,c1,c0)
            if lb_l < UB
                push!(nodes_list,[ub_l,lb_l,x_lb_l,X_lb_l,C_l,d_l])
            end
        end

        if isempty(nodes_list)
            if res_r
                push!(ub_list, ub_r)
                push!(lb_list, lb_r)
            end
            if res_l
                push!(ub_list, ub_l)
                push!(lb_list, lb_l)
            end
            break
        else
            ind = argmin([nodes_list[i][2] for i in 1:length(nodes_list)])
            cur_node = nodes_list[ind]
            deleteat!(nodes_list, ind)
            ub, lb = cur_node[1], cur_node[2]
            x_cur, X_cur = cur_node[3], cur_node[4]
            C, d = cur_node[5], cur_node[6]
            LB = lb
            if ub < UB
                UB = ub
                opt_sol, opt_val = cur_node[3], cur_node[1]
            end
            push!(ub_list, ub)
            push!(lb_list, lb)
        end
        t0_2 = time_ns()
        total_time = (t0_2-t0_1)*10^(-9)
    end
    return opt_sol, opt_val, gen_hyper, ub_list, lb_list, hyp_list
end

function rpt_bb_X1_gersh(C_init,d_init,c3,c2,c1,c0,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    unc_set_list = get_uncertainty_set_z2(c3,c2,c1,c0)

    # Root Node
    res, x_lb, X_lb, lb = solve_rpt_relaxation_X1_best_slc_gersh(unc_set_list,C,d,use_lmi)
    X = calculate_candidate_vectors(x_lb,X_lb)
    ub = get_ub_X1(X,C,d,c3,c2,c1,c0)

    x_cur, X_cur = x_lb, X_lb

    UB, LB, opt_sol, opt_val = ub, lb, x_lb, ub

    nodes_list = []
    ub_list, lb_list, hyp_list = [], [], []
    push!(ub_list, ub)
    push!(lb_list, lb)

    t0_1 = time_ns()
    total_time = 0.0

    while UB - LB > δ && total_time < 3600
        res, f_opt, l_opt = generate_hyperplane_eigen(x_cur,X_cur)

        f_r, l_r, f_l, l_l  = f_opt, l_opt, -f_opt, -l_opt
        C_r, d_r, C_l, d_l = vcat(C,f_r'), vcat(d,l_r), vcat(C,f_l'), vcat(d,l_l)
        gen_hyper += 1
        push!(hyp_list, [f_opt, l_opt])

        # Right child
        res_r, x_lb_r, X_lb_r, lb_r = solve_rpt_relaxation_X1_best_slc_gersh(unc_set_list,C_r,d_r,use_lmi)
        if res_r
            X_r = calculate_candidate_vectors(x_lb_r,X_lb_r)
            ub_r = get_ub_X1(X_r,C_r,d_r,c3,c2,c1,c0)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r])
            end
        end
        # Left child
        res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_X1_best_slc_gersh(unc_set_list,C_l,d_l,use_lmi)
        if res_l
            X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
            ub_l = get_ub_X1(X_l,C_l,d_l,c3,c2,c1,c0)
            if lb_l < UB
                push!(nodes_list,[ub_l,lb_l,x_lb_l,X_lb_l,C_l,d_l])
            end
        end

        if isempty(nodes_list)
            if res_r
                push!(ub_list, ub_r)
                push!(lb_list, lb_r)
            end
            if res_l
                push!(ub_list, ub_l)
                push!(lb_list, lb_l)
            end
            break
        else
            ind = argmin([nodes_list[i][2] for i in 1:length(nodes_list)])
            cur_node = nodes_list[ind]
            deleteat!(nodes_list, ind)
            ub, lb = cur_node[1], cur_node[2]
            x_cur, X_cur = cur_node[3], cur_node[4]
            C, d = cur_node[5], cur_node[6]
            LB = lb
            if ub < UB
                UB = ub
                opt_sol, opt_val = cur_node[3], cur_node[1]
            end
            push!(ub_list, ub)
            push!(lb_list, lb)
        end
        t0_2 = time_ns()
        total_time = (t0_2-t0_1)*10^(-9)
    end
    return opt_sol, opt_val, gen_hyper, ub_list, lb_list, hyp_list
end

############################     X_2      ############################

function is_point_feasible_X2(C,d,p_2_coeffs,x)
    c3_2, c2_2, c1_2, c0_2 = p_2_coeffs[1], p_2_coeffs[2], p_2_coeffs[3], p_2_coeffs[4]
    res = (sum(C*x .<= d)  == size(C,1) && poly3_init(c3_2,c2_2,c1_2,c0_2,x) <= 0)
    return res
end

function solve_rpt_relaxation_X2_random_slc(C,d,p2_list_1,q2_list_1,p2_list_2,q2_list_2,use_lmi)
    L, n = size(C)
    P_list_1, r_list_1, w_list_1 = p2_list_1[1], p2_list_1[2], p2_list_1[3]
    Q_list_1, f_list_1, g_list_1 = q2_list_1[1], q2_list_1[2], q2_list_1[3]
    P_list_2, r_list_2, w_list_2 = p2_list_2[1], p2_list_2[2], p2_list_2[3]
    Q_list_2, f_list_2, g_list_2 = q2_list_2[1], q2_list_2[2], q2_list_2[3]

    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, t1[1:n])
    @variable(m1, v1[1:n])
    @variable(m1, t2[1:n])
    @variable(m1, v2[1:n])

    @constraint(m1, [i in 1:n], [t1[i]+x[i];
                                vcat(2*P_list_1[i]^(0.5)*X[:,i], t1[i]-x[i])]
                                in SecondOrderCone())

    @constraint(m1, [i in 1:n], [v1[i]+1-x[i];
                                vcat(2*Q_list_1[i]^(0.5)*(x.-X[:,i]), v1[i]-1+x[i])]
                                in SecondOrderCone())

    @constraint(m1, [i in 1:n], [t2[i]+x[i];
                                vcat(2*P_list_2[i]^(0.5)*X[:,i], t2[i]-x[i])]
                                in SecondOrderCone())

    @constraint(m1, [i in 1:n], [v2[i]+1-x[i];
                                vcat(2*Q_list_2[i]^(0.5)*(x.-X[:,i]), v2[i]-1+x[i])]
                                in SecondOrderCone())

    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], C*X[:,i] .<= x[i]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')

    @constraint(m1, sum(t2[i] + r_list_2[i]'*X[:,i] + w_list_2[i]*x[i] +
                        v2[i] + f_list_2[i]'*(x .- X[:,i]) + g_list_2[i]*(1 - x[i])
                        for i in 1:n) <= 0)

    if use_lmi
        @constraint(m1, [X  x;  x' 1] in PSDCone())
    end
    @objective(m1, Min, sum(t1[i] + r_list_1[i]'*X[:,i] + w_list_1[i]*x[i] +
                            v1[i] + f_list_1[i]'*(x .- X[:,i]) + g_list_1[i]*(1 - x[i]) for i in 1:n))
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL
        return true, JuMP.value.(x), JuMP.value.(X), objective_value(m1)
    else
        return false, zeros(n), zeros(n,n), 1e6
    end
end

function solve_rpt_relaxation_X2_best_slc(unc_set_list_1,unc_set_list_2,C,d,use_lmi)
    L, n = size(C)
    A_1_list, B_1_list, c_1_list = unc_set_list_1[1], unc_set_list_1[2], unc_set_list_1[3]
    d_1_list, μ_1_list, ν_1_list = unc_set_list_1[4], unc_set_list_1[5], unc_set_list_1[6]
    Ξ_1_list, ω_1_list, γ_1_list = unc_set_list_1[7], unc_set_list_1[8], unc_set_list_1[9]
    s1_1 = unc_set_list_1[10]
    L1_1 = length(s1_1)
    A_2_list, B_2_list, c_2_list = unc_set_list_2[1], unc_set_list_2[2], unc_set_list_2[3]
    d_2_list, μ_2_list, ν_2_list = unc_set_list_2[4], unc_set_list_2[5], unc_set_list_2[6]
    Ξ_2_list, ω_2_list, γ_2_list = unc_set_list_2[7], unc_set_list_2[8], unc_set_list_2[9]
    s1_2 = unc_set_list_2[10]
    L1_2 = length(s1_2)

    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, Y_1[1:n,1:n,1:n+1])
    @variable(m1, V_1[1:n,1:n,1:n])
    @variable(m1, λ_1[1:L1_1])
    @variable(m1, Y_2[1:n,1:n,1:n+1])
    @variable(m1, V_2[1:n,1:n,1:n])
    @variable(m1, λ_2[1:L1_2])

    @constraint(m1, [i in 1:n], -Y_1[:,:,i] .- sum(λ_1[j]*A_1_list[j][:,:,i] for j in 1:L1_1) in PSDCone())
    @constraint(m1, [i in 1:n], -V_1[:,:,i] .- sum(λ_1[j]*B_1_list[j][:,:,i] for j in 1:L1_1) in PSDCone())
    @constraint(m1, -Y_1[:,:,n+1] .- sum(λ_1[j]*Ξ_1_list[j] for j in 1:L1_1) in PSDCone())

    @constraint(m1, [i in 1:n], -Y_2[:,:,i] .- sum(λ_2[j]*A_2_list[j][:,:,i] for j in 1:L1_2) in PSDCone())
    @constraint(m1, [i in 1:n], -V_2[:,:,i] .- sum(λ_2[j]*B_2_list[j][:,:,i] for j in 1:L1_2) in PSDCone())
    @constraint(m1, -Y_2[:,:,n+1] .- sum(λ_2[j]*Ξ_2_list[j] for j in 1:L1_2) in PSDCone())

    @constraint(m1, [i in 1:n], X[:,i] .+ sum(λ_1[j]*c_1_list[j][:,i] for j in 1:L1_1) .== 0)
    @constraint(m1, [i in 1:n], x .- X[:,i] .+ sum(λ_1[j]*d_1_list[j][:,i] for j in 1:L1_1) .== 0)
    @constraint(m1, x .+ sum(λ_1[j]*ω_1_list[j] for j in 1:L1_1) .== 0)

    @constraint(m1, [i in 1:n], X[:,i] .+ sum(λ_2[j]*c_2_list[j][:,i] for j in 1:L1_2) .== 0)
    @constraint(m1, [i in 1:n], x .- X[:,i] .+ sum(λ_2[j]*d_2_list[j][:,i] for j in 1:L1_2) .== 0)
    @constraint(m1, x .+ sum(λ_2[j]*ω_2_list[j] for j in 1:L1_2) .== 0)

    @constraint(m1, [i in 1:n], x[i] + sum(λ_1[j]*μ_1_list[j][i] for j in 1:L1_1) == 0)
    @constraint(m1, [i in 1:n], 1 - x[i] + sum(λ_1[j]*ν_1_list[j][i] for j in 1:L1_1) == 0)
    @constraint(m1, 1 + sum(λ_1[j]*γ_1_list[j] for j in 1:L1_1) == 0)

    @constraint(m1, [i in 1:n], x[i] + sum(λ_2[j]*μ_2_list[j][i] for j in 1:L1_2) == 0)
    @constraint(m1, [i in 1:n], 1 - x[i] + sum(λ_2[j]*ν_2_list[j][i] for j in 1:L1_2) == 0)
    @constraint(m1, 1 + sum(λ_2[j]*γ_2_list[j] for j in 1:L1_2) == 0)

    @constraint(m1, [i in 1:n], [Y_1[:,:,i]  X[:,i]; (X[:,i])' x[i]] in PSDCone())
    @constraint(m1, [i in 1:n], [V_1[:,:,i]  (x.-X[:,i]); (x.-X[:,i])' (1-x[i])] in PSDCone())
    @constraint(m1, [Y_1[:,:,n+1] x; x' 1] in PSDCone())
    @constraint(m1, [i in 1:n], [Y_2[:,:,i]  X[:,i]; (X[:,i])' x[i]] in PSDCone())
    @constraint(m1, [i in 1:n], [V_2[:,:,i]  (x.-X[:,i]); (x.-X[:,i])' (1-x[i])] in PSDCone())
    @constraint(m1, [Y_2[:,:,n+1] x; x' 1] in PSDCone())

    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], C*X[:,i] .<= x[i]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    if use_lmi
        @constraint(m1, [X  x;  x' 1] in PSDCone())
    end
    @constraint(m1, -sum(s1_2[j]*λ_2[j] for j in 1:L1_2) <= 0)
    @objective(m1, Min, -sum(s1_1[j]*λ_1[j] for j in 1:L1_1))
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL
        return true, JuMP.value.(x), JuMP.value.(X), objective_value(m1)
    else
        return false, zeros(n), zeros(n,n), 1e6
    end
end

function ub_ipopt_X2(C,d,p_1_coeffs,p_2_coeffs,x0)
    n = size(C,2)
    c3_1, c2_1, c1_1, c0_1 = p_1_coeffs[1], p_1_coeffs[2], p_1_coeffs[3], p_1_coeffs[4]
    c3_2, c2_2, c1_2, c0_2 = p_2_coeffs[1], p_2_coeffs[2], p_2_coeffs[3], p_2_coeffs[4]
    m1 = Model(Ipopt.Optimizer)
    @variable(m1, x[1:n])
    # for j in 1:n
    #     JuMP.set_start_value(x[j], x0[j])
    # end
    # Constraints
    @constraint(m1, C*x .<= d)
    @NLconstraint(m1, sum(c3_2[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n) +
                      sum(c2_2[i,j]*x[i]*x[j] for i=1:n, j=1:n) +
                      sum(c1_2[i]*x[i] for i=1:n) + c0_2 <= 0)
    # Objective function
    @NLexpression(m1, obj_term, sum(c3_1[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n) +
                                sum(c2_1[i,j]*x[i]*x[j] for i=1:n, j=1:n) +
                                sum(c1_1[i]*x[i] for i=1:n) + c0_1)
    @NLobjective(m1, Min, obj_term)
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.LOCALLY_SOLVED
        return objective_value(m1)
    else
        return 1e6
    end
end

function get_ub_X2(X,C,d,p_1_coeffs,p_2_coeffs)
    c3_1, c2_1, c1_1, c0_1 = p_1_coeffs[1], p_1_coeffs[2], p_1_coeffs[3], p_1_coeffs[4]
    best_ub = 1e6
    for i in 1:length(X)
        x = X[i]
        if is_point_feasible_X2(C,d,p_2_coeffs,x)
            cur_ub = poly3_init(c3_1,c2_1,c1_1,c0_1,x)
            if cur_ub < best_ub
                best_ub = cur_ub
            end
        end
        # cur_ub = ub_ipopt_X3(C,d,p_1_coeffs,p_2_coeffs,x)
    end
    return best_ub
end

function rpt_bb_X2(C_init,d_init,p_1_coeffs,p_2_coeffs,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    c3_1, c2_1, c1_1, c0_1 = p_1_coeffs[1], p_1_coeffs[2], p_1_coeffs[3], p_1_coeffs[4]
    c3_2, c2_2, c1_2, c0_2 = p_2_coeffs[1], p_2_coeffs[2], p_2_coeffs[3], p_2_coeffs[4]

    unc_set_list_1 = get_uncertainty_set_z1(c3_1,c2_1,c1_1,c0_1)
    unc_set_list_2 = get_uncertainty_set_z1(c3_2,c2_2,c1_2,c0_2)

    # Root Node
    res, x_lb, X_lb, lb = solve_rpt_relaxation_X2_best_slc(unc_set_list_1,unc_set_list_2,C,d,use_lmi)
    X = calculate_candidate_vectors(x_lb,X_lb)
    ub = get_ub_X2(X,C,d,p_1_coeffs,p_2_coeffs)

    x_cur, X_cur = x_lb, X_lb

    UB, LB, opt_sol, opt_val = ub, lb, x_lb, ub

    nodes_list = []
    ub_list, lb_list, hyp_list = [], [], []
    push!(ub_list, ub)
    push!(lb_list, lb)

    t0_1 = time_ns()
    total_time = 0.0

    while UB - LB > δ && total_time < 3600
        res, f_opt, l_opt = generate_hyperplane_eigen(x_cur,X_cur)

        f_r, l_r, f_l, l_l  = f_opt, l_opt, -f_opt, -l_opt
        C_r, d_r, C_l, d_l = vcat(C,f_r'), vcat(d,l_r), vcat(C,f_l'), vcat(d,l_l)
        gen_hyper += 1
        push!(hyp_list, [f_opt, l_opt])

        # Right child
        res_r, x_lb_r, X_lb_r, lb_r = solve_rpt_relaxation_X2_best_slc(unc_set_list_1,unc_set_list_2,C_r,d_r,use_lmi)
        if res_r
            X_r = calculate_candidate_vectors(x_lb_r,X_lb_r)
            ub_r = get_ub_X2(X_r,C_r,d_r,p_1_coeffs,p_2_coeffs)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r])
            end
        end
        # Left child
        res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_X2_best_slc(unc_set_list_1,unc_set_list_2,C_l,d_l,use_lmi)
        if res_l
            X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
            ub_l = get_ub_X2(X_l,C_l,d_l,p_1_coeffs,p_2_coeffs)
            if lb_l < UB
                push!(nodes_list,[ub_l,lb_l,x_lb_l,X_lb_l,C_l,d_l])
            end
        end

        if isempty(nodes_list)
            if res_r
                push!(ub_list, ub_r)
                push!(lb_list, lb_r)
            end
            if res_l
                push!(ub_list, ub_l)
                push!(lb_list, lb_l)
            end
            break
        else
            ind = argmin([nodes_list[i][2] for i in 1:length(nodes_list)])
            cur_node = nodes_list[ind]
            deleteat!(nodes_list, ind)
            ub, lb = cur_node[1], cur_node[2]
            x_cur, X_cur = cur_node[3], cur_node[4]
            C, d = cur_node[5], cur_node[6]
            LB = lb
            if ub < UB
                UB = ub
                opt_sol, opt_val = cur_node[3], cur_node[1]
            end
            push!(ub_list, ub)
            push!(lb_list, lb)
        end
        t0_2 = time_ns()
        total_time = (t0_2-t0_1)*10^(-9)
    end
    return opt_sol, opt_val, gen_hyper, ub_list, lb_list, hyp_list
end

#############################    X_3    #####################################

function is_point_feasible_X3(C,d,x,α)
    n = size(x,1)
    res = (sum(C*x .<= d)  == size(C,1)) && log(sum(exp(x[i]) for i=1:n)) <= α
    return res
end

function solve_rpt_relaxation_X3_best_slc(unc_set_list,C,d,use_lmi,α)
    L, n = size(C)
    A_list, B_list, c_list = unc_set_list[1], unc_set_list[2], unc_set_list[3]
    d_list, μ_list, ν_list = unc_set_list[4], unc_set_list[5], unc_set_list[6]
    Ξ_list, ω_list, γ_list = unc_set_list[7], unc_set_list[8], unc_set_list[9]
    s1 = unc_set_list[10]
    L1 = length(s1)

    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, Y[1:n,1:n,1:n+1])
    @variable(m1, V[1:n,1:n,1:n])
    @variable(m1, λ[1:L1])
    @variable(m1, z[1:n])
    @variable(m1, Z[1:n,1:n], Symmetric)
    @variable(m1, Q[1:n,1:n])

    @constraint(m1, [i in 1:n], -Y[:,:,i] .- sum(λ[j]*A_list[j][:,:,i] for j in 1:L1) in PSDCone())
    @constraint(m1, [i in 1:n], -V[:,:,i] .- sum(λ[j]*B_list[j][:,:,i] for j in 1:L1) in PSDCone())
    @constraint(m1, -Y[:,:,n+1] .- sum(λ[j]*Ξ_list[j] for j in 1:L1) in PSDCone())

    @constraint(m1, [i in 1:n], X[:,i] .+ sum(λ[j]*c_list[j][:,i] for j in 1:L1) .== 0)
    @constraint(m1, [i in 1:n], x .- X[:,i] .+ sum(λ[j]*d_list[j][:,i] for j in 1:L1) .== 0)
    @constraint(m1, x .+ sum(λ[j]*ω_list[j] for j in 1:L1) .== 0)

    @constraint(m1, [i in 1:n], x[i] + sum(λ[j]*μ_list[j][i] for j in 1:L1) == 0)
    @constraint(m1, [i in 1:n], 1 - x[i] + sum(λ[j]*ν_list[j][i] for j in 1:L1) == 0)
    @constraint(m1, 1 + sum(λ[j]*γ_list[j] for j in 1:L1) == 0)

    @constraint(m1, [i in 1:n], [Y[:,:,i]  X[:,i]; (X[:,i])' x[i]] in PSDCone())
    @constraint(m1, [i in 1:n], [V[:,:,i]  (x.-X[:,i]); (x.-X[:,i])' (1-x[i])] in PSDCone())
    @constraint(m1, [Y[:,:,n+1] x; x' 1] in PSDCone())

    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], C*X[:,i] .<= x[i]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    @constraint(m1, sum(z[i] for i in 1:n) <= 1)
    @constraint(m1, [i in 1:n], [x[i] - α, 1, z[i]] in MOI.ExponentialCone())
    @constraint(m1, C*x .- sum(C*Q[:,i] for i in 1:n) .<= d*(1-sum(z[i] for i in 1:n)))
    @constraint(m1, sum(Q[:,i] for i in 1:n) .<= x)
    @constraint(m1, 1 - 2*sum(z[i] for i in 1:n) + sum(Z[i,j] for i in 1:n for j in 1:n) >= 0)
    @constraint(m1, [i in 1:n], [x[i] - α - sum(Q[i,j] for j in 1:n) + α*sum(z[j] for j in 1:n),
                                 1 - sum(z[j] for j in 1:n),
                                 z[i]-sum(Z[j,i] for j in 1:n)] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in 1:n], [X[i,j] - α*x[j], x[j], Q[j,i]] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in 1:L], [d[j]*x[i]-α*d[j]-C[j,:]'*X[:,i]+α*C[j,:]'*x,
                                           d[j]-C[j,:]'*x,
                                           d[j]*z[i] - C[j,:]'*Q[:,i]] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in i:n], [x[i]+x[j]-2*α,1,Z[i,j]] in MOI.ExponentialCone())
    if use_lmi
        @constraint(m1, [X Q x; Q' Z z; x' z' 1] in PSDCone())
    end

    @objective(m1, Min, -sum(s1[j]*λ[j] for j in 1:L1))
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL
        return true, JuMP.value.(x), JuMP.value.(X), objective_value(m1)
    elseif termination_status(m1) == MOI.SLOW_PROGRESS
        x_opt, X_opt = JuMP.value.(x), JuMP.value.(X)
        if is_point_feasible_X3(C,d,x_opt,α)
            return true, x_opt, X_opt, objective_value(m1)
        else
            return false, zeros(n), zeros(n,n), 1e6
        end
    else
        return false, zeros(n), zeros(n,n), 1e6
    end
end

function ub_ipopt_X3(C,d,c3,c2,c1,c0,x0,α)
    n = size(c1,1)
    m1 = Model(Ipopt.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, t)
    for j in 1:n
        JuMP.set_start_value(x[j], x0[j])
    end
    @constraint(m1, C*x .<= d)
    # new constraint
    # @NLconstraint(m1, t == sum(exp(x[i]) for i in 1:n))
    # @NLconstraint(m1, log(t) <= α)
    @NLconstraint(m1, log(sum(exp(x[i]) for i in 1:n)) <= α)

    # Objective function
    @NLexpression(m1, obj_term, sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n) +
                                sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n) +
                                sum(c1[i]*x[i] for i=1:n) + c0)
    @NLobjective(m1, Min, obj_term)
    optimize!(m1)
    if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.LOCALLY_SOLVED
        return objective_value(m1)
    else
        return 1e6
    end
end

function get_ub_X3(X,C,d,c3,c2,c1,c0,α)
    best_ub = 1e6
    for i in 1:length(X)
        x = X[i]
        if is_point_feasible_X3(C,d,x,α)
            cur_ub_1 = poly3_init(c3,c2,c1,c0,x)
            cur_ub_2 = ub_ipopt_X3(C,d,c3,c2,c1,c0,x,α)
            best_ub = min(best_ub,cur_ub_1,cur_ub_2)
        end
    end
    return best_ub
end

function rpt_bb_X3(C_init,d_init,c3,c2,c1,c0,δ,use_lmi,α)
    C, d = C_init, d_init
    gen_hyper = 0

    unc_set_list = get_uncertainty_set_z1(c3,c2,c1,c0)

    # Root Node
    res, x_lb, X_lb, lb = solve_rpt_relaxation_X3_best_slc(unc_set_list,C,d,use_lmi,α)
    X = calculate_candidate_vectors(x_lb,X_lb)
    ub = get_ub_X3(X,C,d,c3,c2,c1,c0,α)

    x_cur, X_cur = x_lb, X_lb

    UB, LB, opt_sol, opt_val = ub, lb, x_lb, ub

    nodes_list = []
    ub_list, lb_list, hyp_list = [], [], []
    push!(ub_list, ub)
    push!(lb_list, lb)

    t0_1 = time_ns()
    total_time = 0.0

    while UB - LB > δ && total_time < 3600
        res, f_opt, l_opt = generate_hyperplane_eigen(x_cur,X_cur)

        f_r, l_r, f_l, l_l  = f_opt, l_opt, -f_opt, -l_opt
        C_r, d_r, C_l, d_l = vcat(C,f_r'), vcat(d,l_r), vcat(C,f_l'), vcat(d,l_l)
        gen_hyper += 1
        push!(hyp_list, [f_opt, l_opt])

        # Right child
        res_r, x_lb_r, X_lb_r, lb_r = solve_rpt_relaxation_X3_best_slc(unc_set_list,C_r,d_r,use_lmi,α)
        if res_r
            X_r = calculate_candidate_vectors(x_lb_r,X_lb_r)
            ub_r = get_ub_X3(X_r,C_r,d_r,c3,c2,c1,c0,α)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r])
            end
        end
        # Left child
        res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_X3_best_slc(unc_set_list,C_l,d_l,use_lmi,α)
        if res_l
            X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
            ub_l = get_ub_X3(X_l,C_l,d_l,c3,c2,c1,c0,α)
            if lb_l < UB
                push!(nodes_list,[ub_l,lb_l,x_lb_l,X_lb_l,C_l,d_l])
            end
        end

        if isempty(nodes_list)
            if res_r
                push!(ub_list, ub_r)
                push!(lb_list, lb_r)
            end
            if res_l
                push!(ub_list, ub_l)
                push!(lb_list, lb_l)
            end
            break
        else
            ind = argmin([nodes_list[i][2] for i in 1:length(nodes_list)])
            cur_node = nodes_list[ind]
            deleteat!(nodes_list, ind)
            ub, lb = cur_node[1], cur_node[2]
            x_cur, X_cur = cur_node[3], cur_node[4]
            C, d = cur_node[5], cur_node[6]
            LB = lb
            if ub < UB
                UB = ub
                opt_sol, opt_val = cur_node[3], cur_node[1]
            end
            push!(ub_list, ub)
            push!(lb_list, lb)
        end
        t0_2 = time_ns()
        total_time = (t0_2-t0_1)*10^(-9)
    end
    return opt_sol, opt_val, gen_hyper, ub_list, lb_list, hyp_list
end


δ = 1e-4
use_lmi = false

n = 30
c3, c2, c1, c0 = generate_degree3_polynomial(n, 0.5)
C, d = zeros(n,n) + I, ones(n)

# t1 = time_ns()
# p2_list, q2_list = get_poly3_slc_decomposition_X1_sdp(c3,c2,c1,c0)
# res_1, x_opt_1, X_opt_1, lb_1 = solve_rpt_relaxation_X1_random_slc(C,d,p2_list,q2_list,use_lmi)
# X = calculate_candidate_vectors(x_opt_1,X_opt_1)
# ub_1 = get_ub_X1(X,C,d,c3,c2,c1,c0)
# t2 = time_ns()
# total_time_1 = (t2-t1)*10^(-9)
#
# t1 = time_ns()
# unc_set_list = get_uncertainty_set_z1(c3,c2,c1,c0)
# res_2, x_opt_2, X_opt_2, lb_2 = solve_rpt_relaxation_X1_best_slc(unc_set_list,C,d,use_lmi)
# X = calculate_candidate_vectors(x_opt_2,X_opt_2)
# ub_2 = get_ub_X1(X,C,d,c3,c2,c1,c0)
# t2 = time_ns()
# total_time_2 = (t2-t1)*10^(-9)

# t1 = time_ns()
# unc_set_list_z2 = get_uncertainty_set_z2(c3,c2,c1,c0)
# res_3, x_opt_3, X_opt_3, lb_3 = solve_rpt_relaxation_X1_best_slc_gersh(unc_set_list_z2,C,d,use_lmi)
# X = calculate_candidate_vectors(x_opt_3,X_opt_3)
# ub_3 = get_ub_X1(X,C,d,c3,c2,c1,c0)
# t2 = time_ns()
# total_time_3 = (t2-t1)*10^(-9)


t1 = time_ns()
x_opt_1, obj_opt_1, gen_hyper_1, ub_list_1, lb_list_1, hyp_list = rpt_bb_X1(C,d,c3,c2,c1,c0,δ,use_lmi)
t2 = time_ns()
total_time_1 = (t2-t1)*10^(-9)

c3_2, c2_2, c1_2, c0_2 = generate_degree3_polynomial(n, 0.5)
p_1_coeffs = [c3,c2,c1,c0]
p_2_coeffs = [c3_2,c2_2,c1_2,c0_2]
C, d = zeros(n,n) + I, ones(n)

t1 = time_ns()
x_opt_2, obj_opt_2, gen_hyper_2, ub_list_2, lb_list_2, hyp_list = rpt_bb_X2(C,d,p_1_coeffs,p_2_coeffs,δ,use_lmi)
t2 = time_ns()
total_time_2 = (t2-t1)*10^(-9)

α = 3.5
C, d = zeros(n,n) + I, ones(n)
t1 = time_ns()
x_opt_3, obj_opt_3, gen_hyper_3, ub_list_3, lb_list_3, hyp_list = rpt_bb_X3(C,d,c3,c2,c1,c0,δ,use_lmi,α)
t2 = time_ns()
total_time_3 = (t2-t1)*10^(-9)


println("Optimal values: ")
println(obj_opt_1)
println(obj_opt_2)
println(obj_opt_3)
println("Total times: ")
println(total_time_1)
println(total_time_2)
println(total_time_3)
println("Generated Hyperplanes: ")
println(gen_hyper_1)
println(gen_hyper_2)
println(gen_hyper_3)


# instances to repeat for n=30: 7

# c3_mat = reshape(c3, :, size(c3, 3))
# writedlm("Polynomial_Instances/Poly3_n30_instance7/c0.csv",  Vector([c0]), ',')
# writedlm("Polynomial_Instances/Poly3_n30_instance7/c1.csv",  c1, ',')
# writedlm("Polynomial_Instances/Poly3_n30_instance7/c2.csv",  c2, ',')
# writedlm("Polynomial_Instances/Poly3_n30_instance7/c3.csv",  c3_mat, ',')
#
#
# c3_2_mat = reshape(c3_2, :, size(c3_2, 3))
# writedlm("Polynomial_Instances/Poly3_n30_instance7/c0_2.csv",  Vector([c0_2]), ',')
# writedlm("Polynomial_Instances/Poly3_n30_instance7/c1_2.csv",  c1_2, ',')
# writedlm("Polynomial_Instances/Poly3_n30_instance7/c2_2.csv",  c2_2, ',')
# writedlm("Polynomial_Instances/Poly3_n30_instance7/c3_2.csv",  c3_2_mat, ',')


here

#########################     Run Averages    ###############################

δ = 1e-4
use_lmi = false

run_X1, run_X2, run_X3 = false, false, true
n = 30
α = 3.5
C, d = zeros(n,n) + I, ones(n)
obj_vals, total_times, gen_hyper_list = [], [], []
lower_bounds_list, upper_bounds_list = [], []
for i in 1:10
    c3_mat = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c3.csv", DataFrame, header=false)
    c3 = reshape(Matrix(c3_mat), n, n, n)
    c2_mat = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c2.csv", DataFrame, header=false)
    c2 = Matrix(c2_mat)
    c1_mat = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c1.csv", DataFrame, header=false)
    c1 = Vector(c1_mat[:,1])
    c0_mat = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c0.csv", DataFrame, header=false)
    c0 = c0_mat[1,1]

    c3_mat_2 = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c3_2.csv", DataFrame, header=false)
    c3_2 = reshape(Matrix(c3_mat_2), n, n, n)
    c2_mat_2 = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c2_2.csv", DataFrame, header=false)
    c2_2 = Matrix(c2_mat_2)
    c1_mat_2 = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c1_2.csv", DataFrame, header=false)
    c1_2 = Vector(c1_mat_2[:,1])
    c0_mat_2 = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c0_2.csv", DataFrame, header=false)
    c0_2 = c0_mat_2[1,1]

    p_1_coeffs = [c3,c2,c1,c0]
    p_2_coeffs = [c3_2,c2_2,c1_2,c0_2]

    if run_X1
        t1 = time_ns()
        x_opt, obj_opt, gen_hyper, ub_list, lb_list, hyp_list = rpt_bb_X1(C,d,c3,c2,c1,c0,δ,use_lmi)
        t2 = time_ns()
        total_time = (t2-t1)*10^(-9)
    elseif run_X2
        t1 = time_ns()
        x_opt, obj_opt, gen_hyper, ub_list, lb_list, hyp_list = rpt_bb_X2(C,d,p_1_coeffs,p_2_coeffs,δ,use_lmi)
        t2 = time_ns()
        total_time = (t2-t1)*10^(-9)
    elseif run_X3
        t1 = time_ns()
        x_opt, obj_opt, gen_hyper, ub_list, lb_list, hyp_list = rpt_bb_X3(C,d,c3,c2,c1,c0,δ,use_lmi,α)
        t2 = time_ns()
        total_time = (t2-t1)*10^(-9)
    end

    push!(obj_vals, obj_opt)
    push!(total_times, total_time)
    push!(gen_hyper_list, gen_hyper)
    push!(lower_bounds_list, lb_list)
    push!(upper_bounds_list, ub_list)
end

println("------------------------------------------------------")
println("Average Results:")
println("-----------------------------------------------------")
println("Objective values: ")
println(mean(obj_vals))
println("Total times: ")
println(mean(total_times))
println("Generated Hyperplanes: ")
println(mean(gen_hyper_list))
println("-----------------------------------------------------")



obj_vals_X3 = obj_vals

here


#################    Random vs BEST SLC    ##############################




δ = 1e-4
use_lmi = false

n = 30
C, d = zeros(n,n) + I, ones(n)
total_times_random, lower_bounds_random, upper_bounds_random = [], [], []
total_times_best, lower_bounds_best, upper_bounds_best = [], [], []
run_X1, run_X2 = false, true
for i in 1:10
    c3_mat = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c3.csv", DataFrame, header=false)
    c3 = reshape(Matrix(c3_mat), n, n, n)
    c2_mat = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c2.csv", DataFrame, header=false)
    c2 = Matrix(c2_mat)
    c1_mat = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c1.csv", DataFrame, header=false)
    c1 = Vector(c1_mat[:,1])
    c0_mat = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c0.csv", DataFrame, header=false)
    c0 = c0_mat[1,1]

    c3_mat_2 = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c3_2.csv", DataFrame, header=false)
    c3_2 = reshape(Matrix(c3_mat_2), n, n, n)
    c2_mat_2 = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c2_2.csv", DataFrame, header=false)
    c2_2 = Matrix(c2_mat_2)
    c1_mat_2 = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c1_2.csv", DataFrame, header=false)
    c1_2 = Vector(c1_mat_2[:,1])
    c0_mat_2 = CSV.read("Polynomial_Instances/Poly3_n"*string(n)*"_instance"*string(i)*"/c0_2.csv", DataFrame, header=false)
    c0_2 = c0_mat_2[1,1]

    p_1_coeffs = [c3,c2,c1,c0]
    p_2_coeffs = [c3_2,c2_2,c1_2,c0_2]

    if run_X1
        t1 = time_ns()
        p2_list, q2_list = get_poly3_slc_decomposition_X1_sdp(c3,c2,c1,c0)
        res_1, x_opt_1, X_opt_1, lb_random = solve_rpt_relaxation_X1_random_slc(C,d,p2_list,q2_list,use_lmi)
        X = calculate_candidate_vectors(x_opt_1,X_opt_1)
        ub_random = get_ub_X1(X,C,d,c3,c2,c1,c0)
        t2 = time_ns()
        total_time_random = (t2-t1)*10^(-9)

        t1 = time_ns()
        unc_set_list = get_uncertainty_set_z1(c3,c2,c1,c0)
        res_2, x_opt_2, X_opt_2, lb_best = solve_rpt_relaxation_X1_best_slc(unc_set_list,C,d,use_lmi)
        X = calculate_candidate_vectors(x_opt_2,X_opt_2)
        ub_best = get_ub_X1(X,C,d,c3,c2,c1,c0)
        t2 = time_ns()
        total_time_best = (t2-t1)*10^(-9)
    elseif run_X2
        t1 = time_ns()
        p2_list_1, q2_list_1 = get_poly3_slc_decomposition_X1_sdp(c3,c2,c1,c0)
        p2_list_2, q2_list_2 = get_poly3_slc_decomposition_X1_sdp(c3_2,c2_2,c1_2,c0_2)
        res_1, x_opt_1, X_opt_1, lb_random = solve_rpt_relaxation_X2_random_slc(C,d,p2_list_1,q2_list_1,p2_list_2,q2_list_2,use_lmi)
        X_1 = calculate_candidate_vectors(x_opt_1,X_opt_1)
        ub_random = get_ub_X2(X_1,C,d,p_1_coeffs,p_2_coeffs)
        t2 = time_ns()
        total_time_random = (t2-t1)*10^(-9)

        t1 = time_ns()
        unc_set_list_1 = get_uncertainty_set_z1(c3,c2,c1,c0)
        unc_set_list_2 = get_uncertainty_set_z1(c3_2,c2_2,c1_2,c0_2)
        res_2, x_opt_2, X_opt_2, lb_best = solve_rpt_relaxation_X2_best_slc(unc_set_list_1,unc_set_list_2,C,d,use_lmi)
        X_2 = calculate_candidate_vectors(x_opt_2,X_opt_2)
        ub_best = get_ub_X2(X_2,C,d,p_1_coeffs,p_2_coeffs)
        t2 = time_ns()
        total_time_best = (t2-t1)*10^(-9)
    end

    push!(total_times_random, total_time_random)
    push!(lower_bounds_random, lb_random)
    if ub_random < 1000000
        push!(upper_bounds_random, ub_random)
    end

    push!(total_times_best, total_time_best)
    push!(lower_bounds_best, lb_best)
    if ub_best < 1000000
        push!(upper_bounds_best, ub_best)
    end
end

println("------------------------------------------------------")
println("Random:")
println("-----------------------------------------------------")
println("Lower bounds: ")
println(mean(lower_bounds_random))
println("Upper bounds: ")
println(mean(upper_bounds_random))
println("Total times: ")
println(mean(total_times_random))
println("Best:")
println("-----------------------------------------------------")
println("Lower bounds: ")
println(mean(lower_bounds_best))
println("Upper bounds: ")
println(mean(upper_bounds_best))
println("Total times: ")
println(mean(total_times_best))
