using LinearAlgebra, JuMP, Gurobi, Distributions, Statistics, Mosek, MosekTools
using Ipopt, DelimitedFiles, HDF5, Random, CSV, DataFrames

###################     Global bb functions   ########################

function generate_hyperplane_eigen(x_opt,X_opt)
    n_x = size(X_opt,1)
    Y = X_opt .- x_opt*x_opt'
    (λ,U) = eigen(Y)
    f = U[:,n_x]
    l = f'*x_opt
    return false, f, l
end

function evaluate_poly3(c3,c2,c1,c0,x)
    n = size(x,1)
    res_1 = sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n)
    res_2 = sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n)
    res_3 = sum(c1[i]*x[i] for i=1:n)
    res_4 = c0
    return res_1+res_2+res_3+res_4
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
    res_1 = (sum(C*x .<= d)  == size(C,1))
    res_2 = all(y -> 0.01 <= y <= 0.99, x)
    return res_1 && res_2
end

function poly4_init(c4,c3,c2,c1,c0,x)
    n = size(x,1)
    res_1 = sum(c4[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n)
    res_2 = sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n)
    res_3 = sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n)
    res_4 = x'*c1 + c0
    return res_1 + res_2 + res_3 + res_4
end

function poly4_slc_obj(Z,x)
    n = size(x,1)
    β_list, P_list, r_list, w_list = Z[1], Z[2], Z[3], Z[4]
    γ_list, Q_list, f_list, g_list = Z[5], Z[6], Z[7], Z[8]

    res_1 = sum(x[i]*(sum(β_list[i][j,k,l]*x[j]*x[k]*x[l] for j=1:n, k=1:n, l=1:n) +
                      x'*P_list[i]*x + x'*r_list[i] + w_list[i]) for i=1:n)

    res_2 = sum((1-x[i])*(sum(γ_list[i][j,k,l]*x[j]*x[k]*x[l] for j=1:n, k=1:n, l=1:n) +
                          x'*Q_list[i]*x + x'*f_list[i] + g_list[i]) for i=1:n)

    return res_1 + res_2
end

function poly4_scc_obj(Z,x)
    n = size(x,1)
    P_vals, r_vals, w_vals = Z[1], Z[2], Z[3]
    Q_vals, f_vals, g_vals = Z[4], Z[5], Z[6]
    T_vals, h_vals, s_vals = Z[7], Z[8], Z[9]

    res_1 = sum(x[i]*x[j]*(x'*P_vals[:,:,i,j]*x + x'*r_vals[:,i,j] + w_vals[i,j]) for i=1:n, j=1:n if j >= i)

    res_2 = sum(x[i]*(1-x[j])*(x'*Q_vals[:,:,i,j]*x + x'*f_vals[:,i,j] + g_vals[i,j]) for i=1:n, j=1:n)

    res_3 = sum((1-x[i])*(1-x[j])*(x'*T_vals[:,:,i,j]*x + x'*h_vals[:,i,j] + s_vals[i,j]) for i=1:n, j=1:n if j >= i)

    return res_1 + res_2 + res_3
end


function generate_degree4_polynomial(n, density=0.2)
    # Initialize coefficients with zeros
    c4 = zeros(n, n, n, n)  # For quartic terms
    c3 = zeros(n, n, n)     # For cubic terms
    c2 = zeros(n, n)        # For quadratic terms
    c1 = zeros(n)           # For linear terms
    c0 = 0.0                # Constant term

    # Populate quartic coefficients sparsely
    num_quartic_terms = Int(round(density * n^4))
    for _ in 1:num_quartic_terms
        i, j, k, l = rand(1:n, 4)  # Random indices
        c4[i, j, k, l] = rand(-5.0:0.1:5.0)  # Random coefficient in range [-5, 5]
    end

    # Populate cubic coefficients sparsely
    num_cubic_terms = Int(round(density * n^3))
    for _ in 1:num_cubic_terms
        i, j, k = rand(1:n, 3)  # Random indices
        c3[i, j, k] = rand(-5.0:0.1:5.0)  # Random coefficient in range [-5, 5]
    end

    # Populate quadratic coefficients sparsely
    num_quadratic_terms = Int(round(density * n^2))
    for _ in 1:num_quadratic_terms
        i, j = rand(1:n, 2)  # Random indices
        c2[i, j] = rand(-5.0:0.1:5.0)  # Random coefficient in range [-5, 5]
    end

    # Populate linear coefficients sparsely
    num_linear_terms = Int(round(density * n))
    for _ in 1:num_linear_terms
        i = rand(1:n)  # Random index
        c1[i] = rand(-5.0:0.1:5.0)  # Random coefficient in range [-5, 5]
    end

    # Set a random constant term
    c0 = rand(-5.0:0.1:5.0)

    return c4, c3, c2, c1, c0
end

############################    X_1   ####################################

function get_uncertainty_set_z1(c4,c3,c2,c1,c0)
    n = size(c1,1)
    Abar_list = []
    Bbar_list = []
    C_list = []
    cbar_list = []
    dbar_list = []
    e_list = []
    μbar_list = []
    νbar_list = []
    ξ_list = []
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
    # fourth degree terms
    for i in 1:n
        for j in i:n
            for k in j:n
                for l in k:n
                    Abar = zeros(n,n,n,n)
                    Bbar = zeros(n,n,n,n)
                    C = zeros(n,n,n,n)
                    cbar = zeros(n,n,n)
                    dbar = zeros(n,n,n)
                    e = zeros(n,n,n)
                    μbar = zeros(n,n)
                    νbar = zeros(n,n)
                    ξ = zeros(n,n)
                    A = zeros(n,n,n)
                    B = zeros(n,n,n)
                    c = zeros(n,n)
                    d = zeros(n,n)
                    μ = zeros(n)
                    ν = zeros(n)
                    Ξ = zeros(n,n)
                    ω = zeros(n)
                    γ = 0
                    if i != j && i != k && i != l && j != k && j != l && k != l
                        Abar[k,l,i,j] = 1
                        Abar[l,k,i,j] = 1
                        Abar[j,l,i,k] = 1
                        Abar[l,j,i,k] = 1
                        Abar[j,k,i,l] = 1
                        Abar[k,j,i,l] = 1
                        Abar[i,l,j,k] = 1
                        Abar[l,i,j,k] = 1
                        Abar[i,k,j,l] = 1
                        Abar[k,i,j,l] = 1
                        Abar[i,j,k,l] = 1
                        Abar[j,i,k,l] = 1

                        C[k,l,i,j] = 1
                        C[l,k,i,j] = 1
                        C[j,l,i,k] = 1
                        C[l,j,i,k] = 1
                        C[j,k,i,l] = 1
                        C[k,j,i,l] = 1
                        C[i,l,j,k] = 1
                        C[l,i,j,k] = 1
                        C[i,k,j,l] = 1
                        C[k,i,j,l] = 1
                        C[i,j,k,l] = 1
                        C[j,i,k,l] = 1

                        Bbar[k,l,i,j] = -1
                        Bbar[l,k,i,j] = -1
                        Bbar[k,l,j,i] = -1
                        Bbar[l,k,j,i] = -1
                        Bbar[j,l,i,k] = -1
                        Bbar[l,j,i,k] = -1
                        Bbar[j,l,k,i] = -1
                        Bbar[l,j,k,i] = -1
                        Bbar[j,k,i,l] = -1
                        Bbar[k,j,i,l] = -1
                        Bbar[j,k,l,i] = -1
                        Bbar[k,j,l,i] = -1
                        Bbar[i,l,j,k] = -1
                        Bbar[l,i,j,k] = -1
                        Bbar[i,l,k,j] = -1
                        Bbar[l,i,k,j] = -1
                        Bbar[i,k,j,l] = -1
                        Bbar[k,i,j,l] = -1
                        Bbar[i,k,l,j] = -1
                        Bbar[k,i,l,j] = -1
                        Bbar[i,j,k,l] = -1
                        Bbar[j,i,k,l] = -1
                        Bbar[i,j,l,k] = -1
                        Bbar[j,i,l,k] = -1

                        push!(s1, c4[i,j,k,l]+c4[i,j,l,k]+c4[i,k,j,l]+c4[i,k,l,j]+
                                  c4[i,l,j,k]+c4[i,l,k,j]+c4[j,i,k,l]+c4[j,i,l,k]+
                                  c4[j,k,i,l]+c4[j,k,l,i]+c4[j,l,i,k]+c4[j,l,k,i]+
                                  c4[k,i,j,l]+c4[k,i,l,j]+c4[k,j,i,l]+c4[k,j,l,i]+
                                  c4[k,l,i,j]+c4[k,l,j,i]+c4[l,i,j,k]+c4[l,i,k,j]+
                                  c4[l,j,i,k]+c4[l,j,k,i]+c4[l,k,i,j]+c4[l,k,j,i])

                    elseif i == j && i != k && i != l && k != l
                        Abar[k,l,i,i] = 1
                        Abar[l,k,i,i] = 1
                        Abar[i,l,i,k] = 1
                        Abar[l,i,i,k] = 1
                        Abar[i,k,i,l] = 1
                        Abar[k,i,i,l] = 1
                        Abar[i,i,k,l] = 1

                        C[k,l,i,i] = 1
                        C[l,k,i,i] = 1
                        C[i,l,i,k] = 1
                        C[l,i,i,k] = 1
                        C[i,k,i,l] = 1
                        C[k,i,i,l] = 1
                        C[i,i,k,l] = 1

                        Bbar[k,l,i,i] = -1
                        Bbar[l,k,i,i] = -1
                        Bbar[i,l,i,k] = -1
                        Bbar[l,i,i,k] = -1
                        Bbar[i,l,k,i] = -1
                        Bbar[l,i,k,i] = -1
                        Bbar[i,k,i,l] = -1
                        Bbar[k,i,i,l] = -1
                        Bbar[i,k,l,i] = -1
                        Bbar[k,i,l,i] = -1
                        Bbar[i,i,k,l] = -1
                        Bbar[i,i,l,k] = -1

                        push!(s1, c4[i,i,k,l]+c4[i,i,l,k]+c4[i,k,i,l]+c4[i,k,l,i]+
                                  c4[i,l,i,k]+c4[i,l,k,i]+c4[k,i,i,l]+c4[k,i,l,i]+
                                  c4[k,l,i,i]+c4[l,i,i,k]+c4[l,i,k,i]+c4[l,k,i,i])


                    elseif j == k && i != j && l != j && i != l
                        Abar[j,j,i,l] = 1
                        Abar[j,l,i,j] = 1
                        Abar[l,j,i,j] = 1
                        Abar[i,j,j,l] = 1
                        Abar[j,i,j,l] = 1
                        Abar[i,l,j,j] = 1
                        Abar[l,i,j,j] = 1

                        C[j,j,i,l] = 1
                        C[j,l,i,j] = 1
                        C[l,j,i,j] = 1
                        C[i,j,j,l] = 1
                        C[j,i,j,l] = 1
                        C[i,l,j,j] = 1
                        C[l,i,j,j] = 1

                        Bbar[j,j,i,l] = -1
                        Bbar[j,j,l,i] = -1
                        Bbar[j,l,i,j] = -1
                        Bbar[l,j,i,j] = -1
                        Bbar[j,l,j,i] = -1
                        Bbar[l,j,j,i] = -1
                        Bbar[i,j,j,l] = -1
                        Bbar[j,i,j,l] = -1
                        Bbar[i,j,l,j] = -1
                        Bbar[j,i,l,j] = -1
                        Bbar[i,l,j,j] = -1
                        Bbar[l,i,j,j] = -1

                        push!(s1, c4[j,j,i,l]+c4[j,j,l,i]+c4[j,i,j,l]+c4[j,i,l,j]+
                                  c4[j,l,j,i]+c4[j,l,i,j]+c4[i,j,j,l]+c4[i,j,l,j]+
                                  c4[i,l,j,j]+c4[l,j,j,i]+c4[l,j,i,j]+c4[l,i,j,j])

                    elseif k == l && i != k && j != k && i != j
                        Abar[k,k,i,j] = 1
                        Abar[j,k,i,k] = 1
                        Abar[k,j,i,k] = 1
                        Abar[i,k,j,k] = 1
                        Abar[k,i,j,k] = 1
                        Abar[i,j,k,k] = 1
                        Abar[j,i,k,k] = 1

                        C[k,k,i,j] = 1
                        C[j,k,i,k] = 1
                        C[k,j,i,k] = 1
                        C[i,k,j,k] = 1
                        C[k,i,j,k] = 1
                        C[i,j,k,k] = 1
                        C[j,i,k,k] = 1

                        Bbar[k,k,i,j] = -1
                        Bbar[k,k,j,i] = -1
                        Bbar[j,k,i,k] = -1
                        Bbar[k,j,i,k] = -1
                        Bbar[j,k,k,i] = -1
                        Bbar[k,j,k,i] = -1
                        Bbar[i,k,j,k] = -1
                        Bbar[k,i,j,k] = -1
                        Bbar[i,k,k,j] = -1
                        Bbar[k,i,k,j] = -1
                        Bbar[i,j,k,k] = -1
                        Bbar[j,i,k,k] = -1

                        push!(s1, c4[k,k,i,j]+c4[k,k,j,i]+c4[k,i,k,j]+c4[k,i,j,k]+
                                  c4[k,j,k,i]+c4[k,j,i,k]+c4[i,k,k,j]+c4[i,k,j,k]+
                                  c4[i,j,k,k]+c4[j,k,k,i]+c4[j,k,i,k]+c4[j,i,k,k])

                    elseif i == j && k == l && k != j
                        Abar[k,k,i,i] = 1
                        Abar[i,i,k,k] = 1
                        Abar[i,k,i,k] = 1
                        Abar[k,i,i,k] = 1

                        C[k,k,i,i] = 1
                        C[i,i,k,k] = 1
                        C[i,k,i,k] = 1
                        C[k,i,i,k] = 1

                        Bbar[k,k,i,i] = -1
                        Bbar[i,i,k,k] = -1
                        Bbar[i,k,i,k] = -1
                        Bbar[k,i,i,k] = -1
                        Bbar[i,k,k,i] = -1
                        Bbar[k,i,k,i] = -1

                        push!(s1, c4[i,i,k,k]+c4[i,k,i,k]+c4[i,k,k,i]+
                                  c4[k,i,i,k]+c4[k,i,k,i]+c4[k,k,i,i])

                    elseif i == j && j == k && l != k
                        Abar[i,i,i,l] = 1
                        Abar[i,l,i,i] = 1
                        Abar[l,i,i,i] = 1

                        C[i,i,i,l] = 1
                        C[i,l,i,i] = 1
                        C[l,i,i,i] = 1

                        Bbar[i,i,i,l] = -1
                        Bbar[i,i,l,i] = -1
                        Bbar[i,l,i,i] = -1
                        Bbar[l,i,i,i] = -1

                        push!(s1, c4[i,i,i,l]+c4[i,i,l,i]+c4[i,l,i,i]+c4[l,i,i,i])

                    elseif j == k && k == l && i != j
                        Abar[j,j,i,j] = 1
                        Abar[i,j,j,j] = 1
                        Abar[j,i,j,j] = 1

                        C[j,j,i,j] = 1
                        C[i,j,j,j] = 1
                        C[j,i,j,j] = 1

                        Bbar[j,j,i,j] = -1
                        Bbar[j,j,j,i] = -1
                        Bbar[i,j,j,j] = -1
                        Bbar[j,i,j,j] = -1

                        push!(s1, c4[j,j,j,i]+c4[j,j,i,j]+c4[j,i,j,j]+c4[i,j,j,j])

                    elseif i == j && j == k && k == l
                        Abar[i,i,i,i] = 1
                        C[i,i,i,i] = 1
                        Bbar[i,i,i,i] = -1
                        push!(s1, c4[i,i,i,i])

                    end
                    push!(Abar_list, Abar)
                    push!(Bbar_list, Bbar)
                    push!(C_list, C)
                    push!(cbar_list, cbar)
                    push!(dbar_list, dbar)
                    push!(e_list, e)
                    push!(μbar_list, μbar)
                    push!(νbar_list, νbar)
                    push!(ξ_list, ξ)
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
    end

    # third degree terms
    for i in 1:n
        for j in i:n
            for k in j:n
                Abar = zeros(n,n,n,n)
                Bbar = zeros(n,n,n,n)
                C = zeros(n,n,n,n)
                cbar = zeros(n,n,n)
                dbar = zeros(n,n,n)
                e = zeros(n,n,n)
                μbar = zeros(n,n)
                νbar = zeros(n,n)
                ξ = zeros(n,n)
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
                    cbar[k,i,j] = 1
                    cbar[j,i,k] = 1
                    cbar[i,j,k] = 1

                    e[k,i,j] = 1
                    e[j,i,k] = 1
                    e[i,j,k] = 1

                    dbar[k,i,j] = -1
                    dbar[k,j,i] = -1
                    dbar[j,i,k] = -1
                    dbar[j,k,i] = -1
                    dbar[i,j,k] = -1
                    dbar[i,k,j] = -1

                    for l in 1:n
                        Bbar[j,k,i,l] = 1
                        Bbar[k,j,i,l] = 1
                        Bbar[i,k,j,l] = 1
                        Bbar[k,i,j,l] = 1
                        Bbar[i,j,k,l] = 1
                        Bbar[j,i,k,l] = 1
                    end

                    C[j,k,i,i] = -2
                    C[k,j,i,i] = -2
                    C[i,k,j,j] = -2
                    C[k,i,j,j] = -2
                    C[i,j,k,k] = -2
                    C[j,i,k,k] = -2

                    for l in 1:n
                        if l < i
                            C[j,k,l,i] = -1
                            C[k,j,l,i] = -1
                        elseif l > i
                            C[j,k,i,l] = -1
                            C[k,j,i,l] = -1
                        end
                        if l < j
                            C[i,k,l,j] = -1
                            C[k,i,l,j] = -1
                        elseif l > j
                            C[i,k,j,l] = -1
                            C[k,i,j,l] = -1
                        end
                        if l < k
                            C[i,j,l,k] = -1
                            C[j,i,l,k] = -1
                        elseif l > k
                            C[i,j,k,l] = -1
                            C[j,i,k,l] = -1
                        end
                    end

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

                elseif i == j && k != j
                    cbar[k,i,i] = 1
                    cbar[i,i,k] = 1

                    e[k,i,i] = 1
                    e[i,i,k] = 1

                    dbar[k,i,i] = -1
                    dbar[i,i,k] = -1
                    dbar[i,k,i] = -1

                    for l in 1:n
                        Bbar[i,k,i,l] = 1
                        Bbar[k,i,i,l] = 1
                        Bbar[i,i,k,l] = 1
                    end

                    C[i,k,i,i] = -2
                    C[k,i,i,i] = -2
                    C[i,i,k,k] = -2

                    for l in 1:n
                        if l < i
                            C[i,k,l,i] = -1
                            C[k,i,l,i] = -1
                        elseif l > i
                            C[i,k,i,l] = -1
                            C[k,i,i,l] = -1
                        end
                        if l < k
                            C[i,i,l,k] = -1
                        elseif l > k
                            C[i,i,k,l] = -1
                        end
                    end

                    A[i,i,k] = 1
                    A[i,k,i] = 1
                    A[k,i,i] = 1
                    B[i,i,k] = -1
                    B[i,k,i] = -1
                    B[k,i,i] = -1

                    push!(s1, c3[i,i,k]+c3[i,k,i]+c3[k,i,i])

                elseif j == k && i != j
                    cbar[j,i,j] = 1
                    cbar[i,j,j] = 1

                    e[j,i,j] = 1
                    e[i,j,j] = 1

                    dbar[j,i,j] = -1
                    dbar[j,j,i] = -1
                    dbar[i,j,j] = -1

                    for l in 1:n
                        Bbar[j,j,i,l] = 1
                        Bbar[i,j,j,l] = 1
                        Bbar[j,i,j,l] = 1
                    end

                    C[j,j,i,i] = -2
                    C[i,j,j,j] = -2
                    C[j,i,j,j] = -2

                    for l in 1:n
                        if l > i
                            C[j,j,i,l] = -1
                        elseif l < i
                            C[j,j,l,i] = -1
                        end
                        if l > j
                            C[i,j,j,l] = -1
                            C[j,i,j,l] = -1
                        elseif l < j
                            C[i,j,l,j] = -1
                            C[j,i,l,j] = -1
                        end
                    end

                    A[j,j,i] = 1
                    A[i,j,j] = 1
                    A[j,i,j] = 1
                    B[j,j,i] = -1
                    B[i,j,j] = -1
                    B[j,i,j] = -1

                    push!(s1, c3[j,j,i]+c3[j,i,j]+c3[i,j,j])

                elseif i == j && j == k
                    cbar[i,i,i] = 1
                    e[i,i,i] = 1
                    dbar[i,i,i] = -1
                    for l in 1:n
                        Bbar[i,i,i,l] = 1
                    end
                    C[i,i,i,i] = -2
                    for l in 1:n
                        if l < i
                            C[i,i,l,i] = -1
                        elseif l > i
                            C[i,i,i,l] = -1
                        end
                    end
                    A[i,i,i] = 1
                    B[i,i,i] = -1
                    push!(s1, c3[i,i,i])
                end

                push!(Abar_list, Abar)
                push!(Bbar_list, Bbar)
                push!(C_list, C)
                push!(cbar_list, cbar)
                push!(dbar_list, dbar)
                push!(e_list, e)
                push!(μbar_list, μbar)
                push!(νbar_list, νbar)
                push!(ξ_list, ξ)
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
            Abar = zeros(n,n,n,n)
            Bbar = zeros(n,n,n,n)
            C = zeros(n,n,n,n)
            cbar = zeros(n,n,n)
            dbar = zeros(n,n,n)
            e = zeros(n,n,n)
            μbar = zeros(n,n)
            νbar = zeros(n,n)
            ξ = zeros(n,n)
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
                μbar[i,j] = 1

                ξ[i,j] = 1

                νbar[i,j] = -1
                νbar[j,i] = -1

                for l in 1:n
                    dbar[j,i,l] = 1
                    dbar[i,j,l] = 1
                end

                e[j,i,i] = -2
                e[i,j,j] = -2
                for l in 1:n
                    if l < i
                        e[j,l,i] = -1
                    elseif l > i
                        e[j,i,l] = -1
                    end
                    if l < j
                        e[i,l,j] = -1
                    elseif l > j
                        e[i,j,l] = -1
                    end
                end

                for k in 1:n
                    for l in k:n
                        C[i,j,k,l] = 1
                        C[j,i,k,l] = 1
                    end
                end

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
                μbar[i,i] = 1
                ξ[i,i] = 1
                νbar[i,i] = -1
                for l in 1:n
                    dbar[i,i,l] = 1
                end
                e[i,i,i] = -2
                for l in 1:n
                    if l < i
                        e[i,l,i] = -1
                    elseif l > i
                        e[i,i,l] = -1
                    end
                end
                for k in 1:n
                    for l in k:n
                        C[i,i,k,l] = 1
                    end
                end
                c[i,i] = 1
                d[i,i] = -1
                for l in 1:n
                    B[i,i,l] = 1
                end
                Ξ[i,i] = 1
                push!(s1, c2[i,i])
            end

            push!(Abar_list, Abar)
            push!(Bbar_list, Bbar)
            push!(C_list, C)
            push!(cbar_list, cbar)
            push!(dbar_list, dbar)
            push!(e_list, e)
            push!(μbar_list, μbar)
            push!(νbar_list, νbar)
            push!(ξ_list, ξ)
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
        Abar = zeros(n,n,n,n)
        Bbar = zeros(n,n,n,n)
        C = zeros(n,n,n,n)
        cbar = zeros(n,n,n)
        dbar = zeros(n,n,n)
        e = zeros(n,n,n)
        μbar = zeros(n,n)
        νbar = zeros(n,n)
        ξ = zeros(n,n)
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
            νbar[i,l] = 1
        end

        for k in 1:n
            for l in k:n
                e[i,k,l] = 1
            end
        end

        ξ[i,i] = -2
        for l in 1:n
            if l < i
                ξ[l,i] = -1
            elseif l > i
                ξ[i,l] = -1
            end
        end

        for l in 1:n
            d[i,l] = 1
        end
        μ[i] = 1
        ν[i] = -1
        ω[i] = 1

        push!(s1, c1[i])

        push!(Abar_list, Abar)
        push!(Bbar_list, Bbar)
        push!(C_list, C)
        push!(cbar_list, cbar)
        push!(dbar_list, dbar)
        push!(e_list, e)
        push!(μbar_list, μbar)
        push!(νbar_list, νbar)
        push!(ξ_list, ξ)
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

    # zero degree terms
    Abar = zeros(n,n,n,n)
    Bbar = zeros(n,n,n,n)
    C = zeros(n,n,n,n)
    cbar = zeros(n,n,n)
    dbar = zeros(n,n,n)
    e = zeros(n,n,n)
    μbar = zeros(n,n)
    νbar = zeros(n,n)
    ξ = zeros(n,n)
    A = zeros(n,n,n)
    B = zeros(n,n,n)
    c = zeros(n,n)
    d = zeros(n,n)
    μ = zeros(n)
    ν = zeros(n)
    Ξ = zeros(n,n)
    ω = zeros(n)
    γ = 1
    for i in 1:n
        for j in i:n
            ξ[i,j] = 1
        end
        ν[i] = 1
    end
    push!(s1, c0)

    push!(Abar_list, Abar)
    push!(Bbar_list, Bbar)
    push!(C_list, C)
    push!(cbar_list, cbar)
    push!(dbar_list, dbar)
    push!(e_list, e)
    push!(μbar_list, μbar)
    push!(νbar_list, νbar)
    push!(ξ_list, ξ)
    push!(A_list, A)
    push!(B_list, B)
    push!(c_list, c)
    push!(d_list, d)
    push!(μ_list, μ)
    push!(ν_list, ν)
    push!(Ξ_list, Ξ)
    push!(ω_list, ω)
    push!(γ_list, γ)

    # return final output
    unc_set_list = [Abar_list, Bbar_list, C_list, cbar_list, dbar_list, e_list,
                    μbar_list, νbar_list, ξ_list, A_list, B_list, c_list, d_list,
                    μ_list, ν_list, Ξ_list, ω_list, γ_list, s1]
    return unc_set_list
end

function solve_rpt_relaxation_X1_best_slc(unc_set_list,C,d,use_lmi)
    L, n = size(C)
    Abar_list, Bbar_list, C_list = unc_set_list[1], unc_set_list[2], unc_set_list[3]
    cbar_list, dbar_list, e_list = unc_set_list[4], unc_set_list[5], unc_set_list[6]
    μbar_list, νbar_list, ξ_list = unc_set_list[7], unc_set_list[8], unc_set_list[9]
    A_list, B_list, c_list = unc_set_list[10], unc_set_list[11], unc_set_list[12]
    d_list, μ_list, ν_list = unc_set_list[13], unc_set_list[14], unc_set_list[15]
    Ξ_list, ω_list, γ_list = unc_set_list[16], unc_set_list[17], unc_set_list[18]
    s1 = unc_set_list[19]
    L1 = length(s1)

    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, V[1:n,1:n,1:n]>=0)
    @variable(m1, Ybar[1:n,1:n,1:n,1:n])
    @variable(m1, Rbar[1:n,1:n,1:n,1:n])
    @variable(m1, E[1:n,1:n,1:n,1:n])
    @variable(m1, Y[1:n,1:n,1:n+1])
    @variable(m1, R[1:n,1:n,1:n])
    @variable(m1, λ[1:L1])

    @constraint(m1, [i in 1:n, j in i:n], -Ybar[:,:,i,j] .- sum(λ[l]*Abar_list[l][:,:,i,j] for l in 1:L1) in PSDCone())
    @constraint(m1, [i in 1:n, j in 1:n], -Rbar[:,:,i,j] .- sum(λ[l]*Bbar_list[l][:,:,i,j] for l in 1:L1) in PSDCone())
    @constraint(m1, [i in 1:n, j in i:n], -E[:,:,i,j] .- sum(λ[l]*C_list[l][:,:,i,j] for l in 1:L1) in PSDCone())

    @constraint(m1, [i in 1:n], -Y[:,:,i] .- sum(λ[l]*A_list[l][:,:,i] for l in 1:L1) in PSDCone())
    @constraint(m1, [i in 1:n], -R[:,:,i] .- sum(λ[l]*B_list[l][:,:,i] for l in 1:L1) in PSDCone())
    @constraint(m1, -Y[:,:,n+1] .- sum(λ[l]*Ξ_list[l] for l in 1:L1) in PSDCone())

    @constraint(m1, [i in 1:n, j in i:n], V[:,i,j] .+ sum(λ[l]*cbar_list[l][:,i,j] for l in 1:L1) .== 0)
    @constraint(m1, [i in 1:n, j in 1:n], X[:,i] .- V[:,i,j] .+ sum(λ[l]*dbar_list[l][:,i,j] for l in 1:L1) .== 0)
    @constraint(m1, [i in 1:n, j in i:n], x .- X[:,i] .- X[:,j] .+ V[:,i,j] .+ sum(λ[l]*e_list[l][:,i,j] for l in 1:L1) .== 0)

    @constraint(m1, [i in 1:n], X[:,i] .+ sum(λ[l]*c_list[l][:,i] for l in 1:L1) .== 0)
    @constraint(m1, [i in 1:n], x .- X[:,i] .+ sum(λ[l]*d_list[l][:,i] for l in 1:L1) .== 0)
    @constraint(m1, x .+ sum(λ[l]*ω_list[l] for l in 1:L1) .== 0)

    @constraint(m1, [i in 1:n, j in i:n], X[i,j] + sum(λ[l]*μbar_list[l][i,j] for l in 1:L1) == 0)
    @constraint(m1, [i in 1:n, j in 1:n], x[i] - X[i,j] + sum(λ[l]*νbar_list[l][i,j] for l in 1:L1) == 0)
    @constraint(m1, [i in 1:n, j in i:n], 1 - x[i] - x[j] + X[i,j] + sum(λ[l]*ξ_list[l][i,j] for l in 1:L1) == 0)

    @constraint(m1, [i in 1:n], x[i] + sum(λ[l]*μ_list[l][i] for l in 1:L1) == 0)
    @constraint(m1, [i in 1:n], 1 - x[i] + sum(λ[l]*ν_list[l][i] for l in 1:L1) == 0)
    @constraint(m1, 1 + sum(λ[l]*γ_list[l] for l in 1:L1) == 0)

    @constraint(m1, [i in 1:n, j in i:n], [Ybar[:,:,i,j] V[:,i,j]; (V[:,i,j])' X[i,j]] in PSDCone())
    @constraint(m1, [i in 1:n, j in 1:n], [Rbar[:,:,i,j] (X[:,i].-V[:,i,j]); (X[:,i].-V[:,i,j])' (x[i]-X[i,j])] in PSDCone())
    @constraint(m1, [i in 1:n, j in i:n], [E[:,:,i,j] (x.-X[:,i].-X[:,j].+V[:,i,j]);
                                          (x.-X[:,i].-X[:,j].+V[:,i,j])' (1-x[i]-x[j]+X[i,j])] in PSDCone())

    @constraint(m1, [i in 1:n], [Y[:,:,i]  X[:,i]; (X[:,i])' x[i]] in PSDCone())
    @constraint(m1, [i in 1:n], [R[:,:,i]  (x.-X[:,i]); (x.-X[:,i])' (1-x[i])] in PSDCone())
    @constraint(m1, [Y[:,:,n+1] x; x' 1] in PSDCone())

    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], C*X[:,i] .<= x[i]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    @constraint(m1, [i in 1:n, j in 1:n], C*V[:,i,j] .<= X[i,j]*d)
    # @constraint(m1, [k in 1:n], d*X[:,k]'*C' .+ C*X[:,k]*d' .<= C*V[k,:,:]*C' .+ x[k]*d*d')

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

function ub_ipopt_X1(C,d,c4,c3,c2,c1,c0,x0)
    n = size(C,2)
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:n]>=0)
    for j in 1:n
        JuMP.set_start_value(x[j], x0[j])
    end
    @constraint(model, C*x .<= d)
    @NLexpression(model, obj_term, sum(c4[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n) +
                                   sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n) +
                                   sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n) +
                                   sum(c1[i]*x[i] for i=1:n) + c0)
    @NLobjective(model, Min, obj_term)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED
        return objective_value(model)
    else
        return 1e6
    end
end

function get_ub_X1(X,C,d,c4,c3,c2,c1,c0)
    best_ub = 1e6
    for i in 1:length(X)
        x = X[i]
        cur_ub = ub_ipopt_X1(C,d,c4,c3,c2,c1,c0,x)
        if cur_ub < best_ub
            best_ub = cur_ub
        end
    end
    return best_ub
end


function rpt_bb_X1(C_init,d_init,c4,c3,c2,c1,c0,δ,use_lmi)
    C, d = C_init, d_init
    gen_hyper = 0

    unc_set_list = get_uncertainty_set_z1(c4,c3,c2,c1,c0)

    # Root Node
    res, x_lb, X_lb, lb = solve_rpt_relaxation_X1_best_slc(unc_set_list,C,d,use_lmi)
    X = calculate_candidate_vectors(x_lb,X_lb)
    ub = get_ub_X1(X,C,d,c4,c3,c2,c1,c0)

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
            ub_r = get_ub_X1(X_r,C_r,d_r,c4,c3,c2,c1,c0)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r])
            end
        end
        # Left child
        res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_X1_best_slc(unc_set_list,C_l,d_l,use_lmi)
        if res_l
            X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
            ub_l = get_ub_X1(X_l,C_l,d_l,c4,c3,c2,c1,c0)
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


############################     X_2    ##################################
function is_point_feasible_X2(C,d,p_2_coeffs,x)
    c4_2, c3_2, c2_2, c1_2, c0_2 = p_2_coeffs[1], p_2_coeffs[2], p_2_coeffs[3], p_2_coeffs[4], p_2_coeffs[5]
    res = (sum(C*x .<= d)  == size(C,1) && poly4_init(c4_2,c3_2,c2_2,c1_2,c0_2,x) <= 0)
    return res
end


function solve_rpt_relaxation_X2_best_slc(unc_set_list_1,unc_set_list_2,C,d,use_lmi)
    L, n = size(C)

    Abar_1_list, Bbar_1_list, C_1_list = unc_set_list_1[1], unc_set_list_1[2], unc_set_list_1[3]
    cbar_1_list, dbar_1_list, e_1_list = unc_set_list_1[4], unc_set_list_1[5], unc_set_list_1[6]
    μbar_1_list, νbar_1_list, ξ_1_list = unc_set_list_1[7], unc_set_list_1[8], unc_set_list_1[9]
    A_1_list, B_1_list, c_1_list = unc_set_list_1[10], unc_set_list_1[11], unc_set_list_1[12]
    d_1_list, μ_1_list, ν_1_list = unc_set_list_1[13], unc_set_list_1[14], unc_set_list_1[15]
    Ξ_1_list, ω_1_list, γ_1_list = unc_set_list_1[16], unc_set_list_1[17], unc_set_list_1[18]
    s1_1 = unc_set_list_1[19]
    L1_1 = length(s1_1)

    Abar_2_list, Bbar_2_list, C_2_list = unc_set_list_2[1], unc_set_list_2[2], unc_set_list_2[3]
    cbar_2_list, dbar_2_list, e_2_list = unc_set_list_2[4], unc_set_list_2[5], unc_set_list_2[6]
    μbar_2_list, νbar_2_list, ξ_2_list = unc_set_list_2[7], unc_set_list_2[8], unc_set_list_2[9]
    A_2_list, B_2_list, c_2_list = unc_set_list_2[10], unc_set_list_2[11], unc_set_list_2[12]
    d_2_list, μ_2_list, ν_2_list = unc_set_list_2[13], unc_set_list_2[14], unc_set_list_2[15]
    Ξ_2_list, ω_2_list, γ_2_list = unc_set_list_2[16], unc_set_list_2[17], unc_set_list_2[18]
    s1_2 = unc_set_list_2[19]
    L1_2 = length(s1_2)


    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, V[1:n,1:n,1:n]>=0)

    @variable(m1, Ybar_1[1:n,1:n,1:n,1:n])
    @variable(m1, Rbar_1[1:n,1:n,1:n,1:n])
    @variable(m1, E_1[1:n,1:n,1:n,1:n])
    @variable(m1, Y_1[1:n,1:n,1:n+1])
    @variable(m1, R_1[1:n,1:n,1:n])
    @variable(m1, λ_1[1:L1_1])

    @variable(m1, Ybar_2[1:n,1:n,1:n,1:n])
    @variable(m1, Rbar_2[1:n,1:n,1:n,1:n])
    @variable(m1, E_2[1:n,1:n,1:n,1:n])
    @variable(m1, Y_2[1:n,1:n,1:n+1])
    @variable(m1, R_2[1:n,1:n,1:n])
    @variable(m1, λ_2[1:L1_2])

    @constraint(m1, [i in 1:n, j in i:n], -Ybar_1[:,:,i,j] .- sum(λ_1[l]*Abar_1_list[l][:,:,i,j] for l in 1:L1_1) in PSDCone())
    @constraint(m1, [i in 1:n, j in 1:n], -Rbar_1[:,:,i,j] .- sum(λ_1[l]*Bbar_1_list[l][:,:,i,j] for l in 1:L1_1) in PSDCone())
    @constraint(m1, [i in 1:n, j in i:n], -E_1[:,:,i,j] .- sum(λ_1[l]*C_1_list[l][:,:,i,j] for l in 1:L1_1) in PSDCone())

    @constraint(m1, [i in 1:n], -Y_1[:,:,i] .- sum(λ_1[l]*A_1_list[l][:,:,i] for l in 1:L1_1) in PSDCone())
    @constraint(m1, [i in 1:n], -R_1[:,:,i] .- sum(λ_1[l]*B_1_list[l][:,:,i] for l in 1:L1_1) in PSDCone())
    @constraint(m1, -Y_1[:,:,n+1] .- sum(λ_1[l]*Ξ_1_list[l] for l in 1:L1_1) in PSDCone())

    @constraint(m1, [i in 1:n, j in i:n], V[:,i,j] .+ sum(λ_1[l]*cbar_1_list[l][:,i,j] for l in 1:L1_1) .== 0)
    @constraint(m1, [i in 1:n, j in 1:n], X[:,i] .- V[:,i,j] .+ sum(λ_1[l]*dbar_1_list[l][:,i,j] for l in 1:L1_1) .== 0)
    @constraint(m1, [i in 1:n, j in i:n], x .- X[:,i] .- X[:,j] .+ V[:,i,j] .+ sum(λ_1[l]*e_1_list[l][:,i,j] for l in 1:L1_1) .== 0)

    @constraint(m1, [i in 1:n], X[:,i] .+ sum(λ_1[l]*c_1_list[l][:,i] for l in 1:L1_1) .== 0)
    @constraint(m1, [i in 1:n], x .- X[:,i] .+ sum(λ_1[l]*d_1_list[l][:,i] for l in 1:L1_1) .== 0)
    @constraint(m1, x .+ sum(λ_1[l]*ω_1_list[l] for l in 1:L1_1) .== 0)

    @constraint(m1, [i in 1:n, j in i:n], X[i,j] + sum(λ_1[l]*μbar_1_list[l][i,j] for l in 1:L1_1) == 0)
    @constraint(m1, [i in 1:n, j in 1:n], x[i] - X[i,j] + sum(λ_1[l]*νbar_1_list[l][i,j] for l in 1:L1_1) == 0)
    @constraint(m1, [i in 1:n, j in i:n], 1 - x[i] - x[j] + X[i,j] + sum(λ_1[l]*ξ_1_list[l][i,j] for l in 1:L1_1) == 0)

    @constraint(m1, [i in 1:n], x[i] + sum(λ_1[l]*μ_1_list[l][i] for l in 1:L1_1) == 0)
    @constraint(m1, [i in 1:n], 1 - x[i] + sum(λ_1[l]*ν_1_list[l][i] for l in 1:L1_1) == 0)
    @constraint(m1, 1 + sum(λ_1[l]*γ_1_list[l] for l in 1:L1_1) == 0)

    @constraint(m1, [i in 1:n, j in i:n], [Ybar_1[:,:,i,j] V[:,i,j]; (V[:,i,j])' X[i,j]] in PSDCone())
    @constraint(m1, [i in 1:n, j in 1:n], [Rbar_1[:,:,i,j] (X[:,i].-V[:,i,j]); (X[:,i].-V[:,i,j])' (x[i]-X[i,j])] in PSDCone())
    @constraint(m1, [i in 1:n, j in i:n], [E_1[:,:,i,j] (x.-X[:,i].-X[:,j].+V[:,i,j]);
                                          (x.-X[:,i].-X[:,j].+V[:,i,j])' (1-x[i]-x[j]+X[i,j])] in PSDCone())

    @constraint(m1, [i in 1:n], [Y_1[:,:,i]  X[:,i]; (X[:,i])' x[i]] in PSDCone())
    @constraint(m1, [i in 1:n], [R_1[:,:,i]  (x.-X[:,i]); (x.-X[:,i])' (1-x[i])] in PSDCone())
    @constraint(m1, [Y_1[:,:,n+1] x; x' 1] in PSDCone())



    @constraint(m1, [i in 1:n, j in i:n], -Ybar_2[:,:,i,j] .- sum(λ_2[l]*Abar_2_list[l][:,:,i,j] for l in 1:L1_2) in PSDCone())
    @constraint(m1, [i in 1:n, j in 1:n], -Rbar_2[:,:,i,j] .- sum(λ_2[l]*Bbar_2_list[l][:,:,i,j] for l in 1:L1_2) in PSDCone())
    @constraint(m1, [i in 1:n, j in i:n], -E_2[:,:,i,j] .- sum(λ_2[l]*C_2_list[l][:,:,i,j] for l in 1:L1_2) in PSDCone())

    @constraint(m1, [i in 1:n], -Y_2[:,:,i] .- sum(λ_2[l]*A_2_list[l][:,:,i] for l in 1:L1_2) in PSDCone())
    @constraint(m1, [i in 1:n], -R_2[:,:,i] .- sum(λ_2[l]*B_2_list[l][:,:,i] for l in 1:L1_2) in PSDCone())
    @constraint(m1, -Y_2[:,:,n+1] .- sum(λ_2[l]*Ξ_2_list[l] for l in 1:L1_2) in PSDCone())

    @constraint(m1, [i in 1:n, j in i:n], V[:,i,j] .+ sum(λ_2[l]*cbar_2_list[l][:,i,j] for l in 1:L1_2) .== 0)
    @constraint(m1, [i in 1:n, j in 1:n], X[:,i] .- V[:,i,j] .+ sum(λ_2[l]*dbar_2_list[l][:,i,j] for l in 1:L1_2) .== 0)
    @constraint(m1, [i in 1:n, j in i:n], x .- X[:,i] .- X[:,j] .+ V[:,i,j] .+ sum(λ_2[l]*e_2_list[l][:,i,j] for l in 1:L1_2) .== 0)

    @constraint(m1, [i in 1:n], X[:,i] .+ sum(λ_2[l]*c_2_list[l][:,i] for l in 1:L1_2) .== 0)
    @constraint(m1, [i in 1:n], x .- X[:,i] .+ sum(λ_2[l]*d_2_list[l][:,i] for l in 1:L1_2) .== 0)
    @constraint(m1, x .+ sum(λ_2[l]*ω_2_list[l] for l in 1:L1_2) .== 0)

    @constraint(m1, [i in 1:n, j in i:n], X[i,j] + sum(λ_2[l]*μbar_2_list[l][i,j] for l in 1:L1_2) == 0)
    @constraint(m1, [i in 1:n, j in 1:n], x[i] - X[i,j] + sum(λ_2[l]*νbar_2_list[l][i,j] for l in 1:L1_2) == 0)
    @constraint(m1, [i in 1:n, j in i:n], 1 - x[i] - x[j] + X[i,j] + sum(λ_2[l]*ξ_2_list[l][i,j] for l in 1:L1_2) == 0)

    @constraint(m1, [i in 1:n], x[i] + sum(λ_2[l]*μ_2_list[l][i] for l in 1:L1_2) == 0)
    @constraint(m1, [i in 1:n], 1 - x[i] + sum(λ_2[l]*ν_2_list[l][i] for l in 1:L1_2) == 0)
    @constraint(m1, 1 + sum(λ_2[l]*γ_2_list[l] for l in 1:L1_2) == 0)

    @constraint(m1, [i in 1:n, j in i:n], [Ybar_2[:,:,i,j] V[:,i,j]; (V[:,i,j])' X[i,j]] in PSDCone())
    @constraint(m1, [i in 1:n, j in 1:n], [Rbar_2[:,:,i,j] (X[:,i].-V[:,i,j]); (X[:,i].-V[:,i,j])' (x[i]-X[i,j])] in PSDCone())
    @constraint(m1, [i in 1:n, j in i:n], [E_2[:,:,i,j] (x.-X[:,i].-X[:,j].+V[:,i,j]);
                                          (x.-X[:,i].-X[:,j].+V[:,i,j])' (1-x[i]-x[j]+X[i,j])] in PSDCone())

    @constraint(m1, [i in 1:n], [Y_2[:,:,i]  X[:,i]; (X[:,i])' x[i]] in PSDCone())
    @constraint(m1, [i in 1:n], [R_2[:,:,i]  (x.-X[:,i]); (x.-X[:,i])' (1-x[i])] in PSDCone())
    @constraint(m1, [Y_2[:,:,n+1] x; x' 1] in PSDCone())


    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], C*X[:,i] .<= x[i]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    @constraint(m1, [i in 1:n, j in 1:n], C*V[:,i,j] .<= X[i,j]*d)
    # @constraint(m1, [k in 1:n], d*X[:,k]'*C' .+ C*X[:,k]*d' .<= C*V[k,:,:]*C' .+ x[k]*d*d')

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
    c4_1, c3_1, c2_1, c1_1, c0_1 = p_1_coeffs[1], p_1_coeffs[2], p_1_coeffs[3], p_1_coeffs[4], p_1_coeffs[5]
    c4_2, c3_2, c2_2, c1_2, c0_2 = p_2_coeffs[1], p_2_coeffs[2], p_2_coeffs[3], p_2_coeffs[4], p_2_coeffs[5]
    m1 = Model(Ipopt.Optimizer)
    @variable(m1, x[1:n])
    for j in 1:n
        JuMP.set_start_value(x[j], x0[j])
    end
    # Constraints
    @constraint(m1, C*x .<= d)
    @NLconstraint(m1, sum(c4_2[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n) +
                      sum(c3_2[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n) +
                      sum(c2_2[i,j]*x[i]*x[j] for i=1:n, j=1:n) +
                      sum(c1_2[i]*x[i] for i=1:n) + c0_2 <= 0)
    # Objective function
    @NLexpression(m1, obj_term, sum(c4_1[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n) +
                                sum(c3_1[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n) +
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
    c4_1, c3_1, c2_1, c1_1, c0_1 = p_1_coeffs[1], p_1_coeffs[2], p_1_coeffs[3], p_1_coeffs[4], p_1_coeffs[5]
    best_ub = 1e6
    for i in 1:length(X)
        x = X[i]
        if is_point_feasible_X2(C,d,p_2_coeffs,x)
            cur_ub = poly4_init(c4_1,c3_1,c2_1,c1_1,c0_1,x)
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

    c4_1, c3_1, c2_1, c1_1, c0_1 = p_1_coeffs[1], p_1_coeffs[2], p_1_coeffs[3], p_1_coeffs[4], p_1_coeffs[5]
    c4_2, c3_2, c2_2, c1_2, c0_2 = p_2_coeffs[1], p_2_coeffs[2], p_2_coeffs[3], p_2_coeffs[4], p_2_coeffs[5]

    unc_set_list_1 = get_uncertainty_set_z1(c4_1,c3_1,c2_1,c1_1,c0_1)
    unc_set_list_2 = get_uncertainty_set_z1(c4_2,c3_2,c2_2,c1_2,c0_2)

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

############################    X_3   ####################################

function is_point_feasible_X3(C,d,x,α)
    n = size(x,1)
    res = (sum(C*x .<= d)  == size(C,1)) && log(sum(exp(x[i]) for i=1:n)) <= α
    return res
end

function solve_rpt_relaxation_X3_best_slc(unc_set_list,C,d,use_lmi,α)
    L, n = size(C)
    Abar_list, Bbar_list, C_list = unc_set_list[1], unc_set_list[2], unc_set_list[3]
    cbar_list, dbar_list, e_list = unc_set_list[4], unc_set_list[5], unc_set_list[6]
    μbar_list, νbar_list, ξ_list = unc_set_list[7], unc_set_list[8], unc_set_list[9]
    A_list, B_list, c_list = unc_set_list[10], unc_set_list[11], unc_set_list[12]
    d_list, μ_list, ν_list = unc_set_list[13], unc_set_list[14], unc_set_list[15]
    Ξ_list, ω_list, γ_list = unc_set_list[16], unc_set_list[17], unc_set_list[18]
    s1 = unc_set_list[19]
    L1 = length(s1)

    m1 = Model(Mosek.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, X[1:n,1:n]>=0, Symmetric)
    @variable(m1, V[1:n,1:n,1:n]>=0)
    @variable(m1, Ybar[1:n,1:n,1:n,1:n])
    @variable(m1, Rbar[1:n,1:n,1:n,1:n])
    @variable(m1, E[1:n,1:n,1:n,1:n])
    @variable(m1, Y[1:n,1:n,1:n+1])
    @variable(m1, R[1:n,1:n,1:n])
    @variable(m1, λ[1:L1])
    @variable(m1, z[1:n])
    @variable(m1, Z_new[1:n,1:n], Symmetric)
    @variable(m1, Q_new[1:n,1:n])

    @constraint(m1, [i in 1:n, j in i:n], -Ybar[:,:,i,j] .- sum(λ[l]*Abar_list[l][:,:,i,j] for l in 1:L1) in PSDCone())
    @constraint(m1, [i in 1:n, j in 1:n], -Rbar[:,:,i,j] .- sum(λ[l]*Bbar_list[l][:,:,i,j] for l in 1:L1) in PSDCone())
    @constraint(m1, [i in 1:n, j in i:n], -E[:,:,i,j] .- sum(λ[l]*C_list[l][:,:,i,j] for l in 1:L1) in PSDCone())

    @constraint(m1, [i in 1:n], -Y[:,:,i] .- sum(λ[l]*A_list[l][:,:,i] for l in 1:L1) in PSDCone())
    @constraint(m1, [i in 1:n], -R[:,:,i] .- sum(λ[l]*B_list[l][:,:,i] for l in 1:L1) in PSDCone())
    @constraint(m1, -Y[:,:,n+1] .- sum(λ[l]*Ξ_list[l] for l in 1:L1) in PSDCone())

    @constraint(m1, [i in 1:n, j in i:n], V[:,i,j] .+ sum(λ[l]*cbar_list[l][:,i,j] for l in 1:L1) .== 0)
    @constraint(m1, [i in 1:n, j in 1:n], X[:,i] .- V[:,i,j] .+ sum(λ[l]*dbar_list[l][:,i,j] for l in 1:L1) .== 0)
    @constraint(m1, [i in 1:n, j in i:n], x .- X[:,i] .- X[:,j] .+ V[:,i,j] .+ sum(λ[l]*e_list[l][:,i,j] for l in 1:L1) .== 0)

    @constraint(m1, [i in 1:n], X[:,i] .+ sum(λ[l]*c_list[l][:,i] for l in 1:L1) .== 0)
    @constraint(m1, [i in 1:n], x .- X[:,i] .+ sum(λ[l]*d_list[l][:,i] for l in 1:L1) .== 0)
    @constraint(m1, x .+ sum(λ[l]*ω_list[l] for l in 1:L1) .== 0)

    @constraint(m1, [i in 1:n, j in i:n], X[i,j] + sum(λ[l]*μbar_list[l][i,j] for l in 1:L1) == 0)
    @constraint(m1, [i in 1:n, j in 1:n], x[i] - X[i,j] + sum(λ[l]*νbar_list[l][i,j] for l in 1:L1) == 0)
    @constraint(m1, [i in 1:n, j in i:n], 1 - x[i] - x[j] + X[i,j] + sum(λ[l]*ξ_list[l][i,j] for l in 1:L1) == 0)

    @constraint(m1, [i in 1:n], x[i] + sum(λ[l]*μ_list[l][i] for l in 1:L1) == 0)
    @constraint(m1, [i in 1:n], 1 - x[i] + sum(λ[l]*ν_list[l][i] for l in 1:L1) == 0)
    @constraint(m1, 1 + sum(λ[l]*γ_list[l] for l in 1:L1) == 0)

    @constraint(m1, [i in 1:n, j in i:n], [Ybar[:,:,i,j] V[:,i,j]; (V[:,i,j])' X[i,j]] in PSDCone())
    @constraint(m1, [i in 1:n, j in 1:n], [Rbar[:,:,i,j] (X[:,i].-V[:,i,j]); (X[:,i].-V[:,i,j])' (x[i]-X[i,j])] in PSDCone())
    @constraint(m1, [i in 1:n, j in i:n], [E[:,:,i,j] (x.-X[:,i].-X[:,j].+V[:,i,j]);
                                          (x.-X[:,i].-X[:,j].+V[:,i,j])' (1-x[i]-x[j]+X[i,j])] in PSDCone())

    @constraint(m1, [i in 1:n], [Y[:,:,i]  X[:,i]; (X[:,i])' x[i]] in PSDCone())
    @constraint(m1, [i in 1:n], [R[:,:,i]  (x.-X[:,i]); (x.-X[:,i])' (1-x[i])] in PSDCone())
    @constraint(m1, [Y[:,:,n+1] x; x' 1] in PSDCone())

    @constraint(m1, C*x .<= d)
    @constraint(m1, [i in 1:n], C*X[:,i] .<= x[i]*d)
    @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
    @constraint(m1, [i in 1:n, j in 1:n], C*V[:,i,j] .<= X[i,j]*d)
    @constraint(m1, sum(z[i] for i in 1:n) <= 1)
    @constraint(m1, [i in 1:n], [x[i] - α, 1, z[i]] in MOI.ExponentialCone())
    @constraint(m1, C*x .- sum(C*Q_new[:,i] for i in 1:n) .<= d*(1-sum(z[i] for i in 1:n)))
    @constraint(m1, sum(Q_new[:,i] for i in 1:n) .<= x)
    @constraint(m1, 1 - 2*sum(z[i] for i in 1:n) + sum(Z_new[i,j] for i in 1:n for j in 1:n) >= 0)
    @constraint(m1, [i in 1:n], [x[i] - α - sum(Q_new[i,j] for j in 1:n) + α*sum(z[j] for j in 1:n),
                                 1 - sum(z[j] for j in 1:n),
                                 z[i]-sum(Z_new[j,i] for j in 1:n)] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in 1:n], [X[i,j] - α*x[j], x[j], Q_new[j,i]] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in 1:L], [d[j]*x[i]-α*d[j]-C[j,:]'*X[:,i]+α*C[j,:]'*x,
                                           d[j]-C[j,:]'*x,
                                           d[j]*z[i] - C[j,:]'*Q_new[:,i]] in MOI.ExponentialCone())
    @constraint(m1, [i in 1:n, j in i:n], [x[i]+x[j]-2*α,1,Z_new[i,j]] in MOI.ExponentialCone())
    if use_lmi
        @constraint(m1, [X Q_new x; Q_new' Z_new z; x' z' 1] in PSDCone())
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

function ub_ipopt_X3(C,d,c4,c3,c2,c1,c0,x0,α)
    n = size(c1,1)
    m1 = Model(Ipopt.Optimizer)
    @variable(m1, x[1:n]>=0)
    @variable(m1, t)
    for j in 1:n
        JuMP.set_start_value(x[j], x0[j])
    end
    @constraint(m1, C*x .<= d)

    @NLconstraint(m1, log(sum(exp(x[i]) for i in 1:n)) <= α)

    @NLexpression(m1, obj_term, sum(c4[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n) +
                                sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n) +
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

function get_ub_X3(X,C,d,c4,c3,c2,c1,c0,α)
    best_ub = 1e6
    for i in 1:length(X)
        x = X[i]
        if is_point_feasible_X3(C,d,x,α)
            cur_ub_1 = poly4_init(c4,c3,c2,c1,c0,x)
            cur_ub_2 = ub_ipopt_X3(C,d,c4,c3,c2,c1,c0,x,α)
            best_ub = min(best_ub,cur_ub_1,cur_ub_2)
        end
    end
    return best_ub
end

function rpt_bb_X3(C_init,d_init,c4,c3,c2,c1,c0,δ,use_lmi,α)
    C, d = C_init, d_init
    gen_hyper = 0

    unc_set_list = get_uncertainty_set_z1(c4,c3,c2,c1,c0)

    # Root Node
    res, x_lb, X_lb, lb = solve_rpt_relaxation_X3_best_slc(unc_set_list,C,d,use_lmi,α)
    X = calculate_candidate_vectors(x_lb,X_lb)
    ub = get_ub_X3(X,C,d,c4,c3,c2,c1,c0,α)

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
            ub_r = get_ub_X3(X_r,C_r,d_r,c4,c3,c2,c1,c0,α)
            if lb_r < UB
                push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r])
            end
        end
        # Left child
        res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_X3_best_slc(unc_set_list,C_l,d_l,use_lmi,α)
        if res_l
            X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
            ub_l = get_ub_X3(X_l,C_l,d_l,c4,c3,c2,c1,c0,α)
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

n = 20
α = 3
c4, c3, c2, c1, c0 = generate_degree4_polynomial(n, 0.5)
C, d = zeros(n,n) + I, ones(n)


t1 = time_ns()
x_opt_1, obj_opt_1, gen_hyper_1, ub_list_1, lb_list_1, hyp_list = rpt_bb_X1(C,d,c4,c3,c2,c1,c0,δ,use_lmi)
t2 = time_ns()
total_time_1 = (t2-t1)*10^(-9)


t1 = time_ns()
x_opt_3, obj_opt_3, gen_hyper_3, ub_list_3, lb_list_3, hyp_list = rpt_bb_X3(C,d,c4,c3,c2,c1,c0,δ,use_lmi,α)
t2 = time_ns()
total_time_3 = (t2-t1)*10^(-9)


c4_2, c3_2, c2_2, c1_2, c0_2 = generate_degree4_polynomial(n, 0.5)
p_1_coeffs = [c4,c3,c2,c1,c0]
p_2_coeffs = [c4_2,c3_2,c2_2,c1_2,c0_2]

t1 = time_ns()
x_opt_2, obj_opt_2, gen_hyper_2, ub_list_2, lb_list_2, hyp_list = rpt_bb_X2(C,d,p_1_coeffs,p_2_coeffs,δ,use_lmi)
t2 = time_ns()
total_time_2 = (t2-t1)*10^(-9)

#########################    Save Data    #####################################


c3_mat = reshape(c3, :, size(c3, 3))
writedlm("Polynomial_Instances/Poly4_n20_instance10/c0.csv",  Vector([c0]), ',')
writedlm("Polynomial_Instances/Poly4_n20_instance10/c1.csv",  c1, ',')
writedlm("Polynomial_Instances/Poly4_n20_instance10/c2.csv",  c2, ',')
writedlm("Polynomial_Instances/Poly4_n20_instance10/c3.csv",  c3_mat, ',')
h5write("Polynomial_Instances/Poly4_n20_instance10/c4.h5", "data", c4)


c3_2_mat = reshape(c3_2, :, size(c3_2, 3))
writedlm("Polynomial_Instances/Poly4_n10_instance2/c0_2.csv",  Vector([c0_2]), ',')
writedlm("Polynomial_Instances/Poly4_n10_instance2/c1_2.csv",  c1_2, ',')
writedlm("Polynomial_Instances/Poly4_n10_instance2/c2_2.csv",  c2_2, ',')
writedlm("Polynomial_Instances/Poly4_n10_instance2/c3_2.csv",  c3_2_mat, ',')
h5write("Polynomial_Instances/Poly4_n10_instance2/c4_2.h5", "data", c4_2)


######################     Run Averages     ################################

δ = 1e-4
use_lmi = false

run_X1, run_X2, run_X3 = true, false, false
n = 20
α = 3
C, d = zeros(n,n) + I, ones(n)
obj_vals, total_times, gen_hyper_list = [], [], []
lower_bounds_list, upper_bounds_list = [], []

for i in 1:10
    c4_mat = h5open("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c4.h5", "r")
    c4 = read(c4_mat["data"])
    c3_mat = CSV.read("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c3.csv", DataFrame, header=false)
    c3 = reshape(Matrix(c3_mat), n, n, n)
    c2_mat = CSV.read("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c2.csv", DataFrame, header=false)
    c2 = Matrix(c2_mat)
    c1_mat = CSV.read("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c1.csv", DataFrame, header=false)
    c1 = Vector(c1_mat[:,1])
    c0_mat = CSV.read("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c0.csv", DataFrame, header=false)
    c0 = c0_mat[1,1]

    # c4_mat_2 = h5open("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c4_2.h5", "r")
    # c4_2 = read(c4_mat_2["data"])
    # c3_mat_2 = CSV.read("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c3_2.csv", DataFrame, header=false)
    # c3_2 = reshape(Matrix(c3_mat_2), n, n, n)
    # c2_mat_2 = CSV.read("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c2_2.csv", DataFrame, header=false)
    # c2_2 = Matrix(c2_mat_2)
    # c1_mat_2 = CSV.read("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c1_2.csv", DataFrame, header=false)
    # c1_2 = Vector(c1_mat_2[:,1])
    # c0_mat_2 = CSV.read("Polynomial_Instances/Poly4_n"*string(n)*"_instance"*string(i)*"/c0_2.csv", DataFrame, header=false)
    # c0_2 = c0_mat_2[1,1]

    p_1_coeffs = [c4,c3,c2,c1,c0]
    p_2_coeffs = [c4_2,c3_2,c2_2,c1_2,c0_2]

    if run_X1
        t1 = time_ns()
        x_opt, obj_opt, gen_hyper, ub_list, lb_list, hyp_list = rpt_bb_X1(C,d,c4,c3,c2,c1,c0,δ,use_lmi)
        t2 = time_ns()
        total_time = (t2-t1)*10^(-9)
    elseif run_X2
        t1 = time_ns()
        x_opt, obj_opt, gen_hyper, ub_list, lb_list, hyp_list = rpt_bb_X2(C,d,p_1_coeffs,p_2_coeffs,δ,use_lmi)
        t2 = time_ns()
        total_time = (t2-t1)*10^(-9)
    elseif run_X3
        t1 = time_ns()
        x_opt, obj_opt, gen_hyper, ub_list, lb_list, hyp_list = rpt_bb_X3(C,d,c4,c3,c2,c1,c0,δ,use_lmi,α)
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


#########################     OLD    ####################################



############################   X_2    ##################################

# function get_uncertainty_set_z2(c4,c3,c2,c1,c0)
#     n = size(c1,1)
#     A_list = []
#     B_list = []
#     C_list = []
#     c_list = []
#     d_list = []
#     e_list = []
#     μ_list = []
#     ν_list = []
#     ξ_list = []
#     s1 = []
#     # fourth degree terms
#     for i in 1:n
#         for j in i:n
#             for k in j:n
#                 for l in k:n
#                     A = zeros(n,n,n,n)
#                     B = zeros(n,n,n,n)
#                     C = zeros(n,n,n,n)
#                     c = zeros(n,n,n)
#                     d = zeros(n,n,n)
#                     e = zeros(n,n,n)
#                     μ = zeros(n,n)
#                     ν = zeros(n,n)
#                     ξ = zeros(n,n)
#                     if i != j && i != k && i != l && j != k && j != l && k != l
#                         A[k,l,i,j] = 1
#                         A[l,k,i,j] = 1
#                         A[j,l,i,k] = 1
#                         A[l,j,i,k] = 1
#                         A[j,k,i,l] = 1
#                         A[k,j,i,l] = 1
#                         A[i,l,j,k] = 1
#                         A[l,i,j,k] = 1
#                         A[i,k,j,l] = 1
#                         A[k,i,j,l] = 1
#                         A[i,j,k,l] = 1
#                         A[j,i,k,l] = 1
#
#                         C[k,l,i,j] = 1
#                         C[l,k,i,j] = 1
#                         C[j,l,i,k] = 1
#                         C[l,j,i,k] = 1
#                         C[j,k,i,l] = 1
#                         C[k,j,i,l] = 1
#                         C[i,l,j,k] = 1
#                         C[l,i,j,k] = 1
#                         C[i,k,j,l] = 1
#                         C[k,i,j,l] = 1
#                         C[i,j,k,l] = 1
#                         C[j,i,k,l] = 1
#
#                         B[k,l,i,j] = -1
#                         B[l,k,i,j] = -1
#                         B[k,l,j,i] = -1
#                         B[l,k,j,i] = -1
#                         B[j,l,i,k] = -1
#                         B[l,j,i,k] = -1
#                         B[j,l,k,i] = -1
#                         B[l,j,k,i] = -1
#                         B[j,k,i,l] = -1
#                         B[k,j,i,l] = -1
#                         B[j,k,l,i] = -1
#                         B[k,j,l,i] = -1
#                         B[i,l,j,k] = -1
#                         B[l,i,j,k] = -1
#                         B[i,l,k,j] = -1
#                         B[l,i,k,j] = -1
#                         B[i,k,j,l] = -1
#                         B[k,i,j,l] = -1
#                         B[i,k,l,j] = -1
#                         B[k,i,l,j] = -1
#                         B[i,j,k,l] = -1
#                         B[j,i,k,l] = -1
#                         B[i,j,l,k] = -1
#                         B[j,i,l,k] = -1
#
#                         push!(s1, c4[i,j,k,l]+c4[i,j,l,k]+c4[i,k,j,l]+c4[i,k,l,j]+
#                                   c4[i,l,j,k]+c4[i,l,k,j]+c4[j,i,k,l]+c4[j,i,l,k]+
#                                   c4[j,k,i,l]+c4[j,k,l,i]+c4[j,l,i,k]+c4[j,l,k,i]+
#                                   c4[k,i,j,l]+c4[k,i,l,j]+c4[k,j,i,l]+c4[k,j,l,i]+
#                                   c4[k,l,i,j]+c4[k,l,j,i]+c4[l,i,j,k]+c4[l,i,k,j]+
#                                   c4[l,j,i,k]+c4[l,j,k,i]+c4[l,k,i,j]+c4[l,k,j,i])
#
#                     elseif i == j && i != k && i != l && k != l
#                         A[k,l,i,i] = 1
#                         A[l,k,i,i] = 1
#                         A[i,l,i,k] = 1
#                         A[l,i,i,k] = 1
#                         A[i,k,i,l] = 1
#                         A[k,i,i,l] = 1
#                         A[i,i,k,l] = 1
#
#                         C[k,l,i,i] = 1
#                         C[l,k,i,i] = 1
#                         C[i,l,i,k] = 1
#                         C[l,i,i,k] = 1
#                         C[i,k,i,l] = 1
#                         C[k,i,i,l] = 1
#                         C[i,i,k,l] = 1
#
#                         B[k,l,i,i] = -1
#                         B[l,k,i,i] = -1
#                         B[i,l,i,k] = -1
#                         B[l,i,i,k] = -1
#                         B[i,l,k,i] = -1
#                         B[l,i,k,i] = -1
#                         B[i,k,i,l] = -1
#                         B[k,i,i,l] = -1
#                         B[i,k,l,i] = -1
#                         B[k,i,l,i] = -1
#                         B[i,i,k,l] = -1
#                         B[i,i,l,k] = -1
#
#                         push!(s1, c4[i,i,k,l]+c4[i,i,l,k]+c4[i,k,i,l]+c4[i,k,l,i]+
#                                   c4[i,l,i,k]+c4[i,l,k,i]+c4[k,i,i,l]+c4[k,i,l,i]+
#                                   c4[k,l,i,i]+c4[l,i,i,k]+c4[l,i,k,i]+c4[l,k,i,i])
#
#
#                     elseif j == k && i != j && l != j && i != l
#                         A[j,j,i,l] = 1
#                         A[j,l,i,j] = 1
#                         A[l,j,i,j] = 1
#                         A[i,j,j,l] = 1
#                         A[j,i,j,l] = 1
#                         A[i,l,j,j] = 1
#                         A[l,i,j,j] = 1
#
#                         C[j,j,i,l] = 1
#                         C[j,l,i,j] = 1
#                         C[l,j,i,j] = 1
#                         C[i,j,j,l] = 1
#                         C[j,i,j,l] = 1
#                         C[i,l,j,j] = 1
#                         C[l,i,j,j] = 1
#
#                         B[j,j,i,l] = -1
#                         B[j,j,l,i] = -1
#                         B[j,l,i,j] = -1
#                         B[l,j,i,j] = -1
#                         B[j,l,j,i] = -1
#                         B[l,j,j,i] = -1
#                         B[i,j,j,l] = -1
#                         B[j,i,j,l] = -1
#                         B[i,j,l,j] = -1
#                         B[j,i,l,j] = -1
#                         B[i,l,j,j] = -1
#                         B[l,i,j,j] = -1
#
#                         push!(s1, c4[j,j,i,l]+c4[j,j,l,i]+c4[j,i,j,l]+c4[j,i,l,j]+
#                                   c4[j,l,j,i]+c4[j,l,i,j]+c4[i,j,j,l]+c4[i,j,l,j]+
#                                   c4[i,l,j,j]+c4[l,j,j,i]+c4[l,j,i,j]+c4[l,i,j,j])
#
#                     elseif k == l && i != k && j != k && i != j
#                         A[k,k,i,j] = 1
#                         A[j,k,i,k] = 1
#                         A[k,j,i,k] = 1
#                         A[i,k,j,k] = 1
#                         A[k,i,j,k] = 1
#                         A[i,j,k,k] = 1
#                         A[j,i,k,k] = 1
#
#                         C[k,k,i,j] = 1
#                         C[j,k,i,k] = 1
#                         C[k,j,i,k] = 1
#                         C[i,k,j,k] = 1
#                         C[k,i,j,k] = 1
#                         C[i,j,k,k] = 1
#                         C[j,i,k,k] = 1
#
#                         B[k,k,i,j] = -1
#                         B[k,k,j,i] = -1
#                         B[j,k,i,k] = -1
#                         B[k,j,i,k] = -1
#                         B[j,k,k,i] = -1
#                         B[k,j,k,i] = -1
#                         B[i,k,j,k] = -1
#                         B[k,i,j,k] = -1
#                         B[i,k,k,j] = -1
#                         B[k,i,k,j] = -1
#                         B[i,j,k,k] = -1
#                         B[j,i,k,k] = -1
#
#                         push!(s1, c4[k,k,i,j]+c4[k,k,j,i]+c4[k,i,k,j]+c4[k,i,j,k]+
#                                   c4[k,j,k,i]+c4[k,j,i,k]+c4[i,k,k,j]+c4[i,k,j,k]+
#                                   c4[i,j,k,k]+c4[j,k,k,i]+c4[j,k,i,k]+c4[j,i,k,k])
#
#                     elseif i == j && k == l && k != j
#                         A[k,k,i,i] = 1
#                         A[i,i,k,k] = 1
#                         A[i,k,i,k] = 1
#                         A[k,i,i,k] = 1
#
#                         C[k,k,i,i] = 1
#                         C[i,i,k,k] = 1
#                         C[i,k,i,k] = 1
#                         C[k,i,i,k] = 1
#
#                         B[k,k,i,i] = -1
#                         B[i,i,k,k] = -1
#                         B[i,k,i,k] = -1
#                         B[k,i,i,k] = -1
#                         B[i,k,k,i] = -1
#                         B[k,i,k,i] = -1
#
#                         push!(s1, c4[i,i,k,k]+c4[i,k,i,k]+c4[i,k,k,i]+
#                                   c4[k,i,i,k]+c4[k,i,k,i]+c4[k,k,i,i])
#
#                     elseif i == j && j == k && l != k
#                         A[i,i,i,l] = 1
#                         A[i,l,i,i] = 1
#                         A[l,i,i,i] = 1
#
#                         C[i,i,i,l] = 1
#                         C[i,l,i,i] = 1
#                         C[l,i,i,i] = 1
#
#                         B[i,i,i,l] = -1
#                         B[i,i,l,i] = -1
#                         B[i,l,i,i] = -1
#                         B[l,i,i,i] = -1
#
#                         push!(s1, c4[i,i,i,l]+c4[i,i,l,i]+c4[i,l,i,i]+c4[l,i,i,i])
#
#                     elseif j == k && k == l && i != j
#                         A[j,j,i,j] = 1
#                         A[i,j,j,j] = 1
#                         A[j,i,j,j] = 1
#
#                         C[j,j,i,j] = 1
#                         C[i,j,j,j] = 1
#                         C[j,i,j,j] = 1
#
#                         B[j,j,i,j] = -1
#                         B[j,j,j,i] = -1
#                         B[i,j,j,j] = -1
#                         B[j,i,j,j] = -1
#
#                         push!(s1, c4[j,j,j,i]+c4[j,j,i,j]+c4[j,i,j,j]+c4[i,j,j,j])
#
#                     elseif i == j && j == k && k == l
#                         A[i,i,i,i] = 1
#                         C[i,i,i,i] = 1
#                         B[i,i,i,i] = -1
#                         push!(s1, c4[i,i,i,i])
#
#                     end
#                     push!(A_list, A)
#                     push!(B_list, B)
#                     push!(C_list, C)
#                     push!(c_list, c)
#                     push!(d_list, d)
#                     push!(e_list, e)
#                     push!(μ_list, μ)
#                     push!(ν_list, ν)
#                     push!(ξ_list, ξ)
#                 end
#             end
#         end
#     end
#
#     # third degree terms
#     for i in 1:n
#         for j in i:n
#             for k in j:n
#                 A = zeros(n,n,n,n)
#                 B = zeros(n,n,n,n)
#                 C = zeros(n,n,n,n)
#                 c = zeros(n,n,n)
#                 d = zeros(n,n,n)
#                 e = zeros(n,n,n)
#                 μ = zeros(n,n)
#                 ν = zeros(n,n)
#                 ξ = zeros(n,n)
#
#                 if i != j && i != k && j != k
#
#                     c[k,i,j] = 1
#                     c[j,i,k] = 1
#                     c[i,j,k] = 1
#
#                     e[k,i,j] = 1
#                     e[j,i,k] = 1
#                     e[i,j,k] = 1
#
#                     d[k,i,j] = -1
#                     d[k,j,i] = -1
#                     d[j,i,k] = -1
#                     d[j,k,i] = -1
#                     d[i,j,k] = -1
#                     d[i,k,j] = -1
#
#                     for l in 1:n
#                         if l != i
#                             B[j,k,i,l] = 1
#                             B[k,j,i,l] = 1
#                             B[j,k,l,i] = -1
#                             B[k,j,l,i] = -1
#                         end
#                         if l != j
#                             B[i,k,j,l] = 1
#                             B[k,i,j,l] = 1
#                             B[i,k,l,j] = -1
#                             B[k,i,l,j] = -1
#                         end
#                         if l != k
#                             B[i,j,k,l] = 1
#                             B[j,i,k,l] = 1
#                             B[i,j,l,k] = -1
#                             B[j,i,l,k] = -1
#                         end
#                     end
#
#                     A[j,k,i,i] = 2
#                     A[k,j,i,i] = 2
#                     A[i,k,j,j] = 2
#                     A[k,i,j,j] = 2
#                     A[i,j,k,k] = 2
#                     A[j,i,k,k] = 2
#
#                     C[j,k,i,i] = -2
#                     C[k,j,i,i] = -2
#                     C[i,k,j,j] = -2
#                     C[k,i,j,j] = -2
#                     C[i,j,k,k] = -2
#                     C[j,i,k,k] = -2
#
#                     for l in 1:n
#                         if l < i
#                             A[j,k,l,i] = 1
#                             A[k,j,l,i] = 1
#
#                             C[j,k,l,i] = -1
#                             C[k,j,l,i] = -1
#                         elseif l > i
#                             A[j,k,i,l] = 1
#                             A[k,j,i,l] = 1
#
#                             C[j,k,i,l] = -1
#                             C[k,j,i,l] = -1
#                         end
#                         if l < j
#                             A[i,k,l,j] = 1
#                             A[k,i,l,j] = 1
#
#                             C[i,k,l,j] = -1
#                             C[k,i,l,j] = -1
#                         elseif l > j
#                             A[i,k,j,l] = 1
#                             A[k,i,j,l] = 1
#
#                             C[i,k,j,l] = -1
#                             C[k,i,j,l] = -1
#                         end
#                         if l < k
#                             A[i,j,l,k] = 1
#                             A[j,i,l,k] = 1
#
#                             C[i,j,l,k] = -1
#                             C[j,i,l,k] = -1
#                         elseif l > k
#                             A[i,j,k,l] = 1
#                             A[j,i,k,l] = 1
#
#                             C[i,j,k,l] = -1
#                             C[j,i,k,l] = -1
#                         end
#                     end
#
#                     push!(s1, c3[i,j,k]+c3[i,k,j]+c3[j,i,k]+c3[j,k,i]+c3[k,i,j]+c3[k,j,i])
#
#                 elseif i == j && k != j
#                     c[k,i,i] = 1
#                     c[i,i,k] = 1
#
#                     e[k,i,i] = 1
#                     e[i,i,k] = 1
#
#                     d[k,i,i] = -1
#                     d[i,i,k] = -1
#                     d[i,k,i] = -1
#
#
#                     for l in 1:n
#                         if l != k
#                             B[i,i,k,l] = 1
#                             B[i,i,l,k] = -1
#                         end
#                         if l != i
#                             B[k,i,i,l] = 1
#                             B[i,k,i,l] = 1
#                             B[k,i,l,i] = -1
#                             B[i,k,l,i] = -1
#                         end
#                     end
#
#                     A[i,k,i,i] = 2
#                     A[k,i,i,i] = 2
#                     A[i,i,k,k] = 2
#
#                     C[i,k,i,i] = -2
#                     C[k,i,i,i] = -2
#                     C[i,i,k,k] = -2
#
#                     for l in 1:n
#                         if l < i
#                             A[i,k,l,i] = 1
#                             A[k,i,l,i] = 1
#
#                             C[i,k,l,i] = -1
#                             C[k,i,l,i] = -1
#                         elseif l > i
#                             A[i,k,i,l] = 1
#                             A[k,i,i,l] = 1
#
#                             C[i,k,i,l] = -1
#                             C[k,i,i,l] = -1
#                         end
#                         if l < k
#                             A[i,i,l,k] = 1
#
#                             C[i,i,l,k] = -1
#                         elseif l > k
#                             A[i,i,k,l] = 1
#
#                             C[i,i,k,l] = -1
#                         end
#                     end
#
#                     push!(s1, c3[i,i,k]+c3[i,k,i]+c3[k,i,i])
#
#                 elseif j == k && i != j
#                     c[j,i,j] = 1
#                     c[i,j,j] = 1
#
#                     e[j,i,j] = 1
#                     e[i,j,j] = 1
#
#                     d[j,i,j] = -1
#                     d[j,j,i] = -1
#                     d[i,j,j] = -1
#
#                     for l in 1:n
#                         if l != i
#                             B[j,j,i,l] = 1
#                             B[j,j,l,i] = -1
#                         end
#                         if l != j
#                             B[i,j,j,l] = 1
#                             B[j,i,j,l] = 1
#                             B[i,j,l,j] = -1
#                             B[j,i,l,j] = -1
#                         end
#                     end
#
#                     A[j,j,i,i] = 2
#                     A[i,j,j,j] = 2
#                     A[j,i,j,j] = 2
#
#                     C[j,j,i,i] = -2
#                     C[i,j,j,j] = -2
#                     C[j,i,j,j] = -2
#
#                     for l in 1:n
#                         if l > i
#                             A[j,j,i,l] = 1
#
#                             C[j,j,i,l] = -1
#                         elseif l < i
#                             A[j,j,l,i] = 1
#
#                             C[j,j,l,i] = -1
#                         end
#                         if l > j
#                             A[i,j,j,l] = 1
#                             A[j,i,j,l] = 1
#
#                             C[i,j,j,l] = -1
#                             C[j,i,j,l] = -1
#                         elseif l < j
#                             A[i,j,l,j] = 1
#                             A[j,i,l,j] = 1
#
#                             C[i,j,l,j] = -1
#                             C[j,i,l,j] = -1
#                         end
#                     end
#
#                     push!(s1, c3[j,j,i]+c3[j,i,j]+c3[i,j,j])
#
#                 elseif i == j && j == k
#                     c[i,i,i] = 1
#                     e[i,i,i] = 1
#                     d[i,i,i] = -1
#
#                     for l in 1:n
#                         if l != i
#                             B[i,i,i,l] = 1
#                             B[i,i,l,i] = -1
#                         end
#                     end
#
#                     A[i,i,i,i] = 2
#                     C[i,i,i,i] = -2
#                     for l in 1:n
#                         if l < i
#                             A[i,i,l,i] = 1
#                             C[i,i,l,i] = -1
#                         elseif l > i
#                             A[i,i,i,l] = 1
#                             C[i,i,i,l] = -1
#                         end
#                     end
#
#                     push!(s1, c3[i,i,i])
#                 end
#                 push!(A_list, A)
#                 push!(B_list, B)
#                 push!(C_list, C)
#                 push!(c_list, c)
#                 push!(d_list, d)
#                 push!(e_list, e)
#                 push!(μ_list, μ)
#                 push!(ν_list, ν)
#                 push!(ξ_list, ξ)
#             end
#         end
#     end
#
#     # second degree terms
#     for i in 1:n
#         for j in i:n
#             A = zeros(n,n,n,n)
#             B = zeros(n,n,n,n)
#             C = zeros(n,n,n,n)
#             c = zeros(n,n,n)
#             d = zeros(n,n,n)
#             e = zeros(n,n,n)
#             μ = zeros(n,n)
#             ν = zeros(n,n)
#             ξ = zeros(n,n)
#             if i != j
#                 μ[i,j] = 1
#                 ξ[i,j] = 1
#                 ν[i,j] = -1
#                 ν[j,i] = -1
#
#                 for l in 1:n
#                     if l != i
#                         d[j,i,l] = 1
#                         d[j,l,i] = -1
#                     end
#                     if l != j
#                         d[i,j,l] = 1
#                         d[i,l,j] = -1
#                     end
#                 end
#
#                 c[j,i,i] = 2
#                 c[i,j,j] = 2
#                 e[j,i,i] = -2
#                 e[i,j,j] = -2
#                 for l in 1:n
#                     if l < i
#                         c[j,l,i] = 1
#                         e[j,l,i] = -1
#                     elseif l > i
#                         c[j,i,l] = 1
#                         e[j,i,l] = -1
#                     end
#                     if l < j
#                         c[i,l,j] = 1
#                         e[i,l,j] = -1
#                     elseif l > j
#                         c[i,j,l] = 1
#                         e[i,j,l] = -1
#                     end
#                 end
#
#                 for k in 1:n
#                     for l in k:n
#                         A[i,j,k,l] = 1
#                         A[j,i,k,l] = 1
#                         C[i,j,k,l] = 1
#                         C[j,i,k,l] = 1
#                     end
#                 end
#
#                 for k in 1:n
#                     for l in 1:n
#                         B[i,j,k,l] = 1
#                         B[j,i,k,l] = 1
#                     end
#                 end
#
#                 push!(s1, c2[i,j]+c2[j,i])
#
#             else
#                 μ[i,i] = 1
#                 ξ[i,i] = 1
#                 ν[i,i] = -1
#
#                 for l in 1:n
#                     if l != i
#                         d[i,i,l] = 1
#                         d[i,l,i] = -1
#                     end
#                 end
#
#                 c[i,i,i] = 2
#                 e[i,i,i] = -2
#                 for l in 1:n
#                     if l < i
#                         c[i,l,i] = 1
#                         e[i,l,i] = -1
#                     elseif l > i
#                         c[i,i,l] = 1
#                         e[i,i,l] = -1
#                     end
#                 end
#
#                 for k in 1:n
#                     for l in k:n
#                         A[i,i,k,l] = 1
#                         C[i,i,k,l] = 1
#                     end
#                 end
#
#                 for k in 1:n
#                     for l in 1:n
#                         B[i,i,k,l] = 1
#                     end
#                 end
#
#                 push!(s1, c2[i,i])
#             end
#             push!(A_list, A)
#             push!(B_list, B)
#             push!(C_list, C)
#             push!(c_list, c)
#             push!(d_list, d)
#             push!(e_list, e)
#             push!(μ_list, μ)
#             push!(ν_list, ν)
#             push!(ξ_list, ξ)
#         end
#     end
#
#     # first degree terms
#     for i in 1:n
#         A = zeros(n,n,n,n)
#         B = zeros(n,n,n,n)
#         C = zeros(n,n,n,n)
#         c = zeros(n,n,n)
#         d = zeros(n,n,n)
#         e = zeros(n,n,n)
#         μ = zeros(n,n)
#         ν = zeros(n,n)
#         ξ = zeros(n,n)
#
#         for k in 1:n
#             for l in k:n
#                 c[i,k,l] = 1
#                 e[i,k,l] = 1
#             end
#         end
#
#         for k in 1:n
#             for l in 1:n
#                 d[i,k,l] = 1
#             end
#         end
#
#         μ[i,i] = 2
#         ξ[i,i] = -2
#         for l in 1:n
#             if l < i
#                 μ[l,i] = 1
#                 ξ[l,i] = -1
#             elseif l > i
#                 μ[i,l] = 1
#                 ξ[i,l] = -1
#             end
#         end
#
#         for l in 1:n
#             if l != i
#                 ν[i,l] = 1
#                 ν[l,i] = -1
#             end
#         end
#
#         push!(s1, c1[i])
#
#         push!(A_list, A)
#         push!(B_list, B)
#         push!(C_list, C)
#         push!(c_list, c)
#         push!(d_list, d)
#         push!(e_list, e)
#         push!(μ_list, μ)
#         push!(ν_list, ν)
#         push!(ξ_list, ξ)
#     end
#
#     # zero degree terms
#     A = zeros(n,n,n,n)
#     B = zeros(n,n,n,n)
#     C = zeros(n,n,n,n)
#     c = zeros(n,n,n)
#     d = zeros(n,n,n)
#     e = zeros(n,n,n)
#     μ = zeros(n,n)
#     ν = zeros(n,n)
#     ξ = zeros(n,n)
#
#     for i in 1:n
#         for j in i:n
#             μ[i,j] = 1
#             ξ[i,j] = 1
#         end
#     end
#     for i in 1:n
#         for j in 1:n
#             ν[i,j] = 1
#         end
#     end
#     push!(s1, c0)
#
#     push!(A_list, A)
#     push!(B_list, B)
#     push!(C_list, C)
#     push!(c_list, c)
#     push!(d_list, d)
#     push!(e_list, e)
#     push!(μ_list, μ)
#     push!(ν_list, ν)
#     push!(ξ_list, ξ)
#
#     # return final output
#     unc_set_list = [A_list, B_list, C_list, c_list, d_list, e_list, μ_list, ν_list, ξ_list, s1]
#     return unc_set_list
# end
#
#
# function solve_rpt_relaxation_X2_best_slc(unc_set_list,C,d,use_lmi)
#     L, n = size(C)
#     A_list, B_list, C_list = unc_set_list[1], unc_set_list[2], unc_set_list[3]
#     c_list, d_list, e_list = unc_set_list[4], unc_set_list[5], unc_set_list[6]
#     μ_list, ν_list, ξ_list = unc_set_list[7], unc_set_list[8], unc_set_list[9]
#     s1 = unc_set_list[10]
#     L1 = length(s1)
#
#     m1 = Model(Mosek.Optimizer)
#     @variable(m1, x[1:n])
#     @variable(m1, X[1:n,1:n], Symmetric)
#     @variable(m1, V[1:n,1:n,1:n])
#     @variable(m1, Y[1:n,1:n,1:n,1:n])
#     @variable(m1, R[1:n,1:n,1:n,1:n])
#     @variable(m1, E[1:n,1:n,1:n,1:n])
#     @variable(m1, λ[1:L1])
#
#     @constraint(m1, [i in 1:n], X[i,i] >= 0)
#
#     @constraint(m1, [i in 1:n, j in i:n], -Y[:,:,i,j] .- sum(λ[l]*A_list[l][:,:,i,j] for l in 1:L1) in PSDCone())
#     @constraint(m1, [i in 1:n, j in 1:n], -R[:,:,i,j] .- sum(λ[l]*B_list[l][:,:,i,j] for l in 1:L1) in PSDCone())
#     @constraint(m1, [i in 1:n, j in i:n], -E[:,:,i,j] .- sum(λ[l]*C_list[l][:,:,i,j] for l in 1:L1) in PSDCone())
#
#     @constraint(m1, [i in 1:n, j in i:n], x .+ X[:,i] .+ X[:,j] .+ V[:,i,j] .+ sum(λ[l]*c_list[l][:,i,j] for l in 1:L1) .== 0)
#     # for i in 1:n
#     #     for j in 1:n
#     #         if i != j
#     #             @constraint(m1, x .+ X[:,i] .- X[:,j] .- V[:,i,j] .+ sum(λ[l]*d_list[l][:,i,j] for l in 1:L1) .== 0)
#     #         else
#     #             @constraint(m1, x .- V[:,i,i] .+ sum(λ[l]*d_list[l][:,i,i] for l in 1:L1) .== 0)
#     #         end
#     #     end
#     # end
#     @constraint(m1, [i in 1:n, j in 1:n], x .+ X[:,i] .- X[:,j] .- V[:,i,j] .+ sum(λ[l]*d_list[l][:,i,j] for l in 1:L1) .== 0)
#     @constraint(m1, [i in 1:n, j in i:n], x .- X[:,i] .- X[:,j] .+ V[:,i,j] .+ sum(λ[l]*e_list[l][:,i,j] for l in 1:L1) .== 0)
#
#     @constraint(m1, [i in 1:n, j in i:n], 1 + x[i] + x[j] + X[i,j] + sum(λ[l]*μ_list[l][i,j] for l in 1:L1) == 0)
#     @constraint(m1, [i in 1:n, j in 1:n], 1 + x[i] - x[j] - X[i,j] + sum(λ[l]*ν_list[l][i,j] for l in 1:L1) == 0)
#     @constraint(m1, [i in 1:n, j in i:n], 1 - x[i] - x[j] + X[i,j] + sum(λ[l]*ξ_list[l][i,j] for l in 1:L1) == 0)
#
#     @constraint(m1, [i in 1:n, j in i:n], [Y[:,:,i,j] (x .+ X[:,i] .+ X[:,j] .+ V[:,i,j]);
#                                           (x .+ X[:,i] .+ X[:,j] .+ V[:,i,j])' (1+x[i]+x[j]+X[i,j])] in PSDCone())
#
#     @constraint(m1, [i in 1:n, j in 1:n], [R[:,:,i,j] (x .+ X[:,i] .- X[:,j] .- V[:,i,j]);
#                                           (x .+ X[:,i] .- X[:,j] .- V[:,i,j])' (1+x[i]-x[j]-X[i,j])] in PSDCone())
#
#     @constraint(m1, [i in 1:n, j in i:n], [E[:,:,i,j] (x .- X[:,i] .- X[:,j] .+ V[:,i,j]);
#                                           (x .- X[:,i] .- X[:,j] .+ V[:,i,j])' (1-x[i]-x[j]+X[i,j])] in PSDCone())
#
#     @constraint(m1, C*x .<= d)
#     @constraint(m1, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
#     @constraint(m1, [i in 1:n], C*V[:,i,i] .<= X[i,i]*d)
#     @constraint(m1, [i in 1:L, j in 1:L, k in 1:L],
#                     d[i]*d[j]*d[k] - d[k]*d[i]*C[j,:]'*x - d[k]*d[j]*C[i,:]'*x - d[i]*d[j]*C[k,:]'*x +
#                     d[k]*C[i,:]'*X*C[j,:] + d[i]*C[j,:]'*X*C[k,:] + d[j]*C[i,:]'*X*C[k,:] -
#                     C[k,:]'*(sum(C[i,l]*V[:,l,m]*C[j,m] for l in 1:n, m in 1:n)) >= 0)
#
#     # @constraint(m1, [i in 1:n, j in 1:n], C*V[:,i,j] .<= X[i,j]*d)
#     # @constraint(m1, [k in 1:n], d*X[:,k]'*C' .+ C*X[:,k]*d' .<= C*V[k,:,:]*C' .+ x[k]*d*d')
#
#     if use_lmi
#         @constraint(m1, [X  x;  x' 1] in PSDCone())
#     end
#     @objective(m1, Min, -sum(s1[j]*λ[j] for j in 1:L1))
#     optimize!(m1)
#     if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
#         return true, JuMP.value.(x), JuMP.value.(X), objective_value(m1)
#     else
#         return false, zeros(n), zeros(n,n), 1e6
#     end
# end
#
# function ub_ipopt_X2(C,d,c4,c3,c2,c1,c0,x0)
#     n = size(C,2)
#     model = Model(Ipopt.Optimizer)
#     @variable(model, x[1:n])
#     for j in 1:n
#         JuMP.set_start_value(x[j], x0[j])
#     end
#     @constraint(model, C*x .<= d)
#     @NLexpression(model, obj_term, sum(c4[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n) +
#                                    sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n) +
#                                    sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n) +
#                                    sum(c1[i]*x[i] for i=1:n) + c0)
#     @NLobjective(model, Min, obj_term)
#     optimize!(model)
#     if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED
#         return objective_value(model)
#     else
#         return 1e6
#     end
# end
#
# function is_point_feasible_X2(C,d,x)
#     res = (sum(C*x .<= d)  == size(C,1))
#     return res
# end
#
# function get_ub_X2(X,C,d,c4,c3,c2,c1,c0)
#     best_ub = 1e6
#     for i in 1:length(X)
#         x = X[i]
#         cur_ub = ub_ipopt_X2(C,d,c4,c3,c2,c1,c0,x)
#         if cur_ub < best_ub
#             best_ub = cur_ub
#         end
#         if is_point_feasible_X2(C,d,x)
#             cur_ub = poly4_init(c4,c3,c2,c1,c0,x)
#             if cur_ub < best_ub
#                 best_ub = cur_ub
#             end
#         end
#     end
#     return best_ub
# end
#
#
# function rpt_bb_X2(C_init,d_init,c4,c3,c2,c1,c0,δ,use_lmi)
#     C, d = C_init, d_init
#     gen_hyper = 0
#
#     unc_set_list = get_uncertainty_set_z2(c4,c3,c2,c1,c0)
#
#     # Root Node
#     res, x_lb, X_lb, lb = solve_rpt_relaxation_X2_best_slc(unc_set_list,C,d,use_lmi)
#     X = calculate_candidate_vectors(x_lb,X_lb)
#     ub = get_ub_X2(X,C,d,c4,c3,c2,c1,c0)
#
#     x_cur, X_cur = x_lb, X_lb
#
#     UB, LB, opt_sol, opt_val = ub, lb, x_lb, ub
#
#     nodes_list = []
#     ub_list, lb_list, hyp_list = [], [], []
#     push!(ub_list, ub)
#     push!(lb_list, lb)
#
#     t0_1 = time_ns()
#     total_time = 0.0
#
#     while UB - LB > δ && total_time < 3600
#         res, f_opt, l_opt = generate_hyperplane_eigen(x_cur,X_cur)
#
#         f_r, l_r, f_l, l_l  = f_opt, l_opt, -f_opt, -l_opt
#         C_r, d_r, C_l, d_l = vcat(C,f_r'), vcat(d,l_r), vcat(C,f_l'), vcat(d,l_l)
#         gen_hyper += 1
#         push!(hyp_list, [f_opt, l_opt])
#
#         # Right child
#         res_r, x_lb_r, X_lb_r, lb_r = solve_rpt_relaxation_X2_best_slc(unc_set_list,C_r,d_r,use_lmi)
#         if res_r
#             X_r = calculate_candidate_vectors(x_lb_r,X_lb_r)
#             ub_r = get_ub_X2(X_r,C_r,d_r,c4,c3,c2,c1,c0)
#             if lb_r < UB
#                 push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r])
#             end
#         end
#         # Left child
#         res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_X2_best_slc(unc_set_list,C_l,d_l,use_lmi)
#         if res_l
#             X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
#             ub_l = get_ub_X2(X_l,C_l,d_l,c4,c3,c2,c1,c0)
#             if lb_l < UB
#                 push!(nodes_list,[ub_l,lb_l,x_lb_l,X_lb_l,C_l,d_l])
#             end
#         end
#
#         if isempty(nodes_list)
#             if res_r
#                 push!(ub_list, ub_r)
#                 push!(lb_list, lb_r)
#             end
#             if res_l
#                 push!(ub_list, ub_l)
#                 push!(lb_list, lb_l)
#             end
#             break
#         else
#             ind = argmin([nodes_list[i][2] for i in 1:length(nodes_list)])
#             cur_node = nodes_list[ind]
#             deleteat!(nodes_list, ind)
#             ub, lb = cur_node[1], cur_node[2]
#             x_cur, X_cur = cur_node[3], cur_node[4]
#             C, d = cur_node[5], cur_node[6]
#             LB = lb
#             if ub < UB
#                 UB = ub
#                 opt_sol, opt_val = cur_node[3], cur_node[1]
#             end
#             push!(ub_list, ub)
#             push!(lb_list, lb)
#         end
#         t0_2 = time_ns()
#         total_time = (t0_2-t0_1)*10^(-9)
#     end
#     return opt_sol, opt_val, gen_hyper, ub_list, lb_list, hyp_list
# end
#
# function get_scc_coeffs(unc_set_list)
#     A_list, B_list, C_list = unc_set_list[1], unc_set_list[2], unc_set_list[3]
#     c_list, d_list, e_list = unc_set_list[4], unc_set_list[5], unc_set_list[6]
#     μ_list, ν_list, ξ_list = unc_set_list[7], unc_set_list[8], unc_set_list[9]
#     s1 = unc_set_list[10]
#     L1 = length(s1)
#     n = size(c_list[1],1)
#
#
#     m1 = Model(Mosek.Optimizer)
#     @variable(m1, P[1:n,1:n,1:n,1:n])
#     @variable(m1, Q[1:n,1:n,1:n,1:n])
#     @variable(m1, T[1:n,1:n,1:n,1:n])
#
#     @variable(m1, r[1:n,1:n,1:n])
#     @variable(m1, f[1:n,1:n,1:n])
#     @variable(m1, h[1:n,1:n,1:n])
#
#     @variable(m1, w[1:n,1:n])
#     @variable(m1, g[1:n,1:n])
#     @variable(m1, s[1:n,1:n])
#
#     @constraint(m1, [l in 1:L1], sum( dot(A_list[l][:,:,i,j], P[:,:,i,j]) +
#                                       dot(C_list[l][:,:,i,j], T[:,:,i,j]) +
#                                       dot(c_list[l][:,i,j], r[:,i,j]) +
#                                       dot(e_list[l][:,i,j], h[:,i,j]) +
#                                       μ_list[l][i,j]*w[i,j] +
#                                       ξ_list[l][i,j]*s[i,j]
#                                       for i=1:n, j=i:n) +
#                                 sum( dot(B_list[l][:,:,i,j], Q[:,:,i,j]) +
#                                      dot(d_list[l][:,i,j], f[:,i,j]) +
#                                      ν_list[l][i,j]*g[i,j] for i=1:n, j=1:n)
#                                     == s1[l])
#
#     @constraint(m1, [i in 1:n, j in i:n], P[:,:,i,j] in PSDCone())
#     @constraint(m1, [i in 1:n, j in 1:n], Q[:,:,i,j] in PSDCone())
#     @constraint(m1, [i in 1:n, j in i:n], T[:,:,i,j] in PSDCone())
#
#     @objective(m1, Min, 0)
#     optimize!(m1)
#
#     if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
#         Z = [JuMP.value.(P), JuMP.value.(r), JuMP.value.(w),
#              JuMP.value.(Q), JuMP.value.(f), JuMP.value.(g),
#              JuMP.value.(T), JuMP.value.(h), JuMP.value.(s)]
#         return true, Z
#     else
#         return false, []
#     end
# end
#
# function poly4_init(c4,c3,c2,c1,c0,x)
#     n = size(x,1)
#     res_1 = sum(c4[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n)
#     res_2 = sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n)
#     res_3 = sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n)
#     res_4 = x'*c1 + c0
#     return res_1 + res_2 + res_3 + res_4
# end
#
# function poly4_scc_obj(Z,x)
#     n = size(x,1)
#     P_vals, r_vals, w_vals = Z[1], Z[2], Z[3]
#     Q_vals, f_vals, g_vals = Z[4], Z[5], Z[6]
#     T_vals, h_vals, s_vals = Z[7], Z[8], Z[9]
#
#     res_1 = sum((x[i]+1)*(x[j]+1)*(x'*P_vals[:,:,i,j]*x + x'*r_vals[:,i,j] + w_vals[i,j]) for i=1:n, j=1:n if j >= i)
#
#     res_2 = sum((x[i]+1)*(1-x[j])*(x'*Q_vals[:,:,i,j]*x + x'*f_vals[:,i,j] + g_vals[i,j]) for i=1:n, j=1:n)
#
#     res_3 = sum((1-x[i])*(1-x[j])*(x'*T_vals[:,:,i,j]*x + x'*h_vals[:,i,j] + s_vals[i,j]) for i=1:n, j=1:n if j >= i)
#
#     return res_1 + res_2 + res_3
# end
#
# function find_match(Z,c4,c3,c2,c1,c0,x)
#     n = size(x,1)
#     P_vals, r_vals, w_vals = Z[1], Z[2], Z[3]
#     Q_vals, f_vals, g_vals = Z[4], Z[5], Z[6]
#     T_vals, h_vals, s_vals = Z[7], Z[8], Z[9]
#
#     res_1_init = sum(c4[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n)
#     res_2_init = sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n)
#     res_3_init = sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n)
#     res_4_init = x'*c1
#     res_5_init = c0
#
#     res_1_new = sum(x[i]*x[j]*(x'*P_vals[:,:,i,j]*x) for i=1:n, j=1:n if j >= i) +
#                 sum(x[i]*x[j]*(x'*T_vals[:,:,i,j]*x) for i=1:n, j=1:n if j >= i) -
#                 sum(x[i]*x[j]*(x'*Q_vals[:,:,i,j]*x) for i=1:n, j=1:n)
#
#     res_2_new = sum(x[i]*x[j]*x'*r_vals[:,i,j] + x[i]*x[j]*x'*h_vals[:,i,j] for i=1:n, j=1:n if j >= i) +
#                 sum(x[i]*x'*P_vals[:,:,i,j]*x + x[j]*x'*P_vals[:,:,i,j]*x for i=1:n, j=1:n if j >= i) +
#                 sum(x[i]*x'*Q_vals[:,:,i,j]*x - x[j]*x'*Q_vals[:,:,i,j]*x - x[i]*x[j]*x'*f_vals[:,i,j] for i=1:n, j=1:n) -
#                 sum(x[i]*x'*T_vals[:,:,i,j]*x + x[j]*x'*T_vals[:,:,i,j]*x for i=1:n, j=1:n if j >= i)
#
#     res_3_new = sum(x[i]*x[j]*w_vals[i,j] + x[i]*x[j]*s_vals[i,j] for i=1:n, j=1:n if j >= i) +
#                 sum(x[i]*x'*f_vals[:,i,j] - x[j]*x'*f_vals[:,i,j] - x[i]*x[j]*g_vals[i,j]
#                     + x'*Q_vals[:,:,i,j]*x  for i=1:n, j=1:n) +
#                 sum(x[i]*x'*r_vals[:,i,j] + x[j]*x'*r_vals[:,i,j] + x'*P_vals[:,:,i,j]*x for i=1:n, j=1:n if j >= i)  -
#                 sum(x[i]*x'*h_vals[:,i,j] + x[j]*x'*h_vals[:,i,j] for i=1:n, j=1:n if j >= i) +
#                 sum(x'*T_vals[:,:,i,j]*x for i=1:n, j=1:n if j >= i)
#
#     res_4_new = sum(x[i]*g_vals[i,j] - x[j]*g_vals[i,j] + x'*f_vals[:,i,j] for i=1:n, j=1:n) +
#                 sum(x[i]*w_vals[i,j] + x[j]*w_vals[i,j] + x'*r_vals[:,i,j] for i=1:n, j=1:n if j >= i) +
#                 sum(x'*h_vals[:,i,j]- x[i]*s_vals[i,j] - x[j]*s_vals[i,j] for i=1:n, j=1:n if j >= i)
#
#     res_5_new = sum(w_vals[i,j]+s_vals[i,j] for i=1:n, j=1:n if j >= i) +
#                 sum(g_vals[i,j] for i=1:n, j=1:n)
#
#     old_res = [res_1_init,res_2_init,res_3_init,res_4_init,res_5_init]
#     new_res = [res_1_new,res_2_new,res_3_new,res_4_new,res_5_new]
#     final_res = hcat(old_res,new_res)
#
#     return final_res
# end
#
#
# n = 10
# c4, c3, c2, c1, c0 = generate_sparse_polynomial_degree4(n, 0.5)
# unc_set_list = get_uncertainty_set_z2(c4,c3,c2,c1,c0)
# res, Z = get_scc_coeffs(unc_set_list)
#
# # x0 = rand(n)
# x0 = rand(Uniform(-1, 1), n)
# final_res = find_match(Z,c4,c3,c2,c1,c0,x0)
#
# x0 = rand(Uniform(-1, 1), n)
# val_1 = poly4_init(c4,c3,c2,c1,c0,x0)
# val_2 = poly4_scc_obj(Z,x0)
#
#
#
# δ = 1e-4
# use_lmi = true
#
# n = 13
# c4, c3, c2, c1, c0 = generate_sparse_polynomial_degree4(n, 0.5)
# C, d = vcat(zeros(n,n) + I, zeros(n,n) - I), ones(2*n)
# # C, d = zeros(n,n) + I, ones(n)
#
# unc_set_list = get_uncertainty_set_z2(c4,c3,c2,c1,c0)
#
# res, x_lb, X_lb, lb = solve_rpt_relaxation_X2_best_slc(unc_set_list,C,d,use_lmi)
# X = calculate_candidate_vectors(x_lb,X_lb)
# ub = get_ub_X2(X,C,d,c4,c3,c2,c1,c0)
#
#
# t1 = time_ns()
# x_opt, obj_opt, gen_hyper, ub_list, lb_list, hyp_list = rpt_bb_X2(C,d,c4,c3,c2,c1,c0,δ,use_lmi)
# t2 = time_ns()
# total_time = (t2-t1)*10^(-9)
#
# println("Optimal value: ")
# println(obj_opt)
# println(" Total time: ")
# println(total_time)
# println("Generated hyperplanes: ")
# println(gen_hyper)
#




####################    used for debugging   #############################

#
# function get_scc_coeffs(unc_set_list)
#     A_list, B_list, C_list = unc_set_list[1], unc_set_list[2], unc_set_list[3]
#     c_list, d_list, e_list = unc_set_list[4], unc_set_list[5], unc_set_list[6]
#     μ_list, ν_list, ξ_list = unc_set_list[7], unc_set_list[8], unc_set_list[9]
#     s1 = unc_set_list[10]
#     L1 = length(s1)
#     n = size(c_list[1],1)
#
#
#     m1 = Model(Mosek.Optimizer)
#     @variable(m1, P[1:n,1:n,1:n,1:n])
#     @variable(m1, Q[1:n,1:n,1:n,1:n])
#     @variable(m1, T[1:n,1:n,1:n,1:n])
#
#     @variable(m1, r[1:n,1:n,1:n])
#     @variable(m1, f[1:n,1:n,1:n])
#     @variable(m1, h[1:n,1:n,1:n])
#
#     @variable(m1, w[1:n,1:n])
#     @variable(m1, g[1:n,1:n])
#     @variable(m1, s[1:n,1:n])
#
#     @constraint(m1, [l in 1:L1], sum( dot(A_list[l][:,:,i,j], P[:,:,i,j]) +
#                                       dot(C_list[l][:,:,i,j], T[:,:,i,j]) +
#                                       dot(c_list[l][:,i,j], r[:,i,j]) +
#                                       dot(e_list[l][:,i,j], h[:,i,j]) +
#                                       μ_list[l][i,j]*w[i,j] +
#                                       ξ_list[l][i,j]*s[i,j]
#                                       for i=1:n, j=i:n) +
#                                 sum( dot(B_list[l][:,:,i,j], Q[:,:,i,j]) +
#                                      dot(d_list[l][:,i,j], f[:,i,j]) +
#                                      ν_list[l][i,j]*g[i,j] for i=1:n, j=1:n)
#                                     == s1[l])
#
#     @constraint(m1, [i in 1:n, j in i:n], P[:,:,i,j] in PSDCone())
#     @constraint(m1, [i in 1:n, j in 1:n], Q[:,:,i,j] in PSDCone())
#     @constraint(m1, [i in 1:n, j in i:n], T[:,:,i,j] in PSDCone())
#
#     @objective(m1, Min, 0)
#     optimize!(m1)
#
#     if termination_status(m1) == MOI.OPTIMAL || termination_status(m1) == MOI.SLOW_PROGRESS
#         Z = [JuMP.value.(P), JuMP.value.(r), JuMP.value.(w),
#              JuMP.value.(Q), JuMP.value.(f), JuMP.value.(g),
#              JuMP.value.(T), JuMP.value.(h), JuMP.value.(s)]
#         return true, Z
#     else
#         return false, []
#     end
# end
#
# function poly4_init(c4,c3,c2,c1,c0,x)
#     n = size(x,1)
#     res_1 = sum(c4[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n)
#     res_2 = sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n)
#     res_3 = sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n)
#     res_4 = x'*c1 + c0
#     return res_1 + res_2 + res_3 + res_4
# end
#
# function poly4_scc_obj(Z,x)
#     n = size(x,1)
#     P_vals, r_vals, w_vals = Z[1], Z[2], Z[3]
#     Q_vals, f_vals, g_vals = Z[4], Z[5], Z[6]
#     T_vals, h_vals, s_vals = Z[7], Z[8], Z[9]
#
#     res_1 = sum(x[i]*x[j]*(x'*P_vals[:,:,i,j]*x + x'*r_vals[:,i,j] + w_vals[i,j]) for i=1:n, j=1:n if j >= i)
#
#     res_2 = sum(x[i]*(1-x[j])*(x'*Q_vals[:,:,i,j]*x + x'*f_vals[:,i,j] + g_vals[i,j]) for i=1:n, j=1:n)
#
#     res_3 = sum((1-x[i])*(1-x[j])*(x'*T_vals[:,:,i,j]*x + x'*h_vals[:,i,j] + s_vals[i,j]) for i=1:n, j=1:n if j >= i)
#
#     return res_1 + res_2 + res_3
# end
#
# function find_match(Z,c4,c3,c2,c1,c0,x)
#     n = size(x,1)
#     P_vals, r_vals, w_vals = Z[1], Z[2], Z[3]
#     Q_vals, f_vals, g_vals = Z[4], Z[5], Z[6]
#     T_vals, h_vals, s_vals = Z[7], Z[8], Z[9]
#
#     res_1_init = sum(c4[i,j,k,l]*x[i]*x[j]*x[k]*x[l] for i=1:n, j=1:n, k=1:n, l=1:n)
#     res_2_init = sum(c3[i,j,k]*x[i]*x[j]*x[k] for i=1:n, j=1:n, k=1:n)
#     res_3_init = sum(c2[i,j]*x[i]*x[j] for i=1:n, j=1:n)
#     res_4_init = x'*c1
#     res_5_init = c0
#
#     res_1_new = sum(x[i]*x[j]*(x'*P_vals[:,:,i,j]*x) for i=1:n, j=1:n if j >= i) +
#                 sum(x[i]*x[j]*(x'*T_vals[:,:,i,j]*x) for i=1:n, j=1:n if j >= i) -
#                 sum(x[i]*x[j]*(x'*Q_vals[:,:,i,j]*x) for i=1:n, j=1:n)
#
#     res_2_new = sum(x[i]*x[j]*x'*r_vals[:,i,j] + x[i]*x[j]*x'*h_vals[:,i,j] for i=1:n, j=1:n if j >= i) +
#                 sum(x[i]*x'*Q_vals[:,:,i,j]*x- x[i]*x[j]*x'*f_vals[:,i,j] for i=1:n, j=1:n) -
#                 sum(x[i]*x'*T_vals[:,:,i,j]*x + x[j]*x'*T_vals[:,:,i,j]*x for i=1:n, j=1:n if j >= i)
#
#     res_3_new = sum(x[i]*x[j]*w_vals[i,j] + x[i]*x[j]*s_vals[i,j] for i=1:n, j=1:n if j >= i) +
#                 sum(x[i]*x'*f_vals[:,i,j]- x[i]*x[j]*g_vals[i,j] for i=1:n, j=1:n) -
#                 sum(x[i]*x'*h_vals[:,i,j] + x[j]*x'*h_vals[:,i,j] for i=1:n, j=1:n if j >= i) +
#                 sum(x'*T_vals[:,:,i,j]*x for i=1:n, j=1:n if j >= i)
#
#     res_4_new = sum(x[i]*g_vals[i,j] for i=1:n, j=1:n) +
#                 sum(x'*h_vals[:,i,j]- x[i]*s_vals[i,j] - x[j]*s_vals[i,j] for i=1:n, j=1:n if j >= i)
#
#     res_5_new = sum(s_vals[i,j] for i=1:n, j=1:n if j >= i)
#
#     old_res = [res_1_init,res_2_init,res_3_init,res_4_init,res_5_init]
#     new_res = [res_1_new,res_2_new,res_3_new,res_4_new,res_5_new]
#     final_res = hcat(old_res,new_res)
#
#     return final_res
# end

#
# n = 10
# c4, c3, c2, c1, c0 = generate_sparse_polynomial_degree4(n, 0.5)
# unc_set_list = get_uncertainty_set_z1(c4,c3,c2,c1,c0)
# res, Z = get_scc_coeffs(unc_set_list)
# x0 = rand(n)
# final_res = find_match(Z,c4,c3,c2,c1,c0,x0)
# x0 = rand(n)
# val_1 = poly4_init(c4,c3,c2,c1,c0,x0)
# val_2 = poly4_scc_obj(Z,x0)
#



########################     OLD      ################################

# function generate_hyperplane_new(x_opt,X_opt,Z)
#     n_x = size(X_opt,1)
#     β_list, P_list, r_list, w_list = Z[1], Z[2], Z[3], Z[4]
#     γ_list, Q_list, f_list, g_list = Z[5], Z[6], Z[7], Z[8]
#     diffs = []
#     for i=1:n_x
#         val_1 = evaluate_poly3(β_list[i], P_list[i], r_list[i], w_list[i], X_opt[:,i]/x_opt[i])
#         val_2 = evaluate_poly3(γ_list[i], Q_list[i], f_list[i], g_list[i], (x_opt.-X_opt[:,i])/(1-x_opt[i]))
#         val_3 = evaluate_poly3(β_list[i], P_list[i], r_list[i], w_list[i], x_opt)
#         val_4 = evaluate_poly3(γ_list[i], Q_list[i], f_list[i], g_list[i], x_opt)
#         push!(diffs, abs(x_opt[i]*(val_1-val_3) + (1-x_opt[i])*(val_2-val_4)))
#     end
#     f = zeros(n_x)
#     ind_max = argmax(diffs)
#     f[ind_max] = 1
#     l = f'*x_opt
#     return false, f, l
# end
#
# function generate_hyperplane_new_2(x_opt,X_opt,coeff_list)
#     n_x = size(X_opt,1)
#     indices_max = []
#     for Z in coeff_list
#         β_list, P_list, r_list, w_list = Z[1], Z[2], Z[3], Z[4]
#         γ_list, Q_list, f_list, g_list = Z[5], Z[6], Z[7], Z[8]
#         diffs = []
#         for i=1:n_x
#             val_1 = evaluate_poly3(β_list[i], P_list[i], r_list[i], w_list[i], X_opt[:,i]/x_opt[i])
#             val_2 = evaluate_poly3(γ_list[i], Q_list[i], f_list[i], g_list[i], (x_opt.-X_opt[:,i])/(1-x_opt[i]))
#             val_3 = evaluate_poly3(β_list[i], P_list[i], r_list[i], w_list[i], x_opt)
#             val_4 = evaluate_poly3(γ_list[i], Q_list[i], f_list[i], g_list[i], x_opt)
#             push!(diffs, abs(x_opt[i]*(val_1-val_3) + (1-x_opt[i])*(val_2-val_4)))
#         end
#         ind_max = argmax(diffs)
#         push!(indices_max, ind_max)
#     end
#     indices_max = unique(indices_max)
#     f = zeros(n_x)
#     for i in indices_max
#         f[i] = 1
#     end
#     l = f'*x_opt
#     return false, f, l
# end

# function get_poly4_random_slc(c4,c3,c2,c1,c0)
#     n = size(c1,1)
#     model = Model(Gurobi.Optimizer)
#     set_optimizer_attribute(model, "OutputFlag", 0)
#     @variable(model, β[1:n,1:n,1:n,1:n])
#     @variable(model, P[1:n,1:n,1:n])
#     @variable(model, r[1:n,1:n])
#     @variable(model, w[1:n])
#     @variable(model, γ[1:n,1:n,1:n,1:n])
#     @variable(model, Q[1:n,1:n,1:n])
#     @variable(model, f[1:n,1:n])
#     @variable(model, g[1:n])
#     # auxiliary variables to model absolute values
#     @variable(model, β_abs[1:n,1:n,1:n,1:n])
#     @variable(model, P_abs[1:n,1:n,1:n])
#     @variable(model, γ_abs[1:n,1:n,1:n,1:n])
#     @variable(model, Q_abs[1:n,1:n,1:n])
#
#     ###################   Equality constraints to match coefficients  ###################
#     # zero degree terms
#     @constraint(model, sum(g[i] for i in 1:n) == c0)
#     # first degree terms
#     @constraint(model, [i in 1:n], w[i]-g[i]+sum(f[i,m] for m in 1:n) == c1[i])
#     # second degree terms
#     for i in 1:n
#         for j in i:n
#             if i != j
#                 @constraint(model, r[j,i]+r[i,j]-f[j,i]-f[i,j]+sum(Q[i,j,m]+Q[j,i,m] for m in 1:n) == c2[i,j]+c2[j,i])
#             else
#                 @constraint(model, r[i,i]-f[i,i]+sum(Q[i,i,m] for m in 1:n) == c2[i,i])
#             end
#         end
#     end
#     # third degree terms
#     for i in 1:n
#         for j in i:n
#             for k in j:n
#                 if i != j && i != k && j != k
#                     @constraint(model, P[j,k,i]+P[k,j,i]+P[i,k,j]+P[k,i,j]+P[i,j,k]+P[j,i,k]
#                                        -Q[j,k,i]-Q[k,j,i]-Q[i,k,j]-Q[k,i,j]-Q[i,j,k]-Q[j,i,k]+
#                                         sum(γ[i,j,k,m]+γ[i,k,j,m]+γ[j,i,k,m]+γ[j,k,i,m]+
#                                             γ[k,i,j,m]+γ[k,j,i,m] for m=1:n)
#                                         == c3[i,j,k]+c3[i,k,j]+c3[j,i,k]+c3[j,k,i]+c3[k,i,j]+c3[k,j,i])
#                 elseif i == j && j != k
#                     @constraint(model, P[i,i,k]+P[i,k,i]+P[k,i,i]-Q[i,i,k]-Q[i,k,i]-Q[k,i,i]+
#                                        sum(γ[k,i,i,m]+γ[i,k,i,m]+γ[i,i,k,m] for m=1:n)
#                                         == c3[k,i,i]+c3[i,k,i]+c3[i,i,k])
#                 elseif i == k && j != i
#                     @constraint(model, P[i,i,j]+P[j,i,i]+P[i,j,i]-Q[i,i,j]-Q[j,i,i]-Q[i,j,i] +
#                                        sum(γ[j,i,i,m]+γ[i,j,i,m]+γ[i,i,j,m] for m=1:n)
#                                         == c3[j,i,i]+c3[i,j,i]+c3[i,i,j])
#                 elseif j == k && i != j
#                     @constraint(model, P[j,j,i]+P[i,j,j]+P[j,i,j]-Q[j,j,i]-Q[i,j,j]-Q[j,i,j]+
#                                         sum(γ[i,j,j,m]+γ[j,i,j,m]+γ[j,j,i,m] for m=1:n)
#                                         == c3[i,j,j]+c3[j,i,j]+c3[j,j,i])
#                 elseif i == j && j == k
#                    @constraint(model, P[i,i,i]-Q[i,i,i]+sum(γ[i,i,i,m] for m=1:n) == c3[i,i,i])
#                end
#            end
#        end
#    end
#    # fourth degree terms
#    for i in 1:n
#        for j in i:n
#            for k in j:n
#                for l in k:n
#                     if i != j && i != k && i != l && j != k && j != l && k != l
#                         @constraint(model, β[j,k,l,i]+β[j,l,k,i]+β[k,j,l,i]+β[k,l,j,i]+β[l,j,k,i]+β[l,k,j,i]+
#                                            β[i,k,l,j]+β[i,l,k,j]+β[k,i,l,j]+β[k,l,i,j]+β[l,i,k,j]+β[l,k,i,j]+
#                                            β[i,j,l,k]+β[i,l,j,k]+β[j,i,l,k]+β[j,l,i,k]+β[l,i,j,k]+β[l,j,i,k]+
#                                            β[i,j,k,l]+β[i,k,j,l]+β[j,i,k,l]+β[j,k,i,l]+β[k,i,j,l]+β[k,j,i,l]-
#                                            γ[j,k,l,i]-γ[j,l,k,i]-γ[k,j,l,i]-γ[k,l,j,i]-γ[l,j,k,i]-γ[l,k,j,i]-
#                                            γ[i,k,l,j]-γ[i,l,k,j]-γ[k,i,l,j]-γ[k,l,i,j]-γ[l,i,k,j]-γ[l,k,i,j]-
#                                            γ[i,j,l,k]-γ[i,l,j,k]-γ[j,i,l,k]-γ[j,l,i,k]-γ[l,i,j,k]-γ[l,j,i,k]-
#                                            γ[i,j,k,l]-γ[i,k,j,l]-γ[j,i,k,l]-γ[j,k,i,l]-γ[k,i,j,l]-γ[k,j,i,l]
#                                            == c4[i,j,k,l]+c4[i,j,l,k]+c4[i,k,j,l]+c4[i,k,l,j]+c4[i,l,j,k]+
#                                            c4[i,l,k,j]+c4[j,i,k,l]+c4[j,i,l,k]+c4[j,k,i,l]+c4[j,k,l,i]+
#                                            c4[j,l,i,k]+c4[j,l,k,i]+c4[k,i,j,l]+c4[k,i,l,j]+c4[k,j,i,l]+
#                                            c4[k,j,l,i]+c4[k,l,i,j]+c4[k,l,j,i]+c4[l,i,j,k]+c4[l,i,k,j]+
#                                            c4[l,j,i,k]+c4[l,j,k,i]+c4[l,k,i,j]+c4[l,k,j,i])
#
#                     elseif i == j && i != k && i != l && k != l
#                         @constraint(model, β[i,k,l,i]+β[i,l,k,i]+β[k,i,l,i]+β[k,l,i,i]+β[l,i,k,i]+β[l,k,i,i]+
#                                            β[l,i,i,k]+β[i,l,i,k]+β[i,i,l,k]+β[k,i,i,l]+β[i,k,i,l]+β[i,i,k,l]-
#                                            γ[i,k,l,i]-γ[i,l,k,i]-γ[k,i,l,i]-γ[k,l,i,i]-γ[l,i,k,i]-γ[l,k,i,i]-
#                                            γ[l,i,i,k]-γ[i,l,i,k]-γ[i,i,l,k]-γ[k,i,i,l]-γ[i,k,i,l]-γ[i,i,k,l]
#                                            == c4[i,i,k,l]+c4[i,i,l,k]+c4[i,k,i,l]+c4[i,k,l,i]+c4[i,l,i,k]+
#                                            c4[i,l,k,i]+c4[k,i,i,l]+c4[k,i,l,i]+c4[k,l,i,i]+c4[l,i,i,k]+
#                                            c4[l,i,k,i]+c4[l,k,i,i])
#
#                     elseif i == k && i != j && i != l && j != l
#                         @constraint(model, β[i,j,l,i]+β[i,l,j,i]+β[j,i,l,i]+β[j,l,i,i]+β[l,i,j,i]+β[l,j,i,i]+
#                                            β[i,i,l,j]+β[i,l,i,j]+β[l,i,i,j]+β[i,i,j,l]+β[i,j,i,l]+β[j,i,i,l]-
#                                            γ[i,j,l,i]-γ[i,l,j,i]-γ[j,i,l,i]-γ[j,l,i,i]-γ[l,i,j,i]-γ[l,j,i,i]-
#                                            γ[i,i,l,j]-γ[i,l,i,j]-γ[l,i,i,j]-γ[i,i,j,l]-γ[i,j,i,l]-γ[j,i,i,l]
#                                            == c4[i,i,j,l]+c4[i,i,l,j]+c4[i,j,i,l]+c4[i,j,l,i]+c4[i,l,i,j]+
#                                            c4[i,l,j,i]+c4[j,i,i,l]+c4[j,i,l,i]+c4[j,l,i,i]+c4[l,i,i,j]+
#                                            c4[l,i,j,i]+c4[l,j,i,i])
#
#                     elseif i == l && i != j && i != k && j != k
#                         @constraint(model, β[i,j,k,i]+β[i,k,j,i]+β[j,i,k,i]+β[j,k,i,i]+β[k,i,j,i]+β[k,j,i,i]+
#                                            β[i,i,k,j]+β[i,k,i,j]+β[k,i,i,j]+β[i,i,j,k]+β[i,j,i,k]+β[j,i,i,k]-
#                                            γ[i,j,k,i]-γ[i,k,j,i]-γ[j,i,k,i]-γ[j,k,i,i]-γ[k,i,j,i]-γ[k,j,i,i]-
#                                            γ[i,i,k,j]-γ[i,k,i,j]-γ[k,i,i,j]-γ[i,i,j,k]-γ[i,j,i,k]-γ[j,i,i,k]
#                                            == c4[i,i,j,k]+c4[i,i,k,j]+c4[i,j,i,k]+c4[i,j,k,i]+c4[i,k,i,j]+
#                                            c4[i,k,j,i]+c4[j,i,i,k]+c4[j,i,k,i]+c4[j,k,i,i]+c4[k,i,i,j]+
#                                            c4[k,i,j,i]+c4[k,j,i,i])
#
#                     elseif j == l && i != j && k != j && i != k
#                         @constraint(model, β[j,j,k,i]+β[j,k,j,i]+β[k,j,j,i]+β[i,j,j,k]+β[j,i,j,k]+β[j,j,i,k]+
#                                            β[i,j,k,j]+β[i,k,j,j]+β[j,i,k,j]+β[j,k,i,j]+β[k,i,j,j]+β[k,j,i,j]-
#                                            γ[j,j,k,i]-γ[j,k,j,i]-γ[k,j,j,i]-γ[i,j,j,k]-γ[j,i,j,k]-γ[j,j,i,k]-
#                                            γ[i,j,k,j]-γ[i,k,j,j]-γ[j,i,k,j]-γ[j,k,i,j]-γ[k,i,j,j]-γ[k,j,i,j]
#                                            == c4[j,j,i,k]+c4[j,j,k,i]+c4[j,k,j,i]+c4[j,k,i,j]+c4[j,i,j,k]+
#                                            c4[j,i,k,j]+c4[i,j,j,k]+c4[i,j,k,j]+c4[i,k,j,j]+c4[k,j,j,i]+
#                                            c4[k,j,i,j]+c4[k,i,j,j])
#
#                     elseif j == k && i != j && l != j && l != i
#                         @constraint(model, β[j,j,l,i]+β[j,l,j,i]+β[l,j,j,i]+β[i,j,j,l]+β[j,i,j,l]+β[j,j,i,l]+
#                                            β[i,j,l,j]+β[i,l,j,j]+β[j,i,l,j]+β[j,l,i,j]+β[l,i,j,j]+β[l,j,i,j]-
#                                            γ[j,j,l,i]-γ[j,l,j,i]-γ[l,j,j,i]-γ[i,j,j,l]-γ[j,i,j,l]-γ[j,j,i,l]-
#                                            γ[i,j,l,j]-γ[i,l,j,j]-γ[j,i,l,j]-γ[j,l,i,j]-γ[l,i,j,j]-γ[l,j,i,j]
#                                            == c4[j,j,i,l]+c4[j,j,l,i]+c4[j,i,j,l]+c4[j,i,l,j]+c4[j,l,j,i]+
#                                            c4[j,l,i,j]+c4[i,j,j,l]+c4[i,j,l,j]+c4[i,l,j,j]+c4[l,j,j,i]+
#                                            c4[l,j,i,j]+c4[l,i,j,j])
#
#                     elseif k == l && i != k && j != k && i != j
#                         @constraint(model, β[j,k,k,i]+β[k,j,k,i]+β[k,k,j,i]+β[i,k,k,j]+β[k,i,k,j]+β[k,k,i,j]+
#                                            β[i,j,k,k]+β[i,k,j,k]+β[j,i,k,k]+β[j,k,i,k]+β[k,i,j,k]+β[k,j,i,k]-
#                                            γ[j,k,k,i]-γ[k,j,k,i]-γ[k,k,j,i]-γ[i,k,k,j]-γ[k,i,k,j]-γ[k,k,i,j]-
#                                            γ[i,j,k,k]-γ[i,k,j,k]-γ[j,i,k,k]-γ[j,k,i,k]-γ[k,i,j,k]-γ[k,j,i,k]
#                                            == c4[k,k,i,j]+c4[k,k,j,i]+c4[k,i,k,j]+c4[k,i,j,k]+c4[k,j,k,i]+
#                                            c4[k,j,i,k]+c4[i,k,k,j]+c4[i,k,j,k]+c4[i,j,k,k]+c4[j,k,k,i]+
#                                            c4[j,k,i,k]+c4[j,i,k,k])
#
#                     elseif i == j && k == l && i != k
#                         @constraint(model, β[i,k,k,i]+β[k,i,k,i]+β[k,k,i,i]+β[k,i,i,k]+β[i,k,i,k]+β[i,i,k,k]-
#                                            γ[i,k,k,i]-γ[k,i,k,i]-γ[k,k,i,i]-γ[k,i,i,k]-γ[i,k,i,k]-γ[i,i,k,k]
#                                            == c4[i,i,k,k]+c4[i,k,i,k]+c4[i,k,k,i]+c4[k,k,i,i]+c4[k,i,k,i]+c4[k,i,i,k])
#
#                     elseif i == k && j == l && i != j
#                         @constraint(model, β[i,j,j,i]+β[j,i,j,i]+β[j,j,i,i]+β[j,i,i,j]+β[i,j,i,j]+β[i,i,j,j]-
#                                            γ[i,j,j,i]-γ[j,i,j,i]-γ[j,j,i,i]-γ[j,i,i,j]-γ[i,j,i,j]-γ[i,i,j,j]
#                                            == c4[i,i,j,j]+c4[i,j,i,j]+c4[i,j,j,i]+c4[j,j,i,i]+c4[j,i,j,i]+c4[j,i,i,j])
#
#                     elseif i == l && j == k && i != j
#                         @constraint(model, β[i,j,j,i]+β[j,i,j,i]+β[j,j,i,i]+β[j,i,i,j]+β[i,j,i,j]+β[i,i,j,j]-
#                                            γ[i,j,j,i]-γ[j,i,j,i]-γ[j,j,i,i]-γ[j,i,i,j]-γ[i,j,i,j]-γ[i,i,j,j]
#                                            == c4[i,i,j,j]+c4[i,j,i,j]+c4[i,j,j,i]+c4[j,j,i,i]+c4[j,i,j,i]+c4[j,i,i,j])
#
#                     elseif i == j && j == k && l != i
#                         @constraint(model, β[l,i,i,i]+β[i,l,i,i]+β[i,i,l,i]+β[i,i,i,l]-
#                                            γ[l,i,i,i]-γ[i,l,i,i]-γ[i,i,l,i]-γ[i,i,i,l]
#                                            == c4[l,i,i,i]+c4[i,l,i,i]+c4[i,i,l,i]+c4[i,i,i,l])
#
#                     elseif i == j && j == l && k != i
#                         @constraint(model, β[k,i,i,i]+β[i,k,i,i]+β[i,i,k,i]+β[i,i,i,k]-
#                                            γ[k,i,i,i]-γ[i,k,i,i]-γ[i,i,k,i]-γ[i,i,i,k]
#                                            == c4[k,i,i,i]+c4[i,k,i,i]+c4[i,i,k,i]+c4[i,i,i,k])
#
#                     elseif i == k && k == l && j != i
#                         @constraint(model, β[j,i,i,i]+β[i,j,i,i]+β[i,i,j,i]+β[i,i,i,j]-
#                                            γ[j,i,i,i]-γ[i,j,i,i]-γ[i,i,j,i]-γ[i,i,i,j]
#                                            == c4[j,i,i,i]+c4[i,j,i,i]+c4[i,i,j,i]+c4[i,i,i,j])
#
#                     elseif j == k && k == l && i != j
#                         @constraint(model, β[i,j,j,j]+β[j,i,j,j]+β[j,j,i,j]+β[j,j,j,i]-
#                                            γ[i,j,j,j]-γ[j,i,j,j]-γ[j,j,i,j]-γ[j,j,j,i]
#                                            == c4[i,j,j,j]+c4[j,i,j,j]+c4[j,j,i,j]+c4[j,j,j,i])
#
#                     elseif i == j && j == k && k == l
#                         @constraint(model, β[i,i,i,i]-γ[i,i,i,i] == c4[i,i,i,i])
#                     end
#                 end
#             end
#         end
#     end
#     ######################################################################################
#
#     ##########################    Inequality constraints for convexity    #############################
#
#     #old
#
#     # @constraint(model, [i=1:n, j=1:n], 6*β_abs[j,j,j,i] + 2*sum(β_abs[j,j,l,i]+β_abs[j,l,j,i]+β_abs[l,j,j,i] for l=1:n if l != j) +
#     #                                     sum(P_abs[j,k,i] + P_abs[k,j,i] + 2*(β_abs[j,j,k,i]+β_abs[j,k,j,i]+β_abs[k,j,j,i]) +
#     #                                         2*(β_abs[k,k,j,i]+β_abs[k,j,k,i]+β_abs[j,k,k,i]) +
#     #                                         sum(β_abs[j,k,l,i]+β_abs[j,l,k,i]+β_abs[k,j,l,i]+β_abs[k,l,j,i]+
#     #                                             β_abs[l,j,k,i]+β_abs[l,k,j,i] for l=1:n if l != j && l != k)
#     #                                             for k=1:n if k != j)
#     #                                     <= 2*P[j,j,i])
#     # #
#     # @constraint(model, [i=1:n, j=1:n], 6*γ_abs[j,j,j,i] + 2*sum(γ_abs[j,j,l,i]+γ_abs[j,l,j,i]+γ_abs[l,j,j,i] for l=1:n if l != j) +
#     #                                          sum(Q_abs[j,k,i] + Q_abs[k,j,i] + 2*(γ_abs[j,j,k,i]+γ_abs[j,k,j,i]+γ_abs[k,j,j,i]) +
#     #                                              2*(γ_abs[k,k,j,i]+γ_abs[k,j,k,i]+γ_abs[j,k,k,i]) +
#     #                                              sum(γ_abs[j,k,l,i]+γ_abs[j,l,k,i]+γ_abs[k,j,l,i]+γ_abs[k,l,j,i]+
#     #                                                  γ_abs[l,j,k,i]+γ_abs[l,k,j,i] for l=1:n if l != j && l != k)
#     #                                             for k=1:n if k != j)
#     #                                          <= 2*Q[j,j,i])
#
#
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], β[j,k,l,i] <= β_abs[j,k,l,i])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], -β[j,k,l,i] <= β_abs[j,k,l,i])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], γ[j,k,l,i] <= γ_abs[j,k,l,i])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], -γ[j,k,l,i] <= γ_abs[j,k,l,i])
#     @constraint(model, [i=1:n], P[:,:,i] .<= P_abs[:,:,i])
#     @constraint(model, [i=1:n], -P[:,:,i] .<= P_abs[:,:,i])
#     @constraint(model, [i=1:n], Q[:,:,i] .<= Q_abs[:,:,i])
#     @constraint(model, [i=1:n], -Q[:,:,i] .<= Q_abs[:,:,i])
#
#
#     @variable(model, v1[1:n,1:n,1:n])
#     @variable(model, v2[1:n,1:n,1:n])
#     @variable(model, v3[1:n,1:n,1:n])
#     @variable(model, v4[1:n,1:n,1:n,1:n])
#     @variable(model, v5[1:n,1:n,1:n])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n], P[j,k,i]+P[k,j,i] <= v1[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -P[j,k,i]-P[k,j,i] <= v1[i,j,k])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n], β[j,j,k,i]+β[j,k,j,i]+β[k,j,j,i] <= v2[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -β[j,j,k,i]-β[j,k,j,i]-β[k,j,j,i] <= v2[i,j,k])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n], β[k,k,j,i]+β[k,j,k,i]+β[j,k,k,i] <= v3[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -β[k,k,j,i]-β[k,j,k,i]-β[j,k,k,i] <= v3[i,j,k])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], β[j,k,l,i]+β[j,l,k,i]+β[k,j,l,i]+β[k,l,j,i]+
#                                                      β[l,j,k,i]+β[l,k,j,i] <= v4[i,j,k,l])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], -β[j,k,l,i]-β[j,l,k,i]-β[k,j,l,i]-β[k,l,j,i]-
#                                                       β[l,j,k,i]-β[l,k,j,i] <= v4[i,j,k,l])
#
#     @constraint(model, [i=1:n, j=1:n, l=1:n], β[j,j,l,i]+β[j,l,j,i]+β[l,j,j,i] <= v5[i,j,l])
#     @constraint(model, [i=1:n, j=1:n, l=1:n], -β[j,j,l,i]-β[j,l,j,i]-β[l,j,j,i] <= v5[i,j,l])
#
#     @constraint(model, [i=1:n, j=1:n], 6*β_abs[j,j,j,i] + 2*sum(v5[i,j,l] for l=1:n if l != j) +
#                                        sum(v1[i,j,k] + 2*v2[i,j,k] + 2*v3[i,j,k]+
#                                            sum(v4[i,j,k,l] for l=1:n if l != j && l != k)
#                                            for k=1:n if k != j)
#                                             <= 2*P[j,j,i])
#
#     @variable(model, s1[1:n,1:n,1:n])
#     @variable(model, s2[1:n,1:n,1:n])
#     @variable(model, s3[1:n,1:n,1:n])
#     @variable(model, s4[1:n,1:n,1:n,1:n])
#     @variable(model, s5[1:n,1:n,1:n])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n], Q[j,k,i]+Q[k,j,i] <= s1[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -Q[j,k,i]-Q[k,j,i] <= s1[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], γ[j,j,k,i]+γ[j,k,j,i]+γ[k,j,j,i] <= s2[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -γ[j,j,k,i]-γ[j,k,j,i]-γ[k,j,j,i] <= s2[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], γ[k,k,j,i]+γ[k,j,k,i]+γ[j,k,k,i] <= s3[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -γ[k,k,j,i]-γ[k,j,k,i]-γ[j,k,k,i] <= s3[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], γ[j,k,l,i]+γ[j,l,k,i]+γ[k,j,l,i]+γ[k,l,j,i]+
#                                                      γ[l,j,k,i]+γ[l,k,j,i] <= s4[i,j,k,l])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], -γ[j,k,l,i]-γ[j,l,k,i]-γ[k,j,l,i]-γ[k,l,j,i]-
#                                                       γ[l,j,k,i]-γ[l,k,j,i] <= s4[i,j,k,l])
#     @constraint(model, [i=1:n, j=1:n, l=1:n], γ[j,j,l,i]+γ[j,l,j,i]+γ[l,j,j,i] <= s5[i,j,l])
#     @constraint(model, [i=1:n, j=1:n, l=1:n], -γ[j,j,l,i]-γ[j,l,j,i]-γ[l,j,j,i] <= s5[i,j,l])
#
#     @constraint(model, [i=1:n, j=1:n], 6*γ_abs[j,j,j,i] + 2*sum(s5[i,j,l] for l=1:n if l != j) +
#                                         sum(s1[i,j,k] + 2*s2[i,j,k] + 2*s3[i,j,k]+
#                                            sum(s4[i,j,k,l] for l=1:n if l != j && l != k)
#                                            for k=1:n if k != j)
#                                            <= 2*Q[j,j,i])
#
#      ################################################################################################
#
#     @objective(model, Min, 0)
#
#     optimize!(model)
#     if termination_status(model) == MOI.OPTIMAL
#         β_list = [JuMP.value.(β)[:,:,:,i] for i=1:n]
#         P_list = [JuMP.value.(P)[:,:,i] for i=1:n]
#         r_list = [JuMP.value.(r)[:,i] for i=1:n]
#         w_list = JuMP.value.(w)
#         γ_list = [JuMP.value.(γ)[:,:,:,i] for i=1:n]
#         Q_list = [JuMP.value.(Q)[:,:,i] for i=1:n]
#         f_list = [JuMP.value.(f)[:,i] for i=1:n]
#         g_list = JuMP.value.(g)
#         Z = [β_list, P_list, r_list, w_list, γ_list, Q_list, f_list, g_list]
#     else
#         println("Error")
#         println(termination_status(model))
#         Z = []
#     end
#     return Z
# end
#
# function get_poly4_best_slc(c4,c3,c2,c1,c0,x,X)
#     n = size(c1,1)
#     model = Model(Gurobi.Optimizer)
#     set_optimizer_attribute(model, "OutputFlag", 0)
#     @variable(model, β[1:n,1:n,1:n,1:n])
#     @variable(model, P[1:n,1:n,1:n])
#     @variable(model, r[1:n,1:n])
#     @variable(model, w[1:n])
#     @variable(model, γ[1:n,1:n,1:n,1:n])
#     @variable(model, Q[1:n,1:n,1:n])
#     @variable(model, f[1:n,1:n])
#     @variable(model, g[1:n])
#     # auxiliary variables to model absolute values
#     @variable(model, β_abs[1:n,1:n,1:n,1:n])
#     @variable(model, γ_abs[1:n,1:n,1:n,1:n])
#     # auxiliary epigraphical variables
#     @variable(model, τ[1:n])
#     @variable(model, η[1:n])
#
#     ###################   Equality constraints to match coefficients  ###################
#     # zero degree terms
#     @constraint(model, sum(g[i] for i=1:n) == c0)
#     # first degree terms
#     @constraint(model, [i=1:n], w[i]-g[i]+sum(f[i,m] for m=1:n) == c1[i])
#     # second degree terms
#     for i in 1:n
#         for j in i:n
#             if i != j
#                 @constraint(model, r[j,i]+r[i,j]-f[j,i]-f[i,j]+sum(Q[i,j,m]+Q[j,i,m] for m=1:n) == c2[i,j]+c2[j,i])
#             else
#                 @constraint(model, r[i,i]-f[i,i]+sum(Q[i,i,m] for m=1:n) == c2[i,i])
#             end
#         end
#     end
#     # third degree terms
#     for i in 1:n
#         for j in i:n
#             for k in j:n
#                 if i != j && i != k && j != k
#                     @constraint(model, P[j,k,i]+P[k,j,i]+P[i,k,j]+P[k,i,j]+P[i,j,k]+P[j,i,k]
#                                        -Q[j,k,i]-Q[k,j,i]-Q[i,k,j]-Q[k,i,j]-Q[i,j,k]-Q[j,i,k]+
#                                         sum(γ[i,j,k,m]+γ[i,k,j,m]+γ[j,i,k,m]+γ[j,k,i,m]+
#                                             γ[k,i,j,m]+γ[k,j,i,m] for m=1:n)
#                                         == c3[i,j,k]+c3[i,k,j]+c3[j,i,k]+c3[j,k,i]+c3[k,i,j]+c3[k,j,i])
#                 elseif i == j && k != i
#                     @constraint(model, P[k,i,i]+P[i,k,i]+P[i,i,k]-Q[k,i,i]-Q[i,k,i]-Q[i,i,k]+
#                                        sum(γ[k,i,i,m]+γ[i,k,i,m]+γ[i,i,k,m] for m=1:n)
#                                         == c3[k,i,i]+c3[i,k,i]+c3[i,i,k])
#                 elseif i == k && j != i
#                     @constraint(model, P[j,i,i]+P[i,j,i]+P[i,i,j]-Q[j,i,i]-Q[i,j,i]-Q[i,i,j]+
#                                        sum(γ[j,i,i,m]+γ[i,j,i,m]+γ[i,i,j,m] for m=1:n)
#                                         == c3[j,i,i]+c3[i,j,i]+c3[i,i,j])
#                 elseif j == k && i != j
#                     @constraint(model,  P[i,j,j]+P[j,i,j]+P[j,j,i]-Q[i,j,j]-Q[j,i,j]-Q[j,j,i]+
#                                         sum(γ[i,j,j,m]+γ[j,i,j,m]+γ[j,j,i,m] for m=1:n)
#                                         == c3[i,j,j]+c3[j,i,j]+c3[j,j,i])
#                 elseif i == j && j == k
#                    @constraint(model, P[i,i,i]-Q[i,i,i]+sum(γ[i,i,i,m] for m=1:n) == c3[i,i,i])
#                end
#            end
#        end
#    end
#    # fourth degree terms
#    for i in 1:n
#        for j in i:n
#            for k in j:n
#                for l in k:n
#                     if i != j && i != k && i != l && j != k && j != l && k != l
#                         @constraint(model, β[j,k,l,i]+β[j,l,k,i]+β[k,j,l,i]+β[k,l,j,i]+β[l,j,k,i]+β[l,k,j,i]+
#                                            β[i,k,l,j]+β[i,l,k,j]+β[k,i,l,j]+β[k,l,i,j]+β[l,i,k,j]+β[l,k,i,j]+
#                                            β[i,j,l,k]+β[i,l,j,k]+β[j,i,l,k]+β[j,l,i,k]+β[l,i,j,k]+β[l,j,i,k]+
#                                            β[i,j,k,l]+β[i,k,j,l]+β[j,i,k,l]+β[j,k,i,l]+β[k,i,j,l]+β[k,j,i,l]-
#                                            γ[j,k,l,i]-γ[j,l,k,i]-γ[k,j,l,i]-γ[k,l,j,i]-γ[l,j,k,i]-γ[l,k,j,i]-
#                                            γ[i,k,l,j]-γ[i,l,k,j]-γ[k,i,l,j]-γ[k,l,i,j]-γ[l,i,k,j]-γ[l,k,i,j]-
#                                            γ[i,j,l,k]-γ[i,l,j,k]-γ[j,i,l,k]-γ[j,l,i,k]-γ[l,i,j,k]-γ[l,j,i,k]-
#                                            γ[i,j,k,l]-γ[i,k,j,l]-γ[j,i,k,l]-γ[j,k,i,l]-γ[k,i,j,l]-γ[k,j,i,l]
#                                            == c4[i,j,k,l]+c4[i,j,l,k]+c4[i,k,j,l]+c4[i,k,l,j]+c4[i,l,j,k]+
#                                            c4[i,l,k,j]+c4[j,i,k,l]+c4[j,i,l,k]+c4[j,k,i,l]+c4[j,k,l,i]+
#                                            c4[j,l,i,k]+c4[j,l,k,i]+c4[k,i,j,l]+c4[k,i,l,j]+c4[k,j,i,l]+
#                                            c4[k,j,l,i]+c4[k,l,i,j]+c4[k,l,j,i]+c4[l,i,j,k]+c4[l,i,k,j]+
#                                            c4[l,j,i,k]+c4[l,j,k,i]+c4[l,k,i,j]+c4[l,k,j,i])
#
#                     elseif i == j && i != k && i != l && k != l
#                         @constraint(model, β[i,k,l,i]+β[i,l,k,i]+β[k,i,l,i]+β[k,l,i,i]+β[l,i,k,i]+β[l,k,i,i]+
#                                            β[l,i,i,k]+β[i,l,i,k]+β[i,i,l,k]+β[k,i,i,l]+β[i,k,i,l]+β[i,i,k,l]-
#                                            γ[i,k,l,i]-γ[i,l,k,i]-γ[k,i,l,i]-γ[k,l,i,i]-γ[l,i,k,i]-γ[l,k,i,i]-
#                                            γ[l,i,i,k]-γ[i,l,i,k]-γ[i,i,l,k]-γ[k,i,i,l]-γ[i,k,i,l]-γ[i,i,k,l]
#                                            == c4[i,i,k,l]+c4[i,i,l,k]+c4[i,k,i,l]+c4[i,k,l,i]+c4[i,l,i,k]+
#                                            c4[i,l,k,i]+c4[k,i,i,l]+c4[k,i,l,i]+c4[k,l,i,i]+c4[l,i,i,k]+
#                                            c4[l,i,k,i]+c4[l,k,i,i])
#
#                     elseif i == k && i != j && i != l && j != l
#                         @constraint(model, β[i,j,l,i]+β[i,l,j,i]+β[j,i,l,i]+β[j,l,i,i]+β[l,i,j,i]+β[l,j,i,i]+
#                                            β[i,i,l,j]+β[i,l,i,j]+β[l,i,i,j]+β[i,i,j,l]+β[i,j,i,l]+β[j,i,i,l]-
#                                            γ[i,j,l,i]-γ[i,l,j,i]-γ[j,i,l,i]-γ[j,l,i,i]-γ[l,i,j,i]-γ[l,j,i,i]-
#                                            γ[i,i,l,j]-γ[i,l,i,j]-γ[l,i,i,j]-γ[i,i,j,l]-γ[i,j,i,l]-γ[j,i,i,l]
#                                            == c4[i,i,j,l]+c4[i,i,l,j]+c4[i,j,i,l]+c4[i,j,l,i]+c4[i,l,i,j]+
#                                            c4[i,l,j,i]+c4[j,i,i,l]+c4[j,i,l,i]+c4[j,l,i,i]+c4[l,i,i,j]+
#                                            c4[l,i,j,i]+c4[l,j,i,i])
#
#                     elseif i == l && i != j && i != k && j != k
#                         @constraint(model, β[i,j,k,i]+β[i,k,j,i]+β[j,i,k,i]+β[j,k,i,i]+β[k,i,j,i]+β[k,j,i,i]+
#                                            β[i,i,k,j]+β[i,k,i,j]+β[k,i,i,j]+β[i,i,j,k]+β[i,j,i,k]+β[j,i,i,k]-
#                                            γ[i,j,k,i]-γ[i,k,j,i]-γ[j,i,k,i]-γ[j,k,i,i]-γ[k,i,j,i]-γ[k,j,i,i]-
#                                            γ[i,i,k,j]-γ[i,k,i,j]-γ[k,i,i,j]-γ[i,i,j,k]-γ[i,j,i,k]-γ[j,i,i,k]
#                                            == c4[i,i,j,k]+c4[i,i,k,j]+c4[i,j,i,k]+c4[i,j,k,i]+c4[i,k,i,j]+
#                                            c4[i,k,j,i]+c4[j,i,i,k]+c4[j,i,k,i]+c4[j,k,i,i]+c4[k,i,i,j]+
#                                            c4[k,i,j,i]+c4[k,j,i,i])
#
#                     elseif j == l && i != j && k != j && i != k
#                         @constraint(model, β[j,j,k,i]+β[j,k,j,i]+β[k,j,j,i]+β[i,j,j,k]+β[j,i,j,k]+β[j,j,i,k]+
#                                            β[i,j,k,j]+β[i,k,j,j]+β[j,i,k,j]+β[j,k,i,j]+β[k,i,j,j]+β[k,j,i,j]-
#                                            γ[j,j,k,i]-γ[j,k,j,i]-γ[k,j,j,i]-γ[i,j,j,k]-γ[j,i,j,k]-γ[j,j,i,k]-
#                                            γ[i,j,k,j]-γ[i,k,j,j]-γ[j,i,k,j]-γ[j,k,i,j]-γ[k,i,j,j]-γ[k,j,i,j]
#                                            == c4[j,j,i,k]+c4[j,j,k,i]+c4[j,k,j,i]+c4[j,k,i,j]+c4[j,i,j,k]+
#                                            c4[j,i,k,j]+c4[i,j,j,k]+c4[i,j,k,j]+c4[i,k,j,j]+c4[k,j,j,i]+
#                                            c4[k,j,i,j]+c4[k,i,j,j])
#
#                     elseif j == k && i != j && l != j && l != i
#                         @constraint(model, β[j,j,l,i]+β[j,l,j,i]+β[l,j,j,i]+β[i,j,j,l]+β[j,i,j,l]+β[j,j,i,l]+
#                                            β[i,j,l,j]+β[i,l,j,j]+β[j,i,l,j]+β[j,l,i,j]+β[l,i,j,j]+β[l,j,i,j]-
#                                            γ[j,j,l,i]-γ[j,l,j,i]-γ[l,j,j,i]-γ[i,j,j,l]-γ[j,i,j,l]-γ[j,j,i,l]-
#                                            γ[i,j,l,j]-γ[i,l,j,j]-γ[j,i,l,j]-γ[j,l,i,j]-γ[l,i,j,j]-γ[l,j,i,j]
#                                            == c4[j,j,i,l]+c4[j,j,l,i]+c4[j,i,j,l]+c4[j,i,l,j]+c4[j,l,j,i]+
#                                            c4[j,l,i,j]+c4[i,j,j,l]+c4[i,j,l,j]+c4[i,l,j,j]+c4[l,j,j,i]+
#                                            c4[l,j,i,j]+c4[l,i,j,j])
#
#                     elseif k == l && i != k && j != k && i != j
#                         @constraint(model, β[j,k,k,i]+β[k,j,k,i]+β[k,k,j,i]+β[i,k,k,j]+β[k,i,k,j]+β[k,k,i,j]+
#                                            β[i,j,k,k]+β[i,k,j,k]+β[j,i,k,k]+β[j,k,i,k]+β[k,i,j,k]+β[k,j,i,k]-
#                                            γ[j,k,k,i]-γ[k,j,k,i]-γ[k,k,j,i]-γ[i,k,k,j]-γ[k,i,k,j]-γ[k,k,i,j]-
#                                            γ[i,j,k,k]-γ[i,k,j,k]-γ[j,i,k,k]-γ[j,k,i,k]-γ[k,i,j,k]-γ[k,j,i,k]
#                                            == c4[k,k,i,j]+c4[k,k,j,i]+c4[k,i,k,j]+c4[k,i,j,k]+c4[k,j,k,i]+
#                                            c4[k,j,i,k]+c4[i,k,k,j]+c4[i,k,j,k]+c4[i,j,k,k]+c4[j,k,k,i]+
#                                            c4[j,k,i,k]+c4[j,i,k,k])
#
#                     elseif i == j && k == l && i != k
#                         @constraint(model, β[i,k,k,i]+β[k,i,k,i]+β[k,k,i,i]+β[k,i,i,k]+β[i,k,i,k]+β[i,i,k,k]-
#                                            γ[i,k,k,i]-γ[k,i,k,i]-γ[k,k,i,i]-γ[k,i,i,k]-γ[i,k,i,k]-γ[i,i,k,k]
#                                            == c4[i,i,k,k]+c4[i,k,i,k]+c4[i,k,k,i]+c4[k,k,i,i]+c4[k,i,k,i]+c4[k,i,i,k])
#
#                     elseif i == k && j == l && i != j
#                         @constraint(model, β[i,j,j,i]+β[j,i,j,i]+β[j,j,i,i]+β[j,i,i,j]+β[i,j,i,j]+β[i,i,j,j]-
#                                            γ[i,j,j,i]-γ[j,i,j,i]-γ[j,j,i,i]-γ[j,i,i,j]-γ[i,j,i,j]-γ[i,i,j,j]
#                                            == c4[i,i,j,j]+c4[i,j,i,j]+c4[i,j,j,i]+c4[j,j,i,i]+c4[j,i,j,i]+c4[j,i,i,j])
#
#                     elseif i == l && j == k && i != j
#                         @constraint(model, β[i,j,j,i]+β[j,i,j,i]+β[j,j,i,i]+β[j,i,i,j]+β[i,j,i,j]+β[i,i,j,j]-
#                                            γ[i,j,j,i]-γ[j,i,j,i]-γ[j,j,i,i]-γ[j,i,i,j]-γ[i,j,i,j]-γ[i,i,j,j]
#                                            == c4[i,i,j,j]+c4[i,j,i,j]+c4[i,j,j,i]+c4[j,j,i,i]+c4[j,i,j,i]+c4[j,i,i,j])
#
#                     elseif i == j && j == k && l != i
#                         @constraint(model, β[i,i,l,i]+β[i,l,i,i]+β[l,i,i,i]+β[i,i,i,l]-
#                                            γ[i,i,l,i]-γ[i,l,i,i]-γ[l,i,i,i]-γ[i,i,i,l]
#                                            == c4[l,i,i,i]+c4[i,l,i,i]+c4[i,i,l,i]+c4[i,i,i,l])
#                     elseif i == j && j == l && k != i
#                         @constraint(model, β[i,i,k,i]+β[i,k,i,i]+β[k,i,i,i]+β[i,i,i,k]-
#                                            γ[i,i,k,i]-γ[i,k,i,i]-γ[k,i,i,i]-γ[i,i,i,k]
#                                            == c4[k,i,i,i]+c4[i,k,i,i]+c4[i,i,k,i]+c4[i,i,i,k])
#                     elseif i == k && k == l && j != i
#                         @constraint(model, β[i,i,j,i]+β[i,j,i,i]+β[j,i,i,i]+β[i,i,i,j]-
#                                            γ[i,i,j,i]-γ[i,j,i,i]-γ[j,i,i,i]-γ[i,i,i,j]
#                                            == c4[j,i,i,i]+c4[i,j,i,i]+c4[i,i,j,i]+c4[i,i,i,j])
#                     elseif j == k && k == l && i != j
#                         @constraint(model, β[j,j,i,j]+β[j,i,j,j]+β[i,j,j,j]+β[j,j,j,i]-
#                                            γ[j,j,i,j]-γ[j,i,j,j]-γ[i,j,j,j]-γ[j,j,j,i]
#                                            == c4[i,j,j,j]+c4[j,i,j,j]+c4[j,j,i,j]+c4[j,j,j,i])
#
#                     elseif i == j && j == k && k == l
#                         @constraint(model, β[i,i,i,i]-γ[i,i,i,i] == c4[i,i,i,i])
#                     end
#                 end
#             end
#         end
#     end
#     ######################################################################################
#
#     ##########################    Inequality constraints for convexity    #############################
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], β[j,k,l,i] <= β_abs[j,k,l,i])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], -β[j,k,l,i] <= β_abs[j,k,l,i])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], γ[j,k,l,i] <= γ_abs[j,k,l,i])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], -γ[j,k,l,i] <= γ_abs[j,k,l,i])
#
#     @variable(model, v1[1:n,1:n,1:n])
#     @variable(model, v2[1:n,1:n,1:n])
#     @variable(model, v3[1:n,1:n,1:n])
#     @variable(model, v4[1:n,1:n,1:n,1:n])
#     @variable(model, v5[1:n,1:n,1:n])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n], P[j,k,i]+P[k,j,i] <= v1[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -P[j,k,i]-P[k,j,i] <= v1[i,j,k])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n], β[j,j,k,i]+β[j,k,j,i]+β[k,j,j,i] <= v2[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -β[j,j,k,i]-β[j,k,j,i]-β[k,j,j,i] <= v2[i,j,k])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n], β[k,k,j,i]+β[k,j,k,i]+β[j,k,k,i] <= v3[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -β[k,k,j,i]-β[k,j,k,i]-β[j,k,k,i] <= v3[i,j,k])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], β[j,k,l,i]+β[j,l,k,i]+β[k,j,l,i]+β[k,l,j,i]+
#                                                      β[l,j,k,i]+β[l,k,j,i] <= v4[i,j,k,l])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], -β[j,k,l,i]-β[j,l,k,i]-β[k,j,l,i]-β[k,l,j,i]-
#                                                       β[l,j,k,i]-β[l,k,j,i] <= v4[i,j,k,l])
#
#     @constraint(model, [i=1:n, j=1:n, l=1:n], β[j,j,l,i]+β[j,l,j,i]+β[l,j,j,i] <= v5[i,j,l])
#     @constraint(model, [i=1:n, j=1:n, l=1:n], -β[j,j,l,i]-β[j,l,j,i]-β[l,j,j,i] <= v5[i,j,l])
#
#     @constraint(model, [i=1:n, j=1:n], 6*β_abs[j,j,j,i] + 2*sum(v5[i,j,l] for l=1:n if l != j) +
#                                        sum(v1[i,j,k] + 2*v2[i,j,k] + 2*v3[i,j,k]+
#                                            sum(v4[i,j,k,l] for l=1:n if l != j && l != k)
#                                            for k=1:n if k != j)
#                                             <= 2*P[j,j,i])
#
#     @variable(model, s1[1:n,1:n,1:n])
#     @variable(model, s2[1:n,1:n,1:n])
#     @variable(model, s3[1:n,1:n,1:n])
#     @variable(model, s4[1:n,1:n,1:n,1:n])
#     @variable(model, s5[1:n,1:n,1:n])
#
#     @constraint(model, [i=1:n, j=1:n, k=1:n], Q[j,k,i]+Q[k,j,i] <= s1[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -Q[j,k,i]-Q[k,j,i] <= s1[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], γ[j,j,k,i]+γ[j,k,j,i]+γ[k,j,j,i] <= s2[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -γ[j,j,k,i]-γ[j,k,j,i]-γ[k,j,j,i] <= s2[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], γ[k,k,j,i]+γ[k,j,k,i]+γ[j,k,k,i] <= s3[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n], -γ[k,k,j,i]-γ[k,j,k,i]-γ[j,k,k,i] <= s3[i,j,k])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], γ[j,k,l,i]+γ[j,l,k,i]+γ[k,j,l,i]+γ[k,l,j,i]+
#                                                      γ[l,j,k,i]+γ[l,k,j,i] <= s4[i,j,k,l])
#     @constraint(model, [i=1:n, j=1:n, k=1:n, l=1:n], -γ[j,k,l,i]-γ[j,l,k,i]-γ[k,j,l,i]-γ[k,l,j,i]-
#                                                       γ[l,j,k,i]-γ[l,k,j,i] <= s4[i,j,k,l])
#     @constraint(model, [i=1:n, j=1:n, l=1:n], γ[j,j,l,i]+γ[j,l,j,i]+γ[l,j,j,i] <= s5[i,j,l])
#     @constraint(model, [i=1:n, j=1:n, l=1:n], -γ[j,j,l,i]-γ[j,l,j,i]-γ[l,j,j,i] <= s5[i,j,l])
#
#     @constraint(model, [i=1:n, j=1:n], 6*γ_abs[j,j,j,i] + 2*sum(s5[i,j,l] for l=1:n if l != j) +
#                                         sum(s1[i,j,k] + 2*s2[i,j,k] + 2*s3[i,j,k]+
#                                            sum(s4[i,j,k,l] for l=1:n if l != j && l != k)
#                                            for k=1:n if k != j)
#                                            <= 2*Q[j,j,i])
#
#     ################################################################################################
#
#     for i in 1:n
#         # @constraint(model, sum(β[j,k,l,i]*X[j,i]*X[k,i]*X[l,i] for j=1:n, k=1:n, l=1:n) +
#         #                                 x[i]*X[:,i]'*P[:,:,i]*X[:,i] + (x[i]^2)*X[:,i]'*r[:,i] +
#         #                                 (x[i]^3)*w[i] >= (x[i]^2)*τ[i])
#         #
#         # @constraint(model, sum(γ[j,k,l,i]*(x[j]-X[j,i])*(x[k]-X[k,i])*(x[l]-X[l,i]) for j=1:n, k=1:n, l=1:n) +
#         #                                 (1-x[i])*(x.-X[:,i])'*Q[:,:,i]*(x.-X[:,i]) + ((1-x[i])^2)*(x.-X[:,i])'*f[:,i] +
#         #                                 ((1-x[i])^3)*g[i] >= ((1-x[i])^2)*η[i])
#         if x[i] <= 0.01
#             @constraint(model, τ[i] == 0)
#         else
#             @constraint(model, sum(β[j,k,l,i]*X[j,i]*X[k,i]*X[l,i] for j=1:n, k=1:n, l=1:n) +
#                                             x[i]*X[:,i]'*P[:,:,i]*X[:,i] + (x[i]^2)*X[:,i]'*r[:,i] +
#                                             (x[i]^3)*w[i] >= (x[i]^2)*τ[i])
#         end
#         if x[i] >= 0.99
#             @constraint(model, η[i] == 0)
#         else
#             @constraint(model, sum(γ[j,k,l,i]*(x[j]-X[j,i])*(x[k]-X[k,i])*(x[l]-X[l,i]) for j=1:n, k=1:n, l=1:n) +
#                                             (1-x[i])*(x.-X[:,i])'*Q[:,:,i]*(x.-X[:,i]) + ((1-x[i])^2)*(x.-X[:,i])'*f[:,i] +
#                                             ((1-x[i])^3)*g[i] >= ((1-x[i])^2)*η[i])
#         end
#     end
#
#     ########################    add bound constraints  #################################
#     @variable(model, β_reg[1:n])
#     @variable(model, P_reg[1:n])
#     @variable(model, r_reg[1:n])
#     @variable(model, w_reg)
#     @variable(model, γ_reg[1:n])
#     @variable(model, Q_reg[1:n])
#     @variable(model, f_reg[1:n])
#     @variable(model, g_reg)
#
#     @constraint(model, [i=1:n], vec(β[:,:,:,i])'*vec(β[:,:,:,i]) <= β_reg[i])
#     @constraint(model, [i=1:n], vec(P[:,:,i])'*vec(P[:,:,i]) <= P_reg[i])
#     @constraint(model, [i=1:n], r[:,i]'*r[:,i] <= r_reg[i])
#     @constraint(model, w'*w <= w_reg)
#     @constraint(model, [i=1:n], vec(γ[:,:,:,i])'*vec(γ[:,:,:,i]) <= γ_reg[i])
#     @constraint(model, [i=1:n], vec(Q[:,:,i])'*vec(Q[:,:,i]) <= Q_reg[i])
#     @constraint(model, [i=1:n], f[:,i]'*f[:,i] <= f_reg[i])
#     @constraint(model, g'*g <= g_reg)
#
#     # @constraint(model, [i=1:n], [β_reg[i];vec(β[:,:,:,i])] in SecondOrderCone())
#     # @constraint(model, [i=1:n], [P_reg[i];vec(P[:,:,i])] in SecondOrderCone())
#     # @constraint(model, [i=1:n], [r_reg[i];r[:,i]] in SecondOrderCone())
#     # @constraint(model, [w_reg;w] in SecondOrderCone())
#     # @constraint(model, [i=1:n], [γ_reg[i];vec(γ[:,:,:,i])] in SecondOrderCone())
#     # @constraint(model, [i=1:n], [Q_reg[i];vec(Q[:,:,i])] in SecondOrderCone())
#     # @constraint(model, [i=1:n], [f_reg[i];f[:,i]] in SecondOrderCone())
#     # @constraint(model, [g_reg;g] in SecondOrderCone())
#
#     # M = 10
#     #
#     # @constraint(model, [i=1:n], vec(β[:,:,:,i])'*vec(β[:,:,:,i]) <= M)
#     # @constraint(model, [i=1:n], vec(P[:,:,i])'*vec(P[:,:,i]) <= M)
#     # @constraint(model, [i=1:n], r[:,i]'*r[:,i] <= M)
#     # @constraint(model, w'*w <= w_reg)
#     # @constraint(model, [i=1:n], vec(γ[:,:,:,i])'*vec(γ[:,:,:,i]) <= M)
#     # @constraint(model, [i=1:n], vec(Q[:,:,i])'*vec(Q[:,:,i]) <= M)
#     # @constraint(model, [i=1:n], f[:,i]'*f[:,i] <= M)
#     # @constraint(model, g'*g <= M)
#
#     #####################################################################################
#
#     # here
#     @objective(model, Max, sum(τ[i]+η[i] for i=1:n) - 0.001*(
#                            sum(β_reg[i]+P_reg[i]+r_reg[i]+γ_reg[i]+Q_reg[i]+f_reg[i]
#                            for i=1:n)+w_reg+g_reg)
#                            )
#
#     optimize!(model)
#     if termination_status(model) == MOI.OPTIMAL
#         β_list = [JuMP.value.(β)[:,:,:,i] for i=1:n]
#         P_list = [JuMP.value.(P)[:,:,i] for i=1:n]
#         r_list = [JuMP.value.(r)[:,i] for i=1:n]
#         w_list = JuMP.value.(w)
#         γ_list = [JuMP.value.(γ)[:,:,:,i] for i=1:n]
#         Q_list = [JuMP.value.(Q)[:,:,i] for i=1:n]
#         f_list = [JuMP.value.(f)[:,i] for i=1:n]
#         g_list = JuMP.value.(g)
#         Z = [β_list, P_list, r_list, w_list, γ_list, Q_list, f_list, g_list]
#         return true, Z
#     else
#         println("Error")
#         println(termination_status(model))
#         return false, []
#     end
# end
#
# function solve_rpt_relaxation_X1_old(C,d,coeff_list,sdp_vecs)
#     n = size(C,2)
#
#     model = Model(Ipopt.Optimizer)
#     @variable(model, x[1:n]>=0)
#     @variable(model, X[1:n,1:n]>=0, Symmetric)
#     @variable(model, τ)
#     # @variable(model, τ2)
#     # @variable(model, τ[1:n])
#     # @variable(model, η[1:n])
#
#     # for j in 1:n
#     #     JuMP.set_start_value(x[j], 0.5)
#     # end
#
#     @constraint(model, C*x .<= d)
#     @constraint(model, [i in 1:n], C*X[:,i] .<= x[i]*d)
#     @constraint(model, d*x'*C' .+ C*x*d' .<= C*X*C' .+ d*d')
#
#     # @constraint(model, [i in 1:n], x[i] <= 0.99)
#     # @constraint(model, [i in 1:n], x[i] >= 0.01)
#
#     if length(sdp_vecs) > 0
#         for i in 1:length(sdp_vecs)
#             u = sdp_vecs[i]
#             @constraint(model, u'*(X.-x*x')*u >= 0)
#         end
#     end
#
#     for Z in coeff_list
#         β_list, P_list, r_list, w_list = Z[1], Z[2], Z[3], Z[4]
#         γ_list, Q_list, f_list, g_list = Z[5], Z[6], Z[7], Z[8]
#
#         # @NLconstraint(model, [i=1:n], sum(β_list[i][j,k,l]*X[j,i]*X[k,i]*X[l,i] for j=1:n, k=1:n, l=1:n) +
#         #                               x[i]*sum(X[k,i]*P_list[i][k,l]*X[l,i] for k=1:n, l=1:n) +
#         #                               (x[i]^2)*sum(r_list[i][k]*X[k,i] for k=1:n) + w_list[i]*(x[i]^3)
#         #                                 <= (x[i]^2)*τ[i])
#         #
#         # @NLconstraint(model, [i=1:n], sum(γ_list[i][j,k,l]*(x[j]-X[j,i])*(x[k]-X[k,i])*(x[l]-X[l,i]) for j=1:n, k=1:n, l=1:n) +
#         #                               (1-x[i])*sum((x[k]-X[k,i])*Q_list[i][k,l]*(x[l]-X[l,i]) for k=1:n, l=1:n) +
#         #                               ((1-x[i])^2)*sum(f_list[i][k]*(x[k]-X[k,i]) for k=1:n) + g_list[i]*((1-x[i])^3)
#         #                                 <= ((1-x[i])^2)*η[i])
#         #
#         @NLconstraint(model,
#                              sum( (1/(x[i]^2))*sum(β_list[i][j,k,l]*X[j,i]*X[k,i]*X[l,i] for j=1:n, k=1:n, l=1:n) +
#                                   (1/x[i])*sum(X[k,i]*P_list[i][k,l]*X[l,i] for k=1:n, l=1:n) +
#                                    sum(r_list[i][k]*X[k,i] for k=1:n) +
#                                    w_list[i]*x[i]
#                                       for i=1:n) +
#                               sum(
#                               (1/((1-x[i])^2))*sum(γ_list[i][j,k,l]*(x[j]-X[j,i])*(x[k]-X[k,i])*(x[l]-X[l,i])
#                                                           for j=1:n, k=1:n, l=1:n) +
#                                    (1/(1-x[i]))*sum((x[k]-X[k,i])*Q_list[i][k,l]*(x[l]-X[l,i]) for k=1:n, l=1:n) +
#                                     sum(f_list[i][k]*(x[k]-X[k,i]) for k=1:n) +
#                                     g_list[i]*(1-x[i])
#                                        for i=1:n)
#                                     <= τ)
#
#         # @NLconstraint(model, sum( (1/(x[i]^2))*sum(β_list[i][j,k,l]*X[j,i]*X[k,i]*X[l,i] for j=1:n, k=1:n, l=1:n) +
#         #                               (1/x[i])*sum(X[k,i]*P_list[i][k,l]*X[l,i] for k=1:n, l=1:n) +
#         #                                sum(r_list[i][k]*X[k,i] for k=1:n) +
#         #                                w_list[i]*x[i]
#         #                                   for i=1:n)
#         #                                       <= τ1)
#         #
#         # @NLconstraint(model, sum( (1/((1-x[i])^2))*sum(γ_list[i][j,k,l]*(x[j]-X[j,i])*(x[k]-X[k,i])*(x[l]-X[l,i])
#         #                                                     for j=1:n, k=1:n, l=1:n) +
#         #                              (1/(1-x[i]))*sum((x[k]-X[k,i])*Q_list[i][k,l]*(x[l]-X[l,i]) for k=1:n, l=1:n) +
#         #                               sum(f_list[i][k]*(x[k]-X[k,i]) for k=1:n) +
#         #                               g_list[i]*(1-x[i])
#         #                                  for i=1:n)
#         #                               <= τ2)
#
#
#     end
#     @objective(model, Min, τ)
#     # @objective(model, Min, sum(τ[i] + η[i] for i=1:n))
#     optimize!(model)
#     if termination_status(model) == MOI.OPTIMAL ||  termination_status(model) == MOI.SLOW_PROGRESS || termination_status(model) == MOI.LOCALLY_SOLVED
#         x_opt, X_opt = JuMP.value.(x), JuMP.value.(X)
#         for i=1:n
#             if x_opt[i] < 0.01
#                 x_opt[i] = 0.01
#             end
#             if x_opt[i] > 0.99
#                 x_opt[i] = 0.99
#             end
#         end
#         for i=1:n
#             for j=1:n
#                 if X_opt[i,j] < 0.01
#                     X_opt[i,j] = 0.01
#                 end
#                 if X_opt[i,j] > 0.99
#                     X_opt[i,j] = 0.99
#                 end
#             end
#         end
#         return true, x_opt, X_opt, objective_value(model)
#     else
#         return false, zeros(n), zeros(n,n), 1e6
#     end
# end
#
# function find_sdp_violating_vecs(x,X)
#     n = size(x,1)
#     D = X .- x*x'
#     λ, U = eigen(D)
#     violating_vecs = [U[:,i] for i=1:n if λ[i] < 0]
#     return violating_vecs
# end
#
# function add_approx_lmi(C,d,coeff_list,sdp_vec_list)
#     cond = true
#     while cond
#         res, x_opt, X_opt, lb = solve_rpt_relaxation_X1(C,d,coeff_list,sdp_vec_list)
#         violating_vecs = find_sdp_violating_vecs(x_opt,X_opt)
#         if length(violating_vecs) > 0
#             for i in 1:length(violating_vecs)
#                 push!(sdp_vec_list, violating_vecs[i])
#             end
#
#         end
#         cond = false
#     end
#     return sdp_vec_list
# end
#
#
# function rpt_bb_cut_planes_X1(c4,c3,c2,c1,c0,C_init,d_init,δ)
#     C, d = C_init, d_init
#     gen_hyper = 0
#     coeff_list_cur = []
#     sdp_vecs = []
#     # sdp_vecs = add_approx_lmi(C,d,coeff_list_cur,sdp_vecs)
#
#     Z_0 = get_poly4_random_slc(c4,c3,c2,c1,c0)
#     push!(coeff_list_cur, Z_0)
#
#     # Root Node
#     res_root, x_lb, X_lb, lb = solve_rpt_relaxation_X1(C,d,coeff_list_cur,sdp_vecs)
#     X = calculate_candidate_vectors(x_lb,X_lb)
#     ub = get_ub_X1(X,C,d,c4,c3,c2,c1,c0)
#     ### extra ####
#     # violating_vecs_curr = find_sdp_violating_vecs(x_lb,X_lb)
#     # append!(sdp_vecs, violating_vecs_curr)
#     ##################
#
#     res_root_1, Z_1 = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb,X_lb)
#     if res_root_1
#         push!(coeff_list_cur, Z_1)
#         # coeff_list_cur = [Z_1]
#     end
#
#     x_cur, X_cur = x_lb, X_lb
#
#     UB, LB, opt_sol, opt_val = ub, lb, x_lb, ub
#
#     nodes_list = []
#     ub_list, lb_list, hyp_list = [], [], []
#     push!(ub_list, ub)
#     push!(lb_list, lb)
#
#     t0_1 = time_ns()
#     total_time = 0.0
#
#     # while (abs(UB - LB)/abs(UB))*100 > δ #&& total_time < 60
#     while UB - LB > δ && total_time < 3600
#         # res, f_opt, l_opt = generate_hyperplane_eigen(x_cur,X_cur)
#         res, f_opt, l_opt = generate_hyperplane_new(x_cur,X_cur,coeff_list_cur[end])
#         # res, f_opt, l_opt = generate_hyperplane_new_2(x_cur,X_cur,coeff_list_cur)
#
#         f_r, l_r, f_l, l_l  = f_opt, l_opt, -f_opt, -l_opt
#         C_r, d_r, C_l, d_l = vcat(C,f_r'), vcat(d,l_r), vcat(C,f_l'), vcat(d,l_l)
#         gen_hyper += 1
#         push!(hyp_list, [f_opt, l_opt])
#
#         coeff_list_r = copy(coeff_list_cur)
#         coeff_list_l = copy(coeff_list_cur)
#
#         # Right child
#         res_r, x_lb_r, X_lb_r, lb_r = solve_rpt_relaxation_X1(C_r,d_r,coeff_list_r,sdp_vecs)
#         if res_r
#             X_r = calculate_candidate_vectors(x_lb_r,X_lb_r)
#             ub_r = get_ub_X1(X_r,C_r,d_r,c4,c3,c2,c1,c0)
#             ### extra ####
#             # violating_vecs_curr = find_sdp_violating_vecs(x_lb_r,X_lb_r)
#             # append!(sdp_vecs, violating_vecs_curr)
#             ##################
#             if length(coeff_list_cur) < 100
#                 res_r_2, Z_r = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb_r,X_lb_r)
#                 if res_r_2
#                     push!(coeff_list_r, Z_r)
#                 end
#             end
#             # res_r_2, Z_r = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb_r,X_lb_r)
#             # if res_r_2
#             #     push!(coeff_list_r, Z_r)
#             # end
#             if lb_r < UB
#                 push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r,coeff_list_r])
#             end
#             # sdp_vecs = add_approx_lmi(C_r,d_r,coeff_list_r,sdp_vecs)
#         end
#         # Left child
#         res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_X1(C_l,d_l,coeff_list_l,sdp_vecs)
#         if res_l
#             X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
#             ub_l = get_ub_X1(X_l,C_l,d_l,c4,c3,c2,c1,c0)
#             ### extra ####
#             # violating_vecs_curr = find_sdp_violating_vecs(x_lb_l,X_lb_l)
#             # append!(sdp_vecs, violating_vecs_curr)
#             ##################
#             if length(coeff_list_cur) < 100
#                 res_l_2, Z_l = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb_l,X_lb_l)
#                 if res_l_2
#                     push!(coeff_list_l, Z_l)
#                 end
#             end
#             # res_l_2, Z_l = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb_l,X_lb_l)
#             # if res_l_2
#             #     push!(coeff_list_l, Z_l)
#             # end
#             if lb_l < UB
#                 push!(nodes_list,[ub_l,lb_l,x_lb_l,X_lb_l,C_l,d_l,coeff_list_l])
#             end
#             # sdp_vecs = add_approx_lmi(C_l,d_l,coeff_list_r,sdp_vecs)
#         end
#
#         if isempty(nodes_list)
#             if res_r
#                 push!(ub_list, ub_r)
#                 push!(lb_list, lb_r)
#             end
#             if res_l
#                 push!(ub_list, ub_l)
#                 push!(lb_list, lb_l)
#             end
#             break
#         else
#             ind = argmin([nodes_list[i][2] for i in 1:length(nodes_list)])
#             cur_node = nodes_list[ind]
#             deleteat!(nodes_list, ind)
#             ub, lb = cur_node[1], cur_node[2]
#             x_cur, X_cur = cur_node[3], cur_node[4]
#             C, d = cur_node[5], cur_node[6]
#             coeff_list_cur = cur_node[7]
#             LB = lb
#             if ub < UB
#                 UB = ub
#                 opt_sol, opt_val = cur_node[3], cur_node[1]
#             end
#             push!(ub_list, ub)
#             push!(lb_list, lb)
#         end
#         t0_2 = time_ns()
#         total_time = (t0_2-t0_1)*10^(-9)
#     end
#     return opt_sol, opt_val, gen_hyper, ub_list, lb_list, coeff_list_cur
# end
#
# function rpt_bb_cut_planes_X1_new(c4,c3,c2,c1,c0,C_init,d_init,δ)
#     C, d = C_init, d_init
#     gen_hyper = 0
#     coeff_list_cur = []
#     sdp_vecs = []
#     # sdp_vecs = add_approx_lmi(C,d,coeff_list_cur,sdp_vecs)
#
#     unc_set_list = get_uncertainty_set_z1(c3,c2,c1,c0)
#     res, x_lb, X_lb, lb = solve_rpt_relaxation_X1_best_slc(unc_set_list,C,d,use_lmi)
#     res, Z_0 = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb,X_lb)
#     push!(coeff_list_cur, Z_0)
#
#     # Root Node
#     res_root, x_lb, X_lb, lb = solve_rpt_relaxation_X1(C,d,coeff_list_cur,sdp_vecs)
#     X = calculate_candidate_vectors(x_lb,X_lb)
#     ub = get_ub_X1(X,C,d,c4,c3,c2,c1,c0)
#     ### extra ####
#     violating_vecs_curr = find_sdp_violating_vecs(x_lb,X_lb)
#     append!(sdp_vecs, violating_vecs_curr)
#     ##################
#
#     res_root_1, Z_1 = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb,X_lb)
#     if res_1
#         push!(coeff_list_cur, Z_1)
#         # coeff_list_cur = [Z_1]
#     end
#
#     x_cur, X_cur = x_lb, X_lb
#
#     UB, LB, opt_sol, opt_val = ub, lb, x_lb, ub
#
#     nodes_list = []
#     ub_list, lb_list, hyp_list = [], [], []
#     push!(ub_list, ub)
#     push!(lb_list, lb)
#
#     t0_1 = time_ns()
#     total_time = 0.0
#
#     # while (abs(UB - LB)/abs(UB))*100 > δ #&& total_time < 60
#     while UB - LB > δ && total_time < 600
#         # res, f_opt, l_opt = generate_hyperplane_eigen(x_cur,X_cur)
#         res, f_opt, l_opt = generate_hyperplane_new(x_cur,X_cur,coeff_list_cur[end])
#         # res, f_opt, l_opt = generate_hyperplane_new_2(x_cur,X_cur,coeff_list_cur)
#
#         f_r, l_r, f_l, l_l  = f_opt, l_opt, -f_opt, -l_opt
#         C_r, d_r, C_l, d_l = vcat(C,f_r'), vcat(d,l_r), vcat(C,f_l'), vcat(d,l_l)
#         gen_hyper += 1
#         push!(hyp_list, [f_opt, l_opt])
#
#         coeff_list_r = copy(coeff_list_cur)
#         coeff_list_l = copy(coeff_list_cur)
#
#         # Right child
#         res_r, x_lb_r, X_lb_r, lb_r = solve_rpt_relaxation_X1(C_r,d_r,coeff_list_r,sdp_vecs)
#         if res_r
#             X_r = calculate_candidate_vectors(x_lb_r,X_lb_r)
#             ub_r = get_ub_X1(X_r,C_r,d_r,c4,c3,c2,c1,c0)
#             ### extra ####
#             # violating_vecs_curr = find_sdp_violating_vecs(x_lb_r,X_lb_r)
#             # append!(sdp_vecs, violating_vecs_curr)
#             ##################
#             if length(coeff_list_cur) < 100
#                 res_r_2, Z_r = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb_r,X_lb_r)
#                 if res_r_2
#                     push!(coeff_list_r, Z_r)
#                 end
#             end
#             # res_r_2, Z_r = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb_r,X_lb_r)
#             # if res_r_2
#             #     push!(coeff_list_r, Z_r)
#             # end
#             if lb_r < UB
#                 push!(nodes_list,[ub_r,lb_r,x_lb_r,X_lb_r,C_r,d_r,coeff_list_r])
#             end
#             # sdp_vecs = add_approx_lmi(C_r,d_r,coeff_list_r,sdp_vecs)
#         end
#         # Left child
#         res_l, x_lb_l, X_lb_l, lb_l = solve_rpt_relaxation_X1(C_l,d_l,coeff_list_l,sdp_vecs)
#         if res_l
#             X_l = calculate_candidate_vectors(x_lb_l,X_lb_l)
#             ub_l = get_ub_X1(X_l,C_l,d_l,c4,c3,c2,c1,c0)
#             ### extra ####
#             # violating_vecs_curr = find_sdp_violating_vecs(x_lb_l,X_lb_l)
#             # append!(sdp_vecs, violating_vecs_curr)
#             ##################
#             if length(coeff_list_cur) < 100
#                 res_l_2, Z_l = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb_l,X_lb_l)
#                 if res_l_2
#                     push!(coeff_list_l, Z_l)
#                 end
#             end
#             # res_l_2, Z_l = get_poly4_best_slc(c4,c3,c2,c1,c0,x_lb_l,X_lb_l)
#             # if res_l_2
#             #     push!(coeff_list_l, Z_l)
#             # end
#             if lb_l < UB
#                 push!(nodes_list,[ub_l,lb_l,x_lb_l,X_lb_l,C_l,d_l,coeff_list_l])
#             end
#             # sdp_vecs = add_approx_lmi(C_l,d_l,coeff_list_r,sdp_vecs)
#         end
#
#         if isempty(nodes_list)
#             if res_r
#                 push!(ub_list, ub_r)
#                 push!(lb_list, lb_r)
#             end
#             if res_l
#                 push!(ub_list, ub_l)
#                 push!(lb_list, lb_l)
#             end
#             break
#         else
#             ind = argmin([nodes_list[i][2] for i in 1:length(nodes_list)])
#             cur_node = nodes_list[ind]
#             deleteat!(nodes_list, ind)
#             ub, lb = cur_node[1], cur_node[2]
#             x_cur, X_cur = cur_node[3], cur_node[4]
#             C, d = cur_node[5], cur_node[6]
#             coeff_list_cur = cur_node[7]
#             LB = lb
#             if ub < UB
#                 UB = ub
#                 opt_sol, opt_val = cur_node[3], cur_node[1]
#             end
#             push!(ub_list, ub)
#             push!(lb_list, lb)
#         end
#         t0_2 = time_ns()
#         total_time = (t0_2-t0_1)*10^(-9)
#     end
#     return opt_sol, opt_val, gen_hyper, ub_list, lb_list, coeff_list_cur
# end
#
#
# function round_coefficients(Z)
#     β_list, P_list, r_list, w_list = Z[1], Z[2], Z[3], Z[4]
#     γ_list, Q_list, f_list, g_list = Z[5], Z[6], Z[7], Z[8]
#     β_list_rounded, P_list_rounded, r_list_rounded, w_list_rounded = [], [], [], []
#     γ_list_rounded, Q_list_rounded, f_list_rounded, g_list_rounded = [], [], [], []
#     n = size(r_list[1],1)
#     for l in 1:length(β_list)
#         β = β_list[l]
#         P = P_list[l]
#         r = r_list[l]
#         w = w_list[l]
#         γ = γ_list[l]
#         Q = Q_list[l]
#         f = f_list[l]
#         g = g_list[l]
#
#         β_new = zeros(size(β))
#         P_new = zeros(size(P))
#         r_new = zeros(size(r))
#         γ_new = zeros(size(γ))
#         Q_new = zeros(size(Q))
#         f_new = zeros(size(f))
#         for i in 1:n
#             for j in 1:n
#                 for k in 1:n
#                     β_new[i,j,k] = round(β[i,j,k], digits=4)
#                     γ_new[i,j,k] = round(γ[i,j,k], digits=4)
#                 end
#             end
#         end
#         for i in 1:n
#             for j in 1:n
#                 P_new[i,j] = round(P[i,j], digits=4)
#                 Q_new[i,j] = round(Q[i,j], digits=4)
#             end
#         end
#         for i in 1:n
#             r_new[i] = round(r[i], digits=4)
#             f_new[i] = round(f[i], digits=4)
#         end
#         w_new = round(w, digits=4)
#         g_new = round(g, digits=4)
#
#         push!(β_list_rounded, β_new)
#         push!(P_list_rounded, P_new)
#         push!(r_list_rounded, r_new)
#         push!(w_list_rounded, w_new)
#         push!(γ_list_rounded, γ_new)
#         push!(Q_list_rounded, Q_new)
#         push!(f_list_rounded, f_new)
#         push!(g_list_rounded, g_new)
#     end
#     Z_rounded = [β_list_rounded, P_list_rounded, r_list_rounded, w_list_rounded,
#                  γ_list_rounded, Q_list_rounded, f_list_rounded, g_list_rounded]
#
#     return Z_rounded
# end


#
# n = 20
# id_mat = zeros(n,n) + I
# c4 = zeros(n,n,n,n)
# c4[1,1,1,1] = 1
# c4[2,2,2,2] = -3
# c4[4,4,4,4] = -6
# c4[6,6,6,6] = -6
# # c4[8,8,8,8] = -6
# c4[10,10,10,10] = 1
# c4[11,11,11,11] = 1
# # c4[14,14,14,14] = -6
# c4[15,15,15,15] = -6
# # c4[17,17,17,17] = -3
# c4[18,18,18,18] = -8
# c4[20,20,20,20] = 1
# c3 = zeros(n,n,n)
# c3[1,1,1] = 4
# c3[2,2,2] = -2
# c3[4,4,4] = 4
# c3[5,5,5] = -2
# c3[7,7,7] = -2
# c3[9,9,9] = 3
# c3[10,10,10] = -1
# c3[12,12,12] = 3
# c3[13,13,13] = -1
# c3[15,15,15] = 4
# # c3[17,17,17] = 4
# c3[18,18,18] = -6
# c3[20,20,20] = 1
# c2 = -5*id_mat
# c1 = 2*ones(n)
# c1[2] = 0
# c1[3] = 0
# c1[4] = 0
# c1[5] = 0
# c1[7] = 0
# c1[9] = 0
# c1[11] = 0
# c1[13] = 0
# c1[17] = 0
# c1[18] = 0
# c0 = 0
# # C, d = id_mat, ones(n)
# C, d = vcat(id_mat, -id_mat), vcat(0.99*ones(n), -0.01*ones(n))
#
# t1 = time_ns()
# x_opt, obj_opt, gen_hyper, ub_list, lb_list, cut_pol_list = rpt_bb_cut_planes_poly4(c4,c3,c2,c1,c0,C,d,δ)
# t2 = time_ns()
# total_time_1 = (t2-t1)*10^(-9)
