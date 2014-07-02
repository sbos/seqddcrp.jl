function time_dist(N::Int64) 
    D = zeros(N, N)
    for i=1:N
        for j=1:N
            D[i, j] = abs(i - j)
        end
    end
    return D
end
    
function ddcrp_F(f, D, alpha, a, sparse_c=false)
    N = size(D, 1)

    f_(x) = f(x, a)
    C = map(f_, D)
    for i=1:N
        C[i, i] = alpha
        C[i, :] /= sum(C[i, :])
    end

    return if sparse_c sparse(C) else C end
end

function seqddcrp_F(f, D, alpha, a, sparse_c=false)
    N = size(D, 1)
    C = zeros(N, N)

    for i=1:N
        for j=i-1:-1:1
            C[i, j] = f(D[i, j], a)
        end
        C[i, i] = alpha
        C[i, :] /= sum(C[i, :])
    end

    return sparse_c ? sparse(C) : C
end

function crp_F(N, alpha)
    C = zeros(N, N)
    for i=1:N
        C[i, i-1:-1:1] = 1
        C[i, i] = alpha
        C[i, :] /= sum(C[i, :])
    end
    return C
end

exp_decay(d, a) = exp(-d / a)

window_decay(d, a) = d < a ? 1. : 0.

linear_window_decay(d, a) = d < a ? 1. - d/a : 0.

sigmoid_decay(d, a) = exp(-d + a) / (1 + exp(-d + a))

exp_window_decay(d, a) = d < a[1] ? exp(-d / a[2]) : 0

draw_c(F) = [rand(Categorical(vec(F[i, :]))) for i=1:size(F, 1)]

function expand(z::Vector{Int64})
    N = length(z)
    c = zeros(N, N)
    for i=1:N
        c[i, z[i]] = 1.
    end
    return c
end

export draw_c, time_dist, ddcrp_F, crp_F, exp_decay, window_decay, linear_window_decay
export sigmoid_decay, exp_window_decay, expand
export seqddcrp_F
