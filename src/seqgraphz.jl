import NumericExtensions.entropy
import Base.BLAS.axpy!

immutable seq_graph_z{Tp}
    G::Matrix{Float64}
    c::Matrix{Float64}
    r::Matrix{Float64}

    theta::Vector{Tp}

    cluster_update::Function
    cluster_likelihood::Function
    object_likelihood::Function
    cluster_entropy::Function

    nzs::Vector{Tsp}

    function seq_graph_z(theta::Vector{Tp}, F::Matrix{Float64}, c::Matrix{Float64}, 
        cluster_update::Function, cluster_likelihood::Function, object_likelihood::Function, 
        cluster_entropy::Function; nzs::Union(Vector{Tsp}, Nothing)=nothing)
        N = size(F, 1)
        r = zeros(N, N)
        if nzs == nothing
            nzs = get_nzs(F)
        end
        get_r!(r, c)

        return new(log(F), c, r, theta, 
            cluster_update, cluster_likelihood, object_likelihood, 
            cluster_entropy, nzs)
    end
end 

function seq_graph_z{Tp}(theta::Vector{Tp}, F::Matrix{Float64}, cluster_update::Function, 
    cluster_likelihood::Function, object_likelihood::Function, cluster_entropy::Function; 
    init_c::Function = singular_init_dist, nzs::Union(Vector{Tsp}, Nothing)=nothing)
    if nzs == nothing
        nzs = get_nzs(F)
    end
    c = init_c(F, nzs)

    return seq_graph_z{Tp}(theta, F, c, cluster_update, cluster_likelihood, object_likelihood, cluster_entropy; nzs=nzs)
end

function seq_graph_z{Tp}(default_theta::Tp, F::Matrix{Float64}, cluster_update::Function, cluster_likelihood::Function, object_likelihood::Function, cluster_entropy::Function; init_c::Function = singular_init_dist, nzs::Union(Vector{Tsp}, Nothing)=nothing)
    N = size(F, 1)
    theta = Array(Tp, N)
    for i=1:N
        theta[i] = default_theta
    end

    return seq_graph_z{Tp}(theta, F, cluster_update, cluster_likelihood, object_likelihood, cluster_entropy; init_c=init_c, nzs=nzs)
end

function num_obj(crp::seq_graph_z) 
    return size(crp.c, 1)
end

function var_lower_bound(crp::seq_graph_z)
    return loglikelihood(crp) + entropy(crp)
end

function get_r!(r::Matrix{Float64}, c::Matrix{Float64}, start_i::Int64=1)
    N = size(r, 1)
    for i in start_i:N
        r[i, :] = 0.
        for j in 1:i-1
            r[i, :] += c[i, j] * r[j, :]
        end
        r[i, i] = 1.        
    end
end

function get_r!(crp::seq_graph_z, start_i::Int64=1)
    get_r!(crp.r, crp.c, start_i)    
end

function loglikelihood(crp::seq_graph_z)
    ll = sublikelihood(crp)
    assert(!isnan(ll))
    assert(!isinf(ll))

    for i in 1:num_obj(crp)
        for j in crp.nzs[i].rowval
            ll += crp.c[i, j] * crp.G[i, j]
        end
    end

    return ll
end

function sublikelihood(crp::seq_graph_z)
    ll = 0.

    zk = zeros(num_obj(crp))
    for k in 1:num_obj(crp)
        #axpy!(crp.c[k, k], crp.r[:, k], zk)
        zk[:] = crp.r[:, k] * crp.c[k, k]
        ll += crp.cluster_likelihood(crp.theta[k], zk)
    end

    return ll
end

function entropy(crp::seq_graph_z)
    ee = entropy(crp.c)

    for i in 1:num_obj(crp)
        ee += crp.cluster_entropy(crp.theta[i])
    end

    return ee
end

function var_update!(crp::seq_graph_z, i::Int64, ci::Vector{Float64}, 
        a::Vector{Float64}=zeros(i); treshold=0.)
    ci[:] = -Inf

    crp.c[i, :] = 0.
    crp.c[i, i] = 1.

    a[:] = 0.
    for k in 1:i    
        if crp.c[k, k] < treshold
            continue
        end
        for s in i:num_obj(crp)
            a[k] += crp.r[s, i] * crp.object_likelihood(crp.theta[k], s)        
        end
        a[k] *= crp.c[k, k]
    end

    for j in 1:i
        ci[j] = crp.G[i, j]
        for k in 1:j
            ci[j] += a[k] * crp.r[j, k]
        end
    end

    adjust!(ci)
end

function alt_var_update!(crp::seq_graph_z, i::Int64, ci::Vector{Float64})
    ci[:] = -Inf

    crp.c[i, :] = 0.
    for j in 1:i
        crp.c[i, j] = 1.
        get_r!(crp, i)
        ci[j] = loglikelihood(crp)
        crp.c[i, j] = 0.
    end

    adjust!(ci)
end

function var_update_clusters(crp::seq_graph_z; start_k::Integer=1)
    zk = zeros(num_obj(crp))
    for k in start_k:num_obj(crp)
        zk[:] = crp.r[:, k] * crp.c[k, k]
        crp.theta[k] = crp.cluster_update(zk)
    end
end

expected_num_clusters(crp::seq_graph_z) = sum([crp.c[j, j] for j in 1:num_obj(crp)])

function infer(crp::seq_graph_z, niter::Int64, ltol::Float64=1e-5; 
    iter_callback::Function = (iter, ll) -> nothing, start_i::Int64=1,
    treshold::Float64=0.)
    N = num_obj(crp)
    ci = zeros(N)
    cj = zeros(N)
    zk = zeros(N)

    u = zeros(N)
    dX = zeros(N, N)

    prev_ll = var_lower_bound(crp)
    iter_callback(0, prev_ll)    

    prev_lb = -Inf

    for iter in 1:niter 
        for i in randperm(num_obj(crp)) #1:num_obj(crp) #
            if i < start_i
                continue
            end
            var_update!(crp, i, ci; treshold=treshold)

            #alt_var_update!(crp, i, cj)       
            #println(vnormdiff(ci, cj, 1.))

            # alpha = ci[i]
            # ci[i] = 0.
            # @devec u[:] = crp.c[i, :] .- ci
            # ci[i] = alpha

            #prepare_update_row!(crp.r, dX, i, u)
            #NumericExtensions.add!(crp.r, dX)
            crp.c[i, :] = ci
            crp.r[:, :] = inv(eye(num_obj(crp)) - crp.c + diagm(diag(crp.c)))

            if false
                lb = var_lower_bound(crp)
                #println(i, "/", num_obj(crp), " ", lb, " ", sum([crp.c[j, j] for j in 1:num_obj(crp)]))
                if lb < prev_lb
                    warn(string("not monotone !!!", prev_lb - lb))
                end
                prev_lb = lb
            end
        end
        get_r!(crp)

        var_update_clusters(crp)

        ll = var_lower_bound(crp)
        println("iteration $iter/$niter lower bound=$ll, clusters=$(expected_num_clusters(crp))")
        iter_callback(iter, ll)

        if abs(ll - prev_ll) < ltol
#            println("converged")
            return
        end
        if ll < prev_ll
            println("not monotone")
        end

        prev_ll = ll
    end

    return prev_ll
end

function map_assignments(crp::seq_graph_z)
    z = zeros(Int64, num_obj(crp))
    zi = zeros(num_obj(crp))
    for i in 1:num_obj(crp)
        zi[:] = crp.r[i, :]
        for k in 1:i
            zi[k] *= crp.c[k, k]
        end
        z[i] = indmax(zi)
    end

    return z
end

function sample_test_likelihood{Tp}(crp::seq_graph_z{Tp}, F1::Matrix{Float64}, 
    nsamples::Int64, predictive::Function, prior::Union(Tp, Nothing))

    const N = num_obj(crp)
    const M = size(F1, 1)
    F = zeros(M, M)
    for i in N+1:M
        F[i, :] = F1[i, :]
        if prior == nothing
            F[i, i] = 0.
            F[i, :] /= sum(F[i, :])
        end
    end
    for i in 1:N
        F[i, 1:N] = crp.c[i, :]
    end

    zk = zeros(M)
    c = zeros(Int64, M)
    tl = 0.
    for t in 1:nsamples
        for i in 1:M
            c[i] = rand(Categorical(vec(F[i, :])))
        end
        ll = 0.
        for (k, zk) in iterate_clusters(c, zk)
            zk[1:N] = 0.
            theta = k > N ? prior : crp.theta[k]
            if theta != nothing
                ll += predictive(theta, zk)
            end
        end
        tl += exp(ll)
    end
    return log(tl) - log(nsamples)
end

export seq_graph_z, num_obj, map_assignments
export singular_init_z, rand_init_z, uniform_init_z
export infer, var_lower_bound, loglikelihood, entropy
export sample_test_likelihood
