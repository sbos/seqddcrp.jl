function conditional!(cp::Vector{Float64}, G::Matrix{Float64}, graph::RestrauntGraph, i::Int64, cluster_likelihood::Function, nzs::Vector{Tsp})
    N = size(G, 1)

    cp[:] = -Inf
    for j in nzs[i].rowval
        if isinf(G[i, j]) continue end

        if i == j
            cp[j] = G[i, i]
            continue
        end

        remove_link!(graph, i)
        K = numcomponents(graph)

        k_cluster = [graph.components[vertex_component(graph, i)]...]
        l_cluster = [graph.components[vertex_component(graph, j)]...]

        add_link!(graph, i, j)
        if numcomponents(graph) == K
            cp[j] = G[i, j]
            continue
        end

        assert(K == numcomponents(graph)+1)

        kl_cluster = vcat(k_cluster, l_cluster)

        kl_clust_l = cluster_likelihood(kl_cluster)
        k_clust_l = cluster_likelihood(k_cluster)
        l_clust_l = cluster_likelihood(l_cluster)
        
        cp[j] = G[i, j] + kl_clust_l - k_clust_l - l_clust_l
#        println("$i -> $j: ", G[i, j], " ", kl_clust_l, " ", k_clust_l, " ", l_clust_l)
#        println("$i -> $j: ", cp[j])
    end

#    println(i, " ", sum(cp[1:i-1]), " ", cp[i])    

    adjust!(cp)
#    println(i, " ", repr(cp))
    
end

function collapsed_gibbs(F::Matrix{Float64}, cluster_likelihood::Function, burnin::Int64, niter::Int64; nzs::Union(Vector{Tsp}, Nothing)=nothing, iter_callback::Function=(c, z, ll) -> true, init_c::Function = singular_init_discrete)
    N = size(F, 1)
    G = log(F)

    samples = zeros(Int64, N, niter)
    loglike = zeros(niter)

    if nzs == nothing
        nzs = Array(Tsp, N)
        for i=1:N
            nzs[i] = sparsevec(vec(F[i, :]))
        end
    end

    graph = RestrauntGraph(init_c(F, nzs))

    cp = log(zeros(N))

    function iterate()
        for i in randperm(N) # [N]
            conditional!(cp, G, graph, i, cluster_likelihood, nzs)
            remove_link!(graph, i)
            j = rand(Categorical(cp))
            add_link!(graph, i, j)
#            println("$i -> $j")
        end
    end

    println("Init: ", loglikelihood(graph.c, G, graph.components, cluster_likelihood))

    for iter=1:burnin
        iterate()
    end

    for iter=1:niter
        @time iterate()
        samples[:, iter] = graph.c
        loglike[iter] = loglikelihood(graph.c, G, graph.components, cluster_likelihood)

        if !iter_callback(graph.c, graph.components, loglike[iter])
            break
        end
    end

    return samples, loglike
end

function loglikelihood(graph::RestrauntGraph, G::Matrix{Float64}, cluster_likelihood::Function)
    return loglikelihood(graph.c, G, graph.components, cluster_likelihood)    
end

function loglikelihood(c::Vector{Int64}, G::Matrix{Float64}, z::Vector{Ts}, cluster_likelihood::Function)
    N = length(c)
    ll = 0.
    
    for i=1:N
        ll += G[i, c[i]]
    end

    ll += sublikelihood(c, z, cluster_likelihood)
    return ll
end

function sublikelihood(c::Vector{Int64}, z::Vector{Ts}, cluster_likelihood::Function)
    ll = 0.

    K = length(z)
    for k=1:K
        zk = [z[k]...]
        ll += cluster_likelihood(zk)
    end

    return ll
end

function predictive_posterior(samples::Matrix{Int64}, loglike::Vector{Float64}, F::Matrix{Float64}, cluster_likelihood::Function, niter::Int64, nzs::Union(Vector{Tsp}, Nothing)=nothing)
    G = log(F)
    N, S = size(samples)
    M = size(F, 1)

    esamples = zeros(Int64, M-N, niter)
    cp = zeros(M, M-N)
    for iter=1:niter 
        for i=N+1:M
            cp[:, i-N] = 0.
            if nzs != nothing
                cs = 0.
                for j in nzs[i].rowval
                    cp[j, i-N] = F[i, j]
                    cs += F[i, j]
                end
                cp[:, i-N] /= cs
            else
                cp[:, i-N] = F[i, :]
            end
            esamples[i-N, iter] = rand(Categorical(cp[:, i-N]))
        end
    end

    results = zeros(S)
    R = zeros(niter, S)

    for iter=1:niter
        c = zeros(Int64, M)
        c[N+1:M] = esamples[:, iter]
        ap = sum([G[i, c[i]] for i=N+1:M])

        ks = KahanState()

        for s=1:S
            c[1:N] = samples[:, s]
            graph = RestrauntGraph(c)
            add!(ks, exp(loglikelihood(graph, G, cluster_likelihood) - loglike[s] - ap))
            R[iter, s] = ks.sum
        end

        println("Predictive $iter/$niter")
    end

    for s=1:S
        ks = KahanState()
        for iter=1:niter
            add!(ks, R[iter, s])
        end
        results[s] = log(ks.sum) - log(s) - log(niter)
    end

    return results
end

function singular_init_discrete(F::Matrix{Float64}, nzs::Vector{Tsp})
    N = size(F, 1)
    return [1:N]
end

function allfirst_init_discrete(F::Matrix{Float64}, nzs::Vector{Tsp})
    N = size(F, 1)
    return [1 for i=1:N]
end

function prior_init_discrete(F::Matrix{Float64}, nzs::Vector{Tsp})
    return draw_c(F)
end

function reweighted_init_discrete(F::Matrix{Float64}, nzs::Vector{Tsp})
    N = size(F, 1)

    c = zeros(Int64, N)
    cp = zeros(N)

    for i=1:N
        cp[:] = 0.
        cs = 0.

        for j in nzs[i].rowval
            cp[j] = F[i, j]
            cs += F[i, j]
        end

        cp[:] /= cs
        c[i] = rand(Categorical(cp))
    end

    return c
end

function uniform_init_discrete(F::Matrix{Float64}, nzs::Vector{Tsp})
    N = size(F, 1)

    c = zeros(Int64, N)
    cp = zeros(N)

    for i=1:N
        cp[:] = 0.

        M = length(nzs[i].rowval)
        for j in nzs[i].rowval
            cp[j] = 1 / M
        end

        c[i] = rand(Categorical(cp))
    end

    return c
end

export collapsed_gibbs, loglikelihood, singular_init_discrete, prior_init_discrete, reweighted_init_discrete, uniform_init_discrete
export predictive_posterior, allfirst_init_discrete
