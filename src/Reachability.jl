Ts = IntSet #Set{Int64}

immutable RestrauntGraph
    c::Array{Int, 1}
    refs::Array{Ts, 1}
    components::Array{Ts, 1}

    function RestrauntGraph(c::Array{Int, 1}, refs::Array{Ts, 1}, components::Array{Ts, 1})
        __check_refs(c, refs)

        all = union(components...)
        veryall = Ts([1:length(c)]...)
        margin = setdiff(veryall, all)

        @assert length(margin) == 0 margin

        K = length(components)
        for k=1:K
            for l=1:K
                if k != l
                    assert(length(intersect(components[k], components[l])) == 0)
                end
            end
        end

        return new(c, refs, components)
    end
end

function RestrauntGraph(c::Array{Int, 1})
    N = length(c)
    refs = map(x -> Ts(), [1:N])
    for i=1:N
        push!(refs[c[i]], i)
    end

    __check_refs(c, refs)
    return RestrauntGraph(c, refs, connected_components(c, refs))
end

function RestrauntGraph(c::Array{Int, 1}, refs::Array{Ts, 1})
    __check_refs(c, refs)

    return RestrauntGraph(c, refs, connected_components(c, refs))
end

function __check_refs(c::Vector{Int64}, refs::Vector{Ts})
    N = length(c)
    for i=1:N
        for j=i+1:N
            @assert length(intersect(refs[i], refs[j])) == 0
        end
        if c[i] == 0
            continue
        end
        @assert i in refs[c[i]]
    end
end

function walk(c::Array{Int, 1}, refs::Array{Ts, 1}, v::Int)
    queue = Ts(v)
    visited = Ts()

    while length(queue) > 0
        u = pop!(queue)
        push!(visited, u)

        for ref in refs[u]
            if !(ref in visited)
                push!(queue, ref)
            end
        end

#        union!(queue, setdiff(refs[u], visited))
        
        y = c[u]
        if y > 0 && !(y in visited)
            push!(queue, y)
        end
    end

    return visited
end

function connected_components(c::Array{Int, 1}, refs::Array{Ts, 1})
    N = length(c)
    components = Array(Ts, 0)
    remaining_vertices = Ts([1:N]...)

    while length(remaining_vertices) > 0
        component = walk(c, refs, pop!(remaining_vertices))
        for i in component
            if i in remaining_vertices
                delete!(remaining_vertices, i)
            end
        end

#        remaining_vertices = setdiff(remaining_vertices, component)
        push!(components, component)
    end

    return components
end

numcomponents(graph::RestrauntGraph) = length(graph.components)

numvertices(graph::RestrauntGraph) = length(graph.c)

vertex_component(graph::RestrauntGraph, v) = vertex_component(graph.components, v)

function vertex_component(components::Array{Ts, 1}, v)
    for k=1:length(components)
        if v in components[k]
            return k
        end
    end
    return -1
end

function add_link!(graph::RestrauntGraph, i, j)
    assert(graph.c[i] == 0)
    
    c = graph.c
    refs = graph.refs
    components = graph.components

    c[i] = j
    push!(refs[j], i)

    i_c = vertex_component(components, i)
    j_c = vertex_component(components, j)

    @assert (i_c != -1)
    @assert (j_c != -1)

    if i_c == j_c
        return graph
    end

    new_cluster = union(components[i_c], components[j_c])
    splice!(components, i_c)
    if i_c < j_c
        splice!(components, j_c-1)
    else
        splice!(components, j_c)
    end

    push!(components, new_cluster)

#    __check_refs(c, refs)

    return graph
end


function add_link(graph::RestrauntGraph, i, j)
    @assert (graph.c[i] == 0)
    
    c = copy(graph.c)
    refs = deepcopy(graph.refs)
    components = deepcopy(graph.components)
 
    return add_link!(RestrauntGraph(c, refs, components), i, j)
end

function remove_link(graph::RestrauntGraph, i)
    if graph.c[i] == 0
        return graph
    end

    c = copy(graph.c)
    refs = deepcopy(graph.refs)
    components = deepcopy(graph.components)

    return remove_link!(RestrauntGraph(c, refs, components), i)
end

function remove_link!(graph::RestrauntGraph, i)
    if graph.c[i] == 0
        return graph
    end

    c = graph.c
    refs = graph.refs

    j = c[i]
    ref_c = refs[j]
    delete!(ref_c, i)
#    refs[j] = ref_c
    c[i] = 0

    components = graph.components

    i_cluster = walk(c, refs, i)
    if j in i_cluster
        return graph
    end

    splice!(components, vertex_component(graph, i))
    push!(components, i_cluster)
    push!(components, walk(c, refs, j))

#    __check_refs(c, refs)

    return graph
end

