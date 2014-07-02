function read_word_topic(filename::String)
    f = open(filename)
    φ = readdlm(f, ',')
    close(f)
    φ = φ'
    V, K = size(φ)

    for k=1:K
        φ[:, k] /= sum(φ[:, k])
    end

    return φ
end

function read_term_index(filename::String)
    f = open(filename)

    term2index = Dict{String, Int64}()
    index2term = Array(String, 0)

    v = 1
    for word in eachline(f)
        w = strip(word)
        push!(term2index, w, v)
        push!(index2term, w)
        
        v += 1
    end
    close(f)

    return term2index, index2term
end

function read_term_count(filename::String)
    f = open(filename)

    term2index = Dict{String, Int64}()
    index2term = Array(String, 0)
    word_count = Array(Float64, 0)

    v = 1
    for line in eachline(f)
        data = split(strip(line))
        if length(data) <= 1
            continue
        end
        w = data[1]
        push!(term2index, w, v)
        push!(index2term, w)
        push!(word_count, float(data[2]))
        
        v += 1
    end
    close(f)

    return term2index, index2term, word_count
end

function read_dataset(filename::String)
    f = open(filename)
    data = readdlm(f, ',')
    close(f)

    D = size(data, 1)
    for d=1:D
        w = data[d, 2]
        try
            w = replace(w, '\"', "")
            w = split(w, ' ')
        catch e
            println("Document $d/$D wasn't read because of error")
#            throw e
        end
#        map!(strip, w)
        data[d, 2] = w
    end

    return data
end

function read_formatted_dataset(filename::String)
    f = open(filename)
    data = readdlm(f, ',')
    close(f)

    D = size(data, 1)
    for d=1:D
        idx = split(strip(replace(data[d, 2], '\"', "")))
        data[d, 2] = map(int, idx) + 1
    end

    return data
end


function index(w::Vector{String}, term2index::Dict{String, Int64})
#    N = length(w)
#    return map(v -> get(term2index, v, -1), w)
    return filter(v -> v != -1, [
        begin 
            v = get(term2index, word, -1)
            if v == -1 
#println("unknown word $word")
            end
            v
        end 
        for word in w])
end
