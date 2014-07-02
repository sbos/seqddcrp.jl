using Devectorize
using NumericExtensions
    
function one_hot(N, i)
    x = zeros(N)
    x[i] = 1.
    return x
end

Tsp = SparseMatrixCSC{Float64, Int64}

function adjust!(x)
	subtract!(x, maximum(x))
    exp!(x)
    divide!(x, sum(x))
end

function iterate_clusters(c::Vector{Int64}, buf::Vector{Float64})
	const N = length(c)

	refs = [IntSet() for i in 1:N]
	for i in 1:N
		if c[i] != i
			push!(refs[c[i]], i)
		end
	end

	function search(i::Int64)
		buf[i] = 1.
		for j in refs[i]
			buf[j] = 1.
			search(j)
		end
	end

	function iterable()
		for k in 1:N
			if c[k] != k continue; end
			buf[:] = 0.
			search(k)
			produce((k, buf))
		end
	end

	return Task(iterable)
end