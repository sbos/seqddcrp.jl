type KahanState{Tv <: FloatingPoint}
    sum::Tv
    corr::Tv

    KahanState() = new(0., 0.)

    KahanState(sum::Tv, corr::Tv) = new(sum, corr)
end

function KahanState()
    return KahanState{Float64}()
end

function add!{Tv <: FloatingPoint}(s::KahanState{Tv}, x::Tv)
    y = x - s.corr
    t = s.sum + y
    s.corr = (t - s.sum) - y
    s.sum = t
end

function add{Tv <: FloatingPoint}(s::KahanState{Tv}, x::Tv)
    sx = KahanState(s.sum, s.corr)
    add!(sx, x)
    return sx
end

