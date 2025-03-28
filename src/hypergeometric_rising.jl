hypgeom_rising(x::Arb, n::Arb; prec = Arblib._precision(x)) = Arblib.hypgeom_rising!(Arb(; prec), x, n; prec = prec)

function hypgeom_rising(x::Dual{<:Any, Arb}, n::Arb; prec = nothing)
    @assert isinteger(n)
    return prod(i -> x + i, 0:Int(n - 1))
end
