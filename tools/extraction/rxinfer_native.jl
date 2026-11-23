# Pinned RxInfer 3.6 / Julia 1.10 binary worker for affine extraction slices.
using RxInfer
using LinearAlgebra

const NoCorrection = ReactiveMP.MatrixCorrectionTools.NoCorrection
BLAS.set_num_threads(1)

@model function affine_slice(source_mean, source_cov, transition, sensor, noise, y, observed)
    source ~ MvNormal(mean = source_mean, covariance = source_cov)
    joint := *(transition, source) where {meta = NoCorrection()}
    if observed
        reading := *(sensor, joint) where {meta = NoCorrection()}
        y ~ MvNormal(mean = reading, covariance = noise)
    else
        joint ~ Uninformative()
    end
end

function slice!(graphs, m, P, transition, H, R, y; score=false)
    prior = (source_mean=m, source_cov=P, transition=transition)
    data = isnothing(y) ? prior : (; prior..., sensor=H, noise=R, y=y)
    key = (size(transition), isnothing(y) ? 0 : length(y))
    graph = get!(graphs, key) do
        model = isnothing(y) ? affine_slice(; observed=false, sensor=nothing, noise=nothing, y=nothing) :
                               affine_slice(; observed=true)
        stream = Rocket.Subject(typeof(data))
        engine = infer(model=model, datastream=stream, autoupdates=RxInfer.EmptyAutoUpdateSpecification,
                       returnvars=(:joint,), keephistory=1)
        (; stream, engine)
    end
    try
        Rocket.next!(graph.stream, data)
        mu, cov = mean_cov(last(graph.engine.history[:joint]))
        all(isfinite, mu) || error("nonfinite native Gaussian mean")
        cov = checked_covariance(cov)
        # Score the forward BP message before evidence. The sensor-node
        # marginal in the conditioned graph already contains the observation.
        nll = if score
            latent = @call_rule typeof(*)(:out, Marginalisation) (
                m_A=PointMass(H * transition), m_in=MvNormalMeanCovariance(m, P), meta=NoCorrection())
            predicted = @call_rule MvNormalMeanCovariance(:out, Marginalisation) (
                m_μ=latent, m_Σ=PointMass(R))
            -logpdf(predicted, y)
        else
            0.
        end
        isfinite(nll) || error("invalid predictive density")
        return Vector(mu), Matrix(cov), nll
    catch
        delete!(graphs, key)
        rethrow()
    end
end

vector(io, n) = read!(io, Vector{Float64}(undef, n))
matrix(io, n, k) = read!(io, Matrix{Float64}(undef, n, k))

function checked_covariance(cov; allow_singular=false)
    all(isfinite, cov) || error("nonfinite covariance")
    tol = 16 * size(cov, 1) * eps(Float64) * maximum(abs, cov)
    maximum(abs, cov - cov') <= tol || error("asymmetric covariance")
    symmetric = Symmetric(cov / 2 + cov' / 2)
    if !isposdef(symmetric) && !(allow_singular && eigmin(symmetric) >= 0)
        error("invalid Gaussian covariance")
    end
    return Matrix(symmetric)
end

function block_covariance(blocks...)
    total = sum(b -> size(b, 1), blocks)
    result = zeros(total, total)
    offset = 0
    for b in blocks
        r = offset+1:offset+size(b, 1)
        result[r, r] = b
        offset = last(r)
    end
    return result
end

function update!(io, out, graphs, n, k)
    m, P = vector(io, n), checked_covariance(matrix(io, n, n); allow_singular=true)
    A, b = matrix(io, n, n), vector(io, n)
    Q = checked_covariance(matrix(io, n, n); allow_singular=true)
    H = matrix(io, k, n)
    R = k == 0 ? nothing : checked_covariance(matrix(io, k, k))
    y = k == 0 ? nothing : vector(io, k)
    # Source = (previous state, process innovation); the innovation has mean b.
    mu, cov, nll = slice!(graphs, [m; b], block_covariance(P, Q), [A I(n)], H, R, y; score=k>0)
    write(out, mu, cov, nll)
end

function plan!(io, out, graphs, n, k, horizon)
    m, P = vector(io, n), checked_covariance(matrix(io, n, n); allow_singular=true)
    C = matrix(io, k, n)
    R, U = checked_covariance(matrix(io, k, k)), checked_covariance(matrix(io, 2, 2))
    H = [zeros(k, 2) C]
    for j in 1:horizon
        A, b, B = matrix(io, n, n), vector(io, n), matrix(io, n, 2)
        Q = checked_covariance(matrix(io, n, n); allow_singular=true)
        u, target = vector(io, 2), vector(io, k)
        # Independent inputs are the carried marginal, this action prior and process noise.
        # The output always retains (first action, next state), including their cross-covariance.
        transition = if j == 1
            [zeros(2, n) I(2) zeros(2, n); A B I(n)]
        else
            [I(2) zeros(2, n+2+n); zeros(n, 2) A B I(n)]
        end
        m, P = slice!(graphs, [m; u; b], block_covariance(P, U, Q), transition, H, R, target)
    end
    write(out, m[1:2])
end

function serve()
    VERSION == v"1.10.11" && pkgversion(RxInfer) == v"3.6.0" || error("unexpected Julia or RxInfer version")
    ENDIAN_BOM == 0x04030201 || error("little-endian worker required")
    graphs = Dict{Tuple{Tuple{Int, Int}, Int}, Any}()
    write(stdout, "RXE1")
    flush(stdout)
    while !eof(stdin)
        size = read(stdin, UInt32)
        io = IOBuffer(read(stdin, size))
        out = IOBuffer()
        code = UInt8(0)
        try
            op = read(io, UInt8)
            n, k, horizon = Int.(read!(io, Vector{Int32}(undef, 3)))
            if op == 1
                update!(io, out, graphs, n, k)
            elseif op == 2
                plan!(io, out, graphs, n, k, horizon)
            else
                error("unknown request")
            end
            eof(io) || error("trailing numerical input")
        catch err
            code = UInt8(1)
            out = IOBuffer()
            print(out, sprint(showerror, err))
        end
        response = take!(out)
        write(stdout, code, UInt32(length(response)), response)
        flush(stdout)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && serve()
