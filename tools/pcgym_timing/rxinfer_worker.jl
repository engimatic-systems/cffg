# Persistent JSON-lines worker for the fixed P4/H15 reactor fixture.
using RxInfer
using LinearAlgebra
using JSON

BLAS.set_num_threads(1)
const NoCorrection = ReactiveMP.MatrixCorrectionTools.NoCorrection

asvector(a) = Float64.(a)
asmatrix(a) = permutedims(reduce(hcat, asvector.(a)))
rows(a) = [collect(r) for r in eachrow(a)]

struct Reactor
    lo::Vector{Float64}
    hi::Vector{Float64}
    ulo::Float64
    uhi::Float64
    dt::Float64
    substeps::Int
    tracked::Int
    theta::Vector{Float64}
end

# Match JAX's half derivative at an exact clip boundary.
function clip_input(x, lo, hi)
    x == lo && return (x + lo) / 2
    x == hi && return (x + hi) / 2
    return clamp(x, lo, hi)
end

function (f::Reactor)(s)
    span = f.hi - f.lo
    ca, t = clip_input.((s[1:2] .+ 1) ./ 2 .* span .+ f.lo, f.lo .- 2span, f.hi .+ 2span)
    ca = ca == 0 ? ca/2 : max(ca, 0)
    u = clip_input((s[4] + 1) / 2 * (f.uhi - f.ulo) + f.ulo, f.ulo - 20, f.uhi + 20)
    k, er, ua, dh = exp(f.theta[1]), f.theta[2] * 1e4, exp(f.theta[3]), exp(f.theta[4])

    function derivative(c, temp)
        reaction = k * exp(-er * (1 / temp - 1 / 330)) * c
        return 1 - c - reaction, 350 - temp + dh * reaction / 239 + ua * (u - temp) / 23900
    end

    h = f.dt / f.substeps
    for _ in 1:f.substeps
        c1, t1 = derivative(ca, t)
        c2, t2 = derivative(ca + h*c1/2, t + h*t1/2)
        c3, t3 = derivative(ca + h*c2/2, t + h*t2/2)
        c4, t4 = derivative(ca + h*c3, t + h*t3)
        ca += h/6 * (c1 + 2c2 + 2c3 + c4)
        t  += h/6 * (t1 + 2t2 + 2t3 + t4)
    end
    x = 2 .* ([ca, t] - f.lo) ./ span .- 1
    x[f.tracked] += s[3]
    return [x; s[3:4]]
end

struct Affine
    A::Matrix{Float64}
    b::Vector{Float64}
end
(f::Affine)(s) = f.A * s + f.b

# Each marginal ends in the four current coordinates. From s5 onward,
# retain s4, then also s5's action. Future data thus smooth these variables
# through their native joint marginal without a separate backward readout.
function advance(previous, innovation, transition, site)
    new_state = transition(previous[end-3:end]) + innovation
    site < 5 && return new_state
    site == 5 && return [previous; new_state]
    site == 6 && return [previous[1:4]; previous[8]; new_state]
    return [previous[1:5]; new_state]
end

@model function reactor_slice(previous_mean, previous_cov, process_cov, transition, site, observed, G, R, y)
    if site == 0
        joint ~ MvNormal(mean = previous_mean, covariance = previous_cov)
    else
        previous ~ MvNormal(mean = previous_mean, covariance = previous_cov)
        innovation ~ MvNormal(mean = zeros(4), covariance = process_cov)
        joint := advance(previous, innovation, transition, site) where {meta = Linearization()}
    end

    if observed
        # The observation has exactly zero precision in unobserved directions.
        # The proper forward prior makes the posterior precision invertible.
        projected := *(G, joint) where {meta = NoCorrection()}
        y ~ MvNormal(mean = projected, covariance = R)
    else
        joint ~ Uninformative()
    end
end

function setup(request)
    a, c = request["arrays"], request["config"]
    c["nd"] == 1 && length(c["tracked"]) == 1 || error("expected one disturbance")
    transition = if isnothing(request["affine"])
        Reactor(asvector(c["x_lo"]), asvector(c["x_hi"]), c["u_lo"], c["u_hi"],
                c["dt"], c["substeps"], only(c["tracked"])+1, asvector(a["theta"]))
    else
        A, b = request["affine"]
        Affine(asmatrix(A), asvector(b))
    end
    sites = map(0:19) do i
        kind = i < 4 ? "past" : i == 4 ? "now" : "goal"
        G, R, z = asmatrix(a["G_"*kind]), asmatrix(a["R_"*kind]), asvector(a["z_"*kind])
        prefix = i <= 4 ? 0 : i == 5 ? 4 : 5
        (G = hcat(zeros(size(G, 1), prefix), G), R = R, z = z)
    end
    return (; transition, sites, m0=asvector(a["m0"]), P0=asmatrix(a["P0"]),
             Q=asmatrix(a["Q"]), Qp=asmatrix(a["Qp"]), graphs=Dict{Tuple{Int,Bool},Any}())
end

# Every event replaces the entire Gaussian prior and, for observed slices,
# the evidence. Empty autoupdates prevent any posterior feedback between events.
function slice_graph(fixture, site, observed, input)
    emission = fixture.sites[site+1]
    optional = observed ? (;) : (y=nothing,)
    model = reactor_slice(; process_cov=site <= 4 ? fixture.Q : fixture.Qp,
                            transition=fixture.transition, site, observed, G=emission.G,
                            R=emission.R, optional...)
    stream = Rocket.Subject(typeof(input))
    engine = infer(model=model, datastream=stream, autoupdates=RxInfer.EmptyAutoUpdateSpecification,
                   returnvars=(:joint,), keephistory=1)
    return (; stream, engine)
end

function posterior(graph, input)
    Rocket.next!(graph.stream, input)
    return mean_cov(last(graph.engine.history[:joint]))
end

function solve(fixture, data, details)
    length(data) == 20 || error("expected 20 observation rows")
    m, P = fixture.m0, fixture.P0
    smallest_pivot = Inf
    approximation = "RxInfer Linearization (forward joint Gaussian message per slice)"
    for i in 0:19
        site = fixture.sites[i+1]
        y = data[i+1]
        # Sites 1..3 share the past transition and likelihood. Sites 7..19
        # share the same augmented transition and goal observation matrix.
        kind = i in 1:3 ? 1 : min(i, 7)
        key = (kind, !isnothing(y))
        try
            if !isnothing(y)
                y isa Vector && length(y) == length(site.z) || error("invalid observation dimension")
            end
            prior = (previous_mean=m, previous_cov=P)
            input = isnothing(y) ? prior : (; prior..., y=asvector(y)-site.z)
            graph = get!(fixture.graphs, key) do
                slice_graph(fixture, kind, !isnothing(y), input)
            end
            m, P = posterior(graph, input)
            all(isfinite, m) && all(isfinite, P) || error("non-finite joint posterior")
            L = cholesky(Symmetric(P)).L
            smallest_pivot = min(smallest_pivot, minimum(diag(L)))
        catch err
            delete!(fixture.graphs, key)
            return (; status="error", action=nothing, approximation, failed_site=i,
                      slices_completed=i, error=sprint(showerror, err))
        end
    end
    answer = (; status="ok", action=m[4], approximation, slices_completed=20, passes_per_slice=1,
                observed_rows=count(!isnothing, data), min_cholesky_diagonal=smallest_pivot)
    return details ? (; answer..., mean=m[1:4], cov=rows(P[1:4, 1:4]), next_action_cov=P[4,5]) : answer
end

function serve()
    fixture = nothing
    for line in eachline(stdin)
        answer = try
            request = JSON.parse(line)
            if request["op"] == "init"
                fixture = setup(request)
                (; status="ready", rxinfer_version=string(pkgversion(RxInfer)), julia_version=string(VERSION),
                   julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads())
            elseif request["op"] == "solve"
                solve(fixture, request["data"], request["details"])
            else
                error("unrecognized request")
            end
        catch err
            showerror(stderr, err, catch_backtrace())
            println(stderr)
            (; status="error", action=nothing, error=sprint(showerror, err))
        end
        println(JSON.json(answer))
        flush(stdout)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && serve()
