# reminder, to type H*, do H\^+
cd("/Users/vbd402/Documents/Projects/mixedbridge/")
outdir="/Users/vbd402/Documents/Projects/mixedbridge/examples/cell_model_runs/"

using Bridge, StaticArrays, Distributions
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using Plots

T = 4.0
dt = 1/500
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood

νHparam = false
generate_data = false

# settings in case of νH - parametrisation
ϵ = 0#10^(-3)
Σdiagel = 10^(-10)

# settings sampler
iterations = 2*10^3
skip_it = 25 # 1000
subsamples = 0:skip_it:iterations

L = @SMatrix [1.0 0.0; 0.0 1.0] # condition on full observation

ρ = 0.98  # pCN parameter

v = ℝ{2}(2.0, -0.1)   # point to condition on at time T (this one is common from forward simulation)
# v = ℝ{2}(1.0, -0.1)     # probably rare event

m, d = size(L)
Σ = SMatrix{m,m}(Σdiagel*I)

# specify target process 
struct CellDiffusion <: ContinuousTimeProcess{ℝ{2}}
    α::Float64
    σ::Float64
end

u(x, α) = x^4/(α + x^4)
Bridge.b(t, x, P::CellDiffusion) = ℝ{2}( u(x[1],P.α) + 1.0 - u(x[2],P.α) -x[1], u(x[2],P.α) + 1.0 - u(x[1],P.α) -x[2])
Bridge.σ(t, x, P::CellDiffusion) = @SMatrix [P.σ 0.0; 0.0 P.σ]
Bridge.constdiff(::CellDiffusion) = true

P = CellDiffusion(1.0/16, 0.1)
x0 = ℝ{2}(.1, -0.1)


# specify auxiliary process (just drop nonlinearities)
struct CellDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    α::Float64
    σ::Float64
end

Random.seed!(42)
Bridge.B(t, P::CellDiffusionAux) = @SMatrix [-1.0 0.0; 0.0 -1.0]
Bridge.β(t, P::CellDiffusionAux) = ℝ{2}(1.0, 1.0)

Bridge.σ(t, x, P::CellDiffusionAux) = @SMatrix [P.σ 0.0; 0.0 P.σ]
Bridge.σ(t, P::CellDiffusionAux) = @SMatrix [P.σ 0.0; 0.0 P.σ]
Bridge.constdiff(::CellDiffusionAux) = true
Bridge.b(t, x, P::CellDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::CellDiffusionAux) = Bridge.σ(t,0,P) * Bridge.σ(t, 0, P)'

Pt = CellDiffusionAux(P.α, P.σ)

# Solve Backward Recursion
Po = νHparam ? Bridge.PartialBridgeνH(tt, P, Pt, L, ℝ{m}(v),ϵ, Σ) : Bridge.PartialBridge(tt, P, Pt, L, ℝ{m}(v), Σ)


####################### MH algorithm ###################
W = sample(tt, Wiener{ℝ{2}}())
X = solve(Euler(), x0, W, P)
Xo = copy(X)
solve!(Euler(), Xo, x0, W, Po)
solve!(Euler(), X, x0, W, Po)

ll = llikelihood(Bridge.LeftRule(), X, Po,skip=sk)

# further initialisation
Wo = copy(W)
W2 = copy(W)
XX = Any[]
if 0 in subsamples
    push!(XX, copy(X))
end

acc = 0
lls = [ll]

for iter in 1:iterations
    # Proposal
    sample!(W2, Wiener{ℝ{2}}())
    
    Wo.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy
    solve!(Euler(), Xo, x0, Wo, Po)

    llo = llikelihood(Bridge.LeftRule(), Xo, Po,skip=sk)
    print("ll $ll $llo, diff_ll: ",round(llo-ll,digits=3))

    if log(rand()) <= llo - ll
        X.yy .= Xo.yy
        W.yy .= Wo.yy
        global ll = llo
        print("✓")
        global acc +=1
    end
    push!(lls, ll)
    println()
    if iter in subsamples
        push!(XX, copy(X))
    end
end

@info "Done."*"\x7"^6

# plot loglikelihood over iterations
plot(lls)

# write mcmc iterates to csv file

fn = outdir*"iterates.csv"
iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:2, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
its = hcat(iterates...)'
outdf = DataFrame(iteration=its[:,1], time=its[:,2], component=its[:,3], value=its[:,4])
CSV.write(fn, outdf)

ave_acc_perc = 100*round(acc/iterations,digits=2)

# write info to txt file
fn = outdir*"info.txt"
f = open(fn,"w")
write(f, "Number of iterations: ",string(iterations),"\n")
write(f, "Skip every ",string(skip_it)," iterations, when saving to csv","\n\n")
write(f, "Starting point: ",string(x0),"\n")
write(f, "End time T: ", string(T),"\n")
write(f, "Endpoint v: ",string(v),"\n")
write(f, "Noise Sigma: ",string(Σ),"\n")
write(f, "L: ",string(L),"\n\n")
write(f,"Mesh width: ",string(dt),"\n")
write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
write(f, "Average acceptance percentage: ",string(ave_acc_perc),"\n\n")
write(f, "Backward type parametrisation in terms of nu and H? ",string(νHparam),"\n")
close(f)


println("Average acceptance percentage: ",ave_acc_perc,"\n")
println("Parametrisation of nu and H? ", νHparam)


# plots of forward simulation of the process
W = sample(tt, Wiener{ℝ{2}}()); X = solve(Euler(), x0, W, P)
p1 = plot(X.tt, first.(X.yy)); p2 = plot(X.tt, last.(X.yy)); plot(p1, p2)