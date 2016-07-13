# Note that μ and σ2 correspond to the sample mean and IID variance of an independent chain
# In particular, σ2 is not the Monte Carlo variance of the chain, it is the ordinary variance estimator for IID samples
# If the true μ and σ2 are known, you may use them instead of simulating an independent chain

Base.cor(v::Vector{Float64}, s::Int, μ::Float64, σ2::Float64, vlen::Int) =
  dot(v[1:(vlen-s)]-μ, v[(s+1):vlen]-μ)/(σ2*(vlen-s))

function ess(v::Vector{Float64}, μ::Float64, σ2::Float64, vlen::Int)
  sumterm = 0.
  s = 1
  ρ = cor(v, s, μ, σ2, vlen)

  while ρ >= 0.05
    sumterm += (1-s/vlen)*ρ
    s += 1
    ρ = cor(v, s, μ, σ2, vlen)
  end

  vlen/(1+2*sumterm)
end

function nutsess(v::Vector{Float64}, μ::Float64, σ2::Float64, vlen::Int)
  sumterm = 0.
  s = 1
  ρ = cor(v, s, μ, σ2, vlen)

  while ρ >= 0.05
    sumterm += (1-s/vlen)*ρ
    println(ρ)
    s += 1
    ρ = cor(v, s, μ, σ2, vlen)
  end

  println(s)
  println(sumterm)
  vlen/(1+2*sumterm)
end
