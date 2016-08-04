# See
# http://what-when-how.com/artificial-intelligence/a-comparison-of-cooling-schedules-for-simulated-annealing-artificial-intelligence/

exp_cool <- function(i, ntot, a, b) {
  (1-b)*exp(-a*i/ntot)+b
}

pow_cool <- function(i, ntot, a, b) {
  (1-b)*(a^(i/ntot))+b
}

lin_cool <- function(i, ntot, a, b) {
  (1-b)*(1/(1+a*i))+b
}

quad_cool <- function(i, ntot, a, b) {
  (1-b)*(1/(1+a*(i^2)))+b
}

log_cool <- function(i, ntot, a, b) {
  (1-b)*(1/(1+a*log(1+i)))+b
}

ntot <- 10000
x <- seq(0, ntot, by=1)

plot(
  x,
  pow_cool(x, ntot, 0.001, 0),
  type="l"
)

lines(
  x,
  pow_cool(x, ntot, 1e-10, 0),
  col="red"
)

lines(
  x,
  lin_cool(x, ntot, 0.01, 0.01),
  col="green"
)

lines(
  x,
  quad_cool(x, ntot, 0.000002, 0),
  col="blue"
)

lines(
  x,
  log_cool(x, ntot, 3, 0),
  col="purple"
)

# plot(
#   x,
#   0.1*cos(100*x*pi/ntot)+0.2,
#   type="l",
#   ylim=c(0, 1)
# )
# 
# plot(
#   x,
#   ((exp(-5*x/ntot)+0.05*cos(100*x*pi/ntot))+0.05)/1.1,
#   type="l"
# )
# 
# plot(
#   seq(-1, 1, by=0.01),
#   seq(-1, 1, by=0.01)^2,
#   type="l"
# )

# 1) Lift the map
# 2) Lift-decay the map
# 3) Discussion: adaptive
# 4) Discussion: partial (parts) of metric
# 5) Discussion: scale with data and parameters
