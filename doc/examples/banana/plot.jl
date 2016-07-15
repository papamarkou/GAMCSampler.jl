using Distributions
using Gadfly

θ = [0., 0.]
pvar = [10., 1.]
# θ = [0., 0.]

plot(
  # z=(x, y) -> pdf(MvNormal(θ, [2., 2.]), [x, y+abs2(x)]),
  z=(x, y) -> pdf(MvNormal(θ, pvar), [x, y+0.03*(abs2(x)-pvar[1])]),
  # z=(x, y) -> pdf(MvNormal(θ, [10., 1.]), [x, y+0.03*abs2(x)-0.3]),
  x=linspace(-25., 25., 100),
  y=linspace(-20., 20., 100),
  Geom.contour
)
