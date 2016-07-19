using Gadfly
using PGUManifoldMC

x = 0:1:(10000-1)
nx =length(x)

myplot = plot(
  layer(
    x=collect(x),
    y=[exp_decay(i, nx, 10) for i in x],
    Geom.line
  ),
  layer(
    x=collect(x),
    y=[exp_decay(i, nx, 5) for i in x],
    Geom.line,
    Theme(default_color=colorant"red")
  ),
  layer(
    x=collect(x),
    y=[exp_decay(i, nx, 15) for i in x],
    Geom.line,
    Theme(default_color=colorant"green")
  )
)

draw(PDF("bernoulli_exp_schedule.pdf", 4inch, 3inch), myplot)
