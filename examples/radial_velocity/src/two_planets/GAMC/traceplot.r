library(data.table)
library(stringr)

cmd_args <- commandArgs()
CURRENTDIR <- dirname(regmatches(cmd_args, regexpr("(?<=^--file=).+", cmd_args, perl=TRUE)))
ROOTDIR <- dirname(dirname(dirname(CURRENTDIR)))
OUTDIR <- file.path(ROOTDIR, "output", "two_planets")

# OUTDIR <- "../../output/two_planets"

SUBOUTDIR <- "GAMC"

true_param <- 3.43399

nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

ci <- 4
pi <- 2

chains <- t(fread(
  file.path(OUTDIR, SUBOUTDIR, paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE
))

chainmean <- mean(chains[, pi])

pdf(file=file.path(OUTDIR, SUBOUTDIR, "rv_two_planets_gamc_traceplot.pdf"), width=10, height=6)

plot(
  1:npostburnin,
  chains[, pi],
  type="l",
  ylim=c(3.3, 3.6),
  col="steelblue2",
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7,
  yaxt="n"
)

axis(
  2,
  at=seq(3.3, 3.6, by=0.1),
  labels=seq(3.3, 3.6, by=0.1),
  cex.axis=1.8,
  las=1
)

# abline(h=true_param, lwd=2, col="black")

dev.off()
