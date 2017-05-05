library(data.table)
library(stringr)

cmd_args <- commandArgs()
CURRENTDIR <- dirname(regmatches(cmd_args, regexpr("(?<=^--file=).+", cmd_args, perl=TRUE)))
ROOTDIR <- dirname(dirname(dirname(CURRENTDIR)))
OUTDIR <- file.path(ROOTDIR, "output", "one_planet")

# OUTDIR <- "../../../output/one_planet"

SUBOUTDIR <- "MALA"

true_param <- 3.04452

nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

ci <- 6
pi <- 2

chains <- t(fread(
  file.path(OUTDIR, SUBOUTDIR, paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE
))

chainmean = mean(chains[, pi])

pdf(file=file.path(OUTDIR, SUBOUTDIR, "rv_one_planet_mala_traceplot.pdf"), width=10, height=6)

plot(
  1:npostburnin,
  chains[, pi],
  type="l",
  ylim=c(2.95, 3.15),
  col="steelblue2",
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7,
  yaxt="n"
)

axis(
  2,
  at=seq(2.9, 3.15, by=0.05),
  labels=seq(2.9, 3.15, by=0.05),
  cex.axis=1.8,
  las=1
)

# abline(h=true_param, lwd=2, col="black")

dev.off()
