esbi <- function(i, a){(1-exp(-a))/(1-exp(-a/i))}
esbia <- function(i, a){(1-exp(-a))/(i*(1-exp(-a/i)))}
esbib <- function(i, a){1-esbia(i, a)}

n <- 1000
a <- 1

x <- seq(0, n, by=1)
plot(x, x, type="l", ylim=c(0, n))
lines(x, log(x), col="green")
lines(x, esbi(x, 1), col="red")
lines(x, esbi(x, 10), col="blue")

x <- seq(0, n, by=1)
plot(x, x*x, type="l", ylim=c(0, n*n))
lines(x, x*log(x), col="green")
lines(x, x*esbi(x, 1), col="red")
lines(x, x*esbi(x, 10), col="blue")

x <- seq(0, n, by=1)
plot(x, x*x, type="l", ylim=c(0, n*n))
lines(x, x*log(x), col="green")
lines(x, x^2*esbia(x, 1), col="red")
lines(x, x^2*esbia(x, 10), col="blue")

x <- seq(0, n, by=1)
plot(x, log(x)*x, type="l", ylim=c(0, log(n)*(n-log(n))))
lines(x, rep(0, length(x)), col="green")
lines(x, x*log(x)*esbib(x, 1), col="red")
lines(x, x*log(x)*esbib(x, 10), col="blue")

x <- seq(0, n, by=1)
plot(x, x*x+log(x)*x, type="l", ylim=c(0, n*n+log(n)*(n-log(n))))
lines(x, x*log(x)+log(x)*log(x), col="green")
lines(x, x^2*esbia(x, 1)+x*log(x)*esbib(x, 1), col="red")
lines(x, x^2*esbia(x, 10)+x*log(x)*esbib(x, 10), col="blue")

x <- seq(0, n, by=1)
plot(x, esbi(x, a)/x, type="l")

x <- seq(0, n, by=1)
plot(x, exp(-1*x/n), type="l", ylim=c(0, 1))
lines(x, exp(-10*x/n), type="l", col="red")
lines(x, exp(-15*x/n), type="l", col="green")
