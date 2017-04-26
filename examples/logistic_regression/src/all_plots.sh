#!/bin/bash

RBIN=Rscript

$RBIN "$HOME/.julia/v0.5/MAMALASampler/examples/logistic_regression/src/acfplot.r"
$RBIN "$HOME/.julia/v0.5/MAMALASampler/examples/logistic_regression/src/meanplot.r"
$HOME/.julia/v0.5/MAMALASampler/examples/logistic_regression/src/traceplots.sh
