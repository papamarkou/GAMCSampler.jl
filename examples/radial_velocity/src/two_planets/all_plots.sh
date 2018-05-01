#!/bin/bash

RBIN=Rscript

$RBIN "$HOME/.julia/v0.6/GAMCSampler/examples/radial_velocity/src/two_planets/acfplot.r"
$RBIN "$HOME/.julia/v0.6/GAMCSampler/examples/radial_velocity/src/two_planets/meanplot.r"
$HOME/.julia/v0.6/GAMCSampler/examples/radial_velocity/src/two_planets/traceplots.sh
