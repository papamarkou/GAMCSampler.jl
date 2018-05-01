#!/bin/bash

RBIN=Rscript

samplers=(AM MALA SMMALA GAMC)

for sampler in ${samplers[@]}
do
  echo "Gennerating $sampler traceplot..."
  $RBIN "$HOME/.julia/v0.6/GAMCSampler/examples/radial_velocity/src/one_planet/$sampler/traceplot.r"
done
