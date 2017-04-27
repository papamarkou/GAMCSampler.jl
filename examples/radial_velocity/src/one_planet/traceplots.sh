#!/bin/bash

RBIN=Rscript

samplers=(AM MALA SMMALA MAMALA)

for sampler in ${samplers[@]}
do
  echo "Gennerating $sampler traceplot..."
  $RBIN "$HOME/.julia/v0.5/MAMALASampler/examples/radial_velocity/src/one_planet/$sampler/traceplot.r"
done
