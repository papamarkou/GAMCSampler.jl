#!/bin/bash

JULIABIN=julia

samplers=(AM MALA SMMALA MAMALA)

for sampler in ${samplers[@]}
do
  echo "Generating $sampler table..."
  $JULIABIN "$HOME/.julia/v0.5/MAMALASampler/examples/t_distribution/src/$sampler/table.jl"
done
