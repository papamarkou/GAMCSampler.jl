#!/bin/bash

JULIABIN=julia

samplers=(AM MALA SMMALA MAMALA)

for sampler in ${samplers[@]}
do
  echo "Generating $sampler table line..."
  $JULIABIN "$HOME/.julia/v0.5/MAMALASampler/examples/logistic_regression/src/$sampler/table.jl"
done
