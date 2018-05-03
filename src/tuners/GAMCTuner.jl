mutable struct GAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  amtune::BasicMCTune
  totaltune::BasicMCTune
  smmalafrequency::Real
  amfrequency::Real
end

GAMCTune(smmalatune::BasicMCTune, amtune::BasicMCTune, totaltune::BasicMCTune) =
  GAMCTune(smmalatune, amtune, totaltune, NaN, NaN)

GAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  GAMCTune(BasicMCTune(smmalastep), BasicMCTune(NaN), BasicMCTune(1.), NaN, NaN)

struct GAMCTuner <: MCTuner
  smmalatuner::VanillaMCTuner
  amtuner::VanillaMCTuner
  totaltuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
  verbose::Bool
end

GAMCTuner(smmalatuner::VanillaMCTuner, amtuner::VanillaMCTuner, totaltuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}) =
  GAMCTuner(smmalatuner, amtuner, totaltuner, (totaltuner.verbose || smmalatuner.verbose || amtuner.verbose) ? true : false)

show(io::IO, tuner::GAMCTuner) = print(io, "GAMCTuner")
