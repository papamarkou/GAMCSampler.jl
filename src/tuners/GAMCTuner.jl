type GAMCMCTune <: MCTunerState
  smmalatune::BasicMCTune
  amtune::BasicMCTune
  totaltune::BasicMCTune
  smmalafrequency::Real
  amfrequency::Real
end

GAMCMCTune(smmalatune::BasicMCTune, amtune::BasicMCTune, totaltune::BasicMCTune) =
  GAMCMCTune(smmalatune, amtune, totaltune, NaN, NaN)

GAMCMCTune(smmalastep::Real=1., malastep::Real=1.) =
  GAMCMCTune(BasicMCTune(smmalastep), BasicMCTune(NaN), BasicMCTune(1.), NaN, NaN)

immutable GAMCMCTuner <: MCTuner
  smmalatuner::VanillaMCTuner
  amtuner::VanillaMCTuner
  totaltuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
end

Base.show(io::IO, tuner::GAMCMCTuner) = print(io, "GAMCMCTuner")

Base.show(io::IO, ::MIME"text/plain", tuner::GAMCMCTuner) = show(io, tuner)
