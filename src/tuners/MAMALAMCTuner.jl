type MAMALAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  malatune::BasicMCTune
  totaltune::BasicMCTune
  smmalafrequency::Real
  malafrequency::Real
end

MAMALAMCTune(smmalatune::BasicMCTune, malatune::BasicMCTune, totaltune::BasicMCTune) =
  MAMALAMCTune(smmalatune, malatune, totaltune, NaN, NaN)

MAMALAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  MAMALAMCTune(BasicMCTune(smmalastep), BasicMCTune(malastep), BasicMCTune(1.), NaN, NaN)

immutable MAMALAMCTuner <: MCTuner
  smmalatuner::VanillaMCTuner
  malatuner::VanillaMCTuner
  totaltuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
end

Base.show(io::IO, tuner::MAMALAMCTuner) = print(io, "MAMALAMCTuner")

Base.show(io::IO, ::MIME"text/plain", tuner::MAMALAMCTuner) = show(io, tuner)
