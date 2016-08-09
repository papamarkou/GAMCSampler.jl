function codegen(::Type{Val{:iterate}}, ::Type{PSMMALA}, job::BasicMCJob)
  result::Expr
  burninbody = []
  malabody = []
  smmalapasttensorbody = []
  smmalabody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Multivariate
    error("Only multivariate parameter states allowed in PSMMALA code generation")
  end

  push!(body, :(_job.sstate.count += 1))

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.malatuner.verbose ||
    isa(job.tuner.smmalatuner, AcceptanceRateMCTuner) ||
    isa(job.tuner.malatuner, AcceptanceRateMCTuner)
  )
    push!(body, :(_job.sstate.tune.totaltune.proposed += 1))
  end

  push!(body, :(_job.sampler.update!(_job.sstate, _job.pstate, _job.sstate.count, _job.range.nsteps)))

  push!(smmalabody, :(_job.sstate.updatetensorcount += 1))

  if job.tuner.smmalatuner.verbose || isa(job.tuner.smmalatuner, AcceptanceRateMCTuner)
    push!(smmalabody, :(_job.sstate.tune.smmalatune.proposed += 1))
  end

  push!(smmalapasttensorbody, :(_job.parameter.uptotensorlogtarget!(_job.pstate)))

  if job.sampler.transform != nothing
    push!(smmalapasttensorbody, :(_job.pstate.tensorlogtarget = _job.sampler.transform(_job.pstate.tensorlogtarget)))
  end

  push!(smmalapasttensorbody, :(_job.sstate.oldinvtensor = inv(_job.pstate.tensorlogtarget)))

  push!(smmalapasttensorbody, :(_job.sstate.oldfirstterm = _job.sstate.oldinvtensor*_job.pstate.gradlogtarget))

  push!(smmalapasttensorbody, :(_job.sstate.cholinvtensor = chol(_job.sstate.oldinvtensor, Val{:L})))

  push!(smmalabody, Expr(:if, :(!_job.sstate.pastupdatetensor), Expr(:block, smmalapasttensorbody...)))

  push!(smmalabody, :(_job.sstate.μ = _job.pstate.value+0.5*_job.sampler.smmalastep*_job.sstate.oldfirstterm))

  push!(
    smmalabody,
    :(
      _job.sstate.pstate.value =
        _job.sstate.μ+_job.sstate.sqrtsmmalastep*_job.sstate.cholinvtensor*randn(_job.pstate.size)
    )
  )

  push!(smmalabody, :(_job.parameter.uptotensorlogtarget!(_job.sstate.pstate)))

  if job.sampler.transform != nothing
    push!(smmalabody, :(_job.sstate.pstate.tensorlogtarget = _job.sampler.transform(_job.sstate.pstate.tensorlogtarget)))
  end

  push!(smmalabody, :(_job.sstate.newinvtensor = inv(_job.sstate.pstate.tensorlogtarget)))

  push!(smmalabody, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  push!(
    smmalabody,
    :(
      _job.sstate.ratio += (
        0.5*(
          logdet(_job.sampler.smmalastep*_job.sstate.oldinvtensor)
          +dot(
            _job.sstate.pstate.value-_job.sstate.μ,
            _job.pstate.tensorlogtarget*(_job.sstate.pstate.value-_job.sstate.μ)
          )/_job.sampler.smmalastep
        )
      )
    )
  )

  push!(smmalabody, :(_job.sstate.newfirstterm = _job.sstate.newinvtensor*_job.sstate.pstate.gradlogtarget))

  push!(smmalabody, :(_job.sstate.μ = _job.sstate.pstate.value+0.5*_job.sampler.smmalastep*_job.sstate.newfirstterm))

  push!(smmalabody, :(
      _job.sstate.ratio -= (
        0.5*(
          logdet(_job.sampler.smmalastep*_job.sstate.newinvtensor)
          +dot(
            _job.pstate.value-_job.sstate.μ,
            _job.sstate.pstate.tensorlogtarget*(_job.pstate.value-_job.sstate.μ)
          )/_job.sampler.smmalastep
        )
      )
    )
  )

  update = []
  noupdate = []

  push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))

  push!(update, :(_job.pstate.gradlogtarget = copy(_job.sstate.pstate.gradlogtarget)))

  if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
    push!(update, :(_job.pstate.gradloglikelihood = copy(_job.sstate.pstate.gradloglikelihood)))
  end

  if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
    push!(update, :(_job.pstate.gradlogprior = copy(_job.sstate.pstate.gradlogprior)))
  end

  push!(update, :(_job.pstate.tensorlogtarget = copy(_job.sstate.pstate.tensorlogtarget)))

  if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
    push!(update, :(_job.pstate.tensorloglikelihood = copy(_job.sstate.pstate.tensorloglikelihood)))
  end

  if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
    push!(update, :(_job.pstate.tensorlogprior = copy(_job.sstate.pstate.tensorlogprior)))
  end

  push!(update, :(_job.sstate.oldinvtensor = copy(_job.sstate.newinvtensor)))

  push!(update, :(_job.sstate.cholinvtensor = chol(_job.sstate.newinvtensor, Val{:L})))

  push!(update, :(_job.sstate.oldfirstterm = copy(_job.sstate.newfirstterm)))

  push!(update, :(_job.pstate.logtarget = _job.sstate.pstate.logtarget))

  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_job.pstate.loglikelihood = _job.sstate.pstate.loglikelihood))
  end

  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_job.pstate.logprior = _job.sstate.pstate.logprior))
  end

  if in(:accept, job.outopts[:diagnostics])
    push!(update, :(_job.pstate.diagnosticvalues[1] = true))
    push!(noupdate, :(_job.pstate.diagnosticvalues[1] = false))
  end

  if job.tuner.totaltuner.verbose
    push!(update, :(_job.sstate.tune.totaltune.accepted += 1))
  end

  if job.tuner.smmalatuner.verbose || isa(job.tuner.smmalatuner, AcceptanceRateMCTuner)
    push!(update, :(_job.sstate.tune.smmalatune.accepted += 1))
  end

  push!(
    smmalabody,
    Expr(
      :if,
      :(_job.sstate.ratio > 0 || (_job.sstate.ratio > log(rand()))),
      Expr(:block, update...),
      Expr(:block, noupdate...)
    )
  )

  if job.tuner.malatuner.verbose || isa(job.tuner.malatuner, AcceptanceRateMCTuner)
    push!(malabody, :(_job.sstate.tune.malatune.proposed += 1))
  end

  if job.sampler.identitymala
    push!(malabody, :(_job.pstate.tensorlogtarget = eye(_job.pstate.size, _job.pstate.size)))

    push!(malabody, :(_job.sstate.oldinvtensor = eye(_job.pstate.size, _job.pstate.size)))

    push!(malabody, :(_job.sstate.oldfirstterm = _job.sstate.oldinvtensor*_job.pstate.gradlogtarget))

    push!(malabody, :(_job.sstate.cholinvtensor = chol(_job.sstate.oldinvtensor, Val{:L})))
  else
    push!(malabody, :(
        if (!_job.sstate.pastupdatetensor && !_job.pstate.diagnosticvalues[1])
          _job.sstate.oldfirstterm = _job.sstate.oldinvtensor*_job.pstate.gradlogtarget
        end
      )
    )
  end

  push!(malabody, :(_job.sstate.μ = _job.pstate.value+0.5*_job.sampler.malastep*_job.sstate.oldfirstterm))

  push!(
    malabody,
    :(
      _job.sstate.pstate.value =
        _job.sstate.μ+_job.sstate.sqrtmalastep*_job.sstate.cholinvtensor*randn(_job.pstate.size)
    )
  )

  push!(malabody, :(_job.parameter.uptogradlogtarget!(_job.sstate.pstate)))

  push!(malabody, :(_job.sstate.ratio = _job.sstate.pstate.logtarget-_job.pstate.logtarget))

  push!(
    malabody,
    :(
      _job.sstate.ratio += (
        0.5*(
          logdet(_job.sampler.malastep*_job.sstate.oldinvtensor)
          +dot(
            _job.sstate.pstate.value-_job.sstate.μ,
            _job.pstate.tensorlogtarget*(_job.sstate.pstate.value-_job.sstate.μ)
          )/_job.sampler.malastep
        )
      )
    )
  )

  push!(malabody, :(_job.sstate.newfirstterm = _job.sstate.oldinvtensor*_job.sstate.pstate.gradlogtarget))

  push!(malabody, :(_job.sstate.μ = _job.sstate.pstate.value+0.5*_job.sampler.malastep*_job.sstate.newfirstterm))

  push!(malabody, :(
      _job.sstate.ratio -= (
        0.5*(
          logdet(_job.sampler.malastep*_job.sstate.oldinvtensor)
          +dot(
            _job.pstate.value-_job.sstate.μ,
            _job.pstate.tensorlogtarget*(_job.pstate.value-_job.sstate.μ)
          )/_job.sampler.malastep
        )
      )
    )
  )

  update = []
  noupdate = []

  push!(update, :(_job.pstate.value = copy(_job.sstate.pstate.value)))

  push!(update, :(_job.pstate.gradlogtarget = copy(_job.sstate.pstate.gradlogtarget)))

  if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
    push!(update, :(_job.pstate.gradloglikelihood = copy(_job.sstate.pstate.gradloglikelihood)))
  end

  if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
    push!(update, :(_job.pstate.gradlogprior = copy(_job.sstate.pstate.gradlogprior)))
  end

  push!(update, :(_job.sstate.oldfirstterm = copy(_job.sstate.newfirstterm)))

  push!(update, :(_job.pstate.logtarget = _job.sstate.pstate.logtarget))

  if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
    push!(update, :(_job.pstate.loglikelihood = _job.sstate.pstate.loglikelihood))
  end

  if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
    push!(update, :(_job.pstate.logprior = _job.sstate.pstate.logprior))
  end

  if in(:accept, job.outopts[:diagnostics])
    push!(update, :(_job.pstate.diagnosticvalues[1] = true))
    push!(noupdate, :(_job.pstate.diagnosticvalues[1] = false))
  end

  if job.tuner.totaltuner.verbose
    push!(update, :(_job.sstate.tune.totaltune.accepted += 1))
  end

  if job.tuner.malatuner.verbose || isa(job.tuner.malatuner, AcceptanceRateMCTuner)
    push!(update, :(_job.sstate.tune.malatune.accepted += 1))
  end

  push!(
    malabody,
    Expr(
      :if,
      :(_job.sstate.ratio > 0 || (_job.sstate.ratio > log(rand()))),
      Expr(:block, update...),
      Expr(:block, noupdate...)
    )
  )

  push!(body, Expr(:if, :(_job.sstate.presentupdatetensor), Expr(:block, smmalabody...), Expr(:block, malabody...)))

  push!(body, :(_job.sstate.pastupdatetensor = _job.sstate.presentupdatetensor))

  if job.tuner.totaltuner.verbose
    push!(
      burninbody,
      :(_job.sstate.tune.totaltune.rate = _job.sstate.tune.totaltune.accepted/_job.sstate.tune.totaltune.proposed)
    )
  end

  if job.tuner.smmalatuner.verbose || isa(job.tuner.smmalatuner, AcceptanceRateMCTuner)
    push!(burninbody, :(rate!(_job.sstate.tune.smmalatune)))
  end

  if job.tuner.malatuner.verbose || isa(job.tuner.malatuner, AcceptanceRateMCTuner)
    push!(burninbody, :(rate!(_job.sstate.tune.malatune)))
  end

  if job.tuner.totaltuner.verbose || job.tuner.smmalatuner.verbose || job.tuner.malatuner.verbose
    fmt_iter = format_iteration(ndigits(job.range.burnin))
    fmt_perc = format_percentage()

    push!(burninbody, :(println(
      "Burnin iteration ",
      $(fmt_iter)(_job.sstate.count),
      " out of ",
      _job.range.burnin,
      "..."
    )))

    if job.tuner.totaltuner.verbose
      push!(burninbody, :(println(
        "  Overall: ",
        $(fmt_iter)(_job.sstate.tune.totaltune.accepted),
        " accepted out of ",
        $(fmt_iter)(_job.sstate.tune.totaltune.proposed),
        " proposed, acceptance rate ",
        $(fmt_perc)(100*_job.sstate.tune.totaltune.rate)
      )))
    end

    if job.tuner.smmalatuner.verbose
      push!(burninbody, :(println(
        "  SMMALA: ",
        $(fmt_iter)(_job.sstate.tune.smmalatune.accepted),
        " accepted out of ",
        $(fmt_iter)(_job.sstate.tune.smmalatune.proposed),
        " proposed, acceptance rate ",
        $(fmt_perc)(100*_job.sstate.tune.smmalatune.rate)
      )))
    end

    if job.tuner.malatuner.verbose
      push!(burninbody, :(println(
        "  MALA: ",
        $(fmt_iter)(_job.sstate.tune.malatune.accepted),
        " accepted out of ",
        $(fmt_iter)(_job.sstate.tune.malatune.proposed),
        " proposed, acceptance rate ",
        $(fmt_perc)(100*_job.sstate.tune.malatune.rate)
      )))
    end
  end

  if isa(job.tuner.smmalatuner, AcceptanceRateMCTuner)
    push!(burninbody, :(tune!(_job.sstate.tune.smmalatune, _job.tuner.smmalatuner)))
  end

  if isa(job.tuner.malatuner, AcceptanceRateMCTuner)
    push!(burninbody, :(tune!(_job.sstate.tune.malatune, _job.tuner.malatuner)))
  end

  if job.tuner.smmalatuner.verbose || isa(job.tuner.smmalatuner, AcceptanceRateMCTuner)
    push!(burninbody, :(reset_burnin!(_job.sstate.tune.smmalatune)))
  end

  if job.tuner.malatuner.verbose || isa(job.tuner.malatuner, AcceptanceRateMCTuner)
    push!(burninbody, :(reset_burnin!(_job.sstate.tune.malatune)))
  end

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.malatuner.verbose ||
    isa(job.tuner.smmalatuner, AcceptanceRateMCTuner) ||
    isa(job.tuner.malatuner, AcceptanceRateMCTuner)
  )
    push!(burninbody, :(_job.sstate.tune.totaltune.totproposed += _job.sstate.tune.totaltune.proposed))

    push!(burninbody, :(_job.sstate.tune.totaltune.proposed = 0))

    if job.tuner.totaltuner.verbose
      push!(burninbody, :(_job.sstate.tune.totaltune.accepted = 0))
      push!(burninbody, :(_job.sstate.tune.totaltune.rate = NaN))
    end

    push!(
      body,
      Expr(
        :if,
        :(
          _job.sstate.tune.totaltune.totproposed <= _job.range.burnin &&
          mod(_job.sstate.tune.totaltune.proposed, _job.tuner.totaltuner.period) == 0
        ),
        Expr(:block, burninbody...)
      )
    )
  end

  if !job.plain
    push!(body, :(produce()))
  end

  @gensym _iterate

  result = quote
    function $_iterate(_job::BasicMCJob)
      $(body...)
    end
  end

  result
end
