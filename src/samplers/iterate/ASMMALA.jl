function codegen(::Type{Val{:iterate}}, ::Type{ASMMALA}, job::BasicMCJob)
  result::Expr
  burninbody = []
  smmalanowtensorbody = []
  body = []

  vform = variate_form(job.pstate)
  if vform != Multivariate
    error("Only multivariate parameter states allowed in ASMMALA code generation")
  end

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    push!(body, :(_job.sstate.tune.proposed += 1))
  end

  ###

  smmalabody = []

  push!(body, :(_job.sstate.μ = _job.pstate.value+0.5*_job.sstate.tune.step*_job.sstate.oldfirstterm))

  push!(
    body,
    :(
      _job.sstate.pstate.value =
        _job.sstate.μ+_job.sstate.sqrttunestep*_job.sstate.oldcholinvtensor*randn(_job.pstate.size)
    )
  )

  push!(
    body,
    :(
      _job.sstate.nowupdatetensor =
        !in_mahalanobis_contour(
          _job.sstate.pstate.value,
          _job.sstate.p0value,
          _job.pstate.tensorlogtarget/_job.sstate.tune.step,
          _job.pstate.size
        )
     )
   )

  malabody = []

  push!(malabody, :(_job.sstate.μ = _job.pstate.value+0.5*_job.sstate.tune.step*_job.sstate.oldfirstterm))

  push!(malabody, :(
      _job.sstate.newpstate.value =
        _job.sstate.μ+_job.sstate.newsqrttunestep*_job.sstate.oldcholinvtensor*randn(_job.pstate.size)
    )
  )

  push!(malabody, :(_job.sampler.update!(_job.sstate, _job.sstate.newpstate, _job.pstate)))
  push!(malabody, :(
    _job.sstate.nowupdatetensor =
      !(
        in_mahalanobis_contour(
          _job.sstate.newpstate.value,
          _job.sstate.oldpstate.value,
          _job.pstate.oldtensorlogtarget/_job.sstate.tune.step,
          _job.pstate.size
        ) &
        in_mahalanobis_contour(
          _job.pstate.value,
          _job.sstate.oldpstate.value,
          _job.pstate.oldtensorlogtarget/_job.sstate.tune.step,
          _job.pstate.size
        )
      )
    )
   )

  push!(body, Expr(:if, :(_job.sstate.pastupdatetensor), Expr(:block, smmalabody...), Expr(:block, malabody...)))

  push!(body, :(_job.sstate.count += 1))

  smmalabody = []

  push!(smmalabody, :(_job.sstate.updatetensorcount += 1))

  push!(smmalanowtensorbody, :(_job.parameter.uptotensorlogtarget!(_job.pstate)))

  if job.sampler.transform != nothing
    push!(smmalanowtensorbody, :(_job.pstate.tensorlogtarget = _job.sampler.transform(_job.pstate.tensorlogtarget)))
  end

  push!(smmalanowtensorbody, :(_job.sstate.nowinvtensor = inv(_job.pstate.tensorlogtarget)))

  push!(smmalanowtensorbody, :(_job.sstate.nowfirstterm = _job.sstate.nowinvtensor*_job.pstate.gradlogtarget))

  push!(smmalanowtensorbody, :(_job.sstate.μ = _job.pstate.value+0.5*_job.sstate.tune.step*_job.sstate.nowfirstterm))

  push!(smmalanowtensorbody, :(
      _job.sstate.nowcholinvtensor = _job.sstate.nowsqrttunestep*chol(_job.sstate.nowinvtensor, Val{:L})
    )
  )

  push!(smmalabody, Expr(:if, :(!_job.sstate.pastupdatetensor), Expr(:block, smmalanowtensorbody...)))

  push!(smmalabody, :(_job.parameter.uptotensorlogtarget!(_job.sstate.newpstate)))

  if job.sampler.transform != nothing
    push!(smmalabody, :(_job.sstate.newpstate.tensorlogtarget = _job.sampler.transform(_job.sstate.newpstate.tensorlogtarget)))
  end

  push!(smmalabody, :(_job.sstate.newinvtensor = inv(_job.sstate.newpstate.tensorlogtarget)))

  push!(smmalabody, :(_job.sstate.ratio = _job.sstate.newpstate.logtarget-_job.pstate.logtarget))

  push!(
    smmalabody,
    :(
      _job.sstate.ratio += (
        0.5*(
          logdet(_job.sstate.tune.step*_job.sstate.nowinvtensor)
          +dot(
            _job.sstate.newpstate.value-_job.sstate.μ,
            _job.pstate.tensorlogtarget*(_job.sstate.newpstate.value-_job.sstate.μ)
          )/_job.sstate.tune.step
        )
      )
    )
  )

  push!(smmalabody, :(_job.sstate.newfirstterm = _job.sstate.newinvtensor*_job.sstate.newpstate.gradlogtarget))

  push!(smmalabody, :(_job.sstate.μ = _job.sstate.newpstate.value+0.5*_job.sstate.tune.step*_job.sstate.newfirstterm))

  push!(smmalabody, :(_job.sstate.newcholinvtensor = _job.sstate.newsqrttunestep*chol(_job.sstate.newinvtensor, Val{:L})))

  push!(smmalabody, :(
      _job.sstate.ratio -= (
        0.5*(
          logdet(_job.sstate.tune.step*_job.sstate.newinvtensor)
          +dot(
            _job.pstate.value-_job.sstate.μ,
            _job.sstate.newpstate.tensorlogtarget*(_job.pstate.value-_job.sstate.μ)
          )/_job.sstate.tune.step
        )
      )
    )
  )

  update = []
  noupdate = []

  push!(update, :(_job.pstate.value = copy(_job.sstate.newpstate.value)))

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

  ###

  if (isa(job.tuner, VanillaMCTuner) && job.tuner.verbose) || isa(job.tuner, AcceptanceRateMCTuner)
    push!(burninbody, :(rate!(_job.sstate.tune)))

    if isa(job.tuner, AcceptanceRateMCTuner)
      push!(burninbody, :(tune!(_job.sstate.tune, _job.tuner)))
      push!(burninbody, :(_job.sstate.oldsqrttunestep = _job.sstate.newsqrttunestep))
      push!(burninbody, :(_job.sstate.newsqrttunestep = sqrt(_job.sstate.tune.step)))
    end

    if job.tuner.verbose
      fmt_iter = format_iteration(ndigits(job.range.burnin))
      fmt_perc = format_percentage()

      push!(burninbody, :(println(
        "Burnin iteration ",
        $(fmt_iter)(_job.sstate.tune.totproposed),
        " of ",
        _job.range.burnin,
        ": ",
        $(fmt_perc)(100*_job.sstate.tune.rate),
        " % acceptance rate"
      )))
    end

    push!(burninbody, :(reset_burnin!(_job.sstate.tune)))

    push!(body, Expr(
      :if,
      :(_job.sstate.tune.totproposed <= _job.range.burnin && mod(_job.sstate.tune.proposed, _job.tuner.period) == 0),
      Expr(:block, burninbody...)
    ))
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
