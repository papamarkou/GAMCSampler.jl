if(!isdefined(:Calculus))        using Calculus        end 
if(!isdefined(:ForwardDiff))        using ForwardDiff       end 

# Manually add code to autodiff mod2pi
function Calculus.differentiate(::Calculus.SymbolParameter{:mod2pi}, args, wrt)
   x = args[1]
   xp = Calculus.differentiate(x, wrt)
   if xp != 0
       return Calculus.@sexpr(xp)
   else
       return 0
   end
end

function Base.mod2pi(n::ForwardDiff.Dual)
   return ForwardDiff.Dual(mod2pi(ForwardDiff.value(n)), ForwardDiff.partials(n))
end

