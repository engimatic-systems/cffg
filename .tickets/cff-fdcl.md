---
id: cff-fdcl
status: closed
deps: [cff-5b8v]
links: []
created: 2026-07-05T23:23:06Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-1d6t
---
# Golden-number parity vs biaslab/LAIF reference

Exact Fig.28 posterior control values + schedule details (2 MP iterations, 20 Newton steps, message order) from the Julia reference; decide whether to match their unconverged 2-iteration schedule or document divergence at our converged fixed point. Sub-agent findings land here.


## Notes

**2026-07-05T23:32:49Z**

SUB-AGENT FINDINGS (ran Julia 1.10 + RxInfer 3.6.0 reference deterministically, biaslab/LAIF@9453782): Fig.28 = CONVERGED values (10 VMP iters, not 2 as paper text implies; iter-2 values 0.26/0.19/0.19/0.37 do NOT match the figure). Golden: u1=[0.25009823,0.19810473,0.19810473,0.35369231], u2=[0.13187529,0.30062771,0.30062771,0.26686930]. Their quirks NOT copied by us: A.+=1e-12, log(d.+1e-6), softmax clamp [-100,0], no damping (reactive scheduling avoids sync limit cycle). Index conventions confirmed identical (ctx fastest, outcome fastest, c=kron(ones(4),[0,0,c,-c])). RESULT: our converged posteriors match to 1.7e-6; golden test at atol 5e-4 in test_tmaze.py. Their pointmass run also selects C at t=1, matching our delta variant.
