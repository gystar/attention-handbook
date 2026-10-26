#!/usr/bin/env python3
"""Chapter 5 algebra and counterexamples; no trained-model experiments."""
import json
import math
import platform
from verify_transformer_attention import matmul


def main():
    errors={}
    def check(name,a,b):
        error=max(abs(x-y) for x,y in zip(a,b))
        assert error<1e-12,(name,error)
        errors[name]=error
    h,g=[2,-1,4],[.1,.8,.25]
    check('elementwise_example',[a*b for a,b in zip(h,g)],[.2,-.8,1])
    before=sum([1,-1]);after=sum(a*b for a,b in zip([1,-1],[.9,.1]))
    assert before==0 and math.isclose(after,.8)
    weights,values,scalar=[.2,.3,.5],[1,-2,4],.25
    check('scalar_effective_weights',[scalar*sum(a*v for a,v in zip(weights,values))],[sum(scalar*a*v for a,v in zip(weights,values))])
    check('effective_weight_sum',[sum(scalar*a for a in weights)],[scalar])
    source_gates=[.1,.8,.4]
    value_gate=sum(a*g*v for a,g,v in zip(weights,source_gates,values))
    output_gate=scalar*sum(a*v for a,v in zip(weights,values))
    assert abs(value_gate-output_gate)>.01
    H=[[1,2,3],[-1,.5,4]];G=[.1,.8,.25];W=[[1,2],[3,1],[-1,2]]
    direct=matmul([[x*g for x,g in zip(row,G)] for row in H],W)
    folded=matmul(H,[[g*x for x in row] for g,row in zip(G,W)])
    check('static_gate_fold',[x for row in direct for x in row],[x for row in folded for x in row])
    sigmoid=lambda x:1/(1+math.exp(-x))
    # x*sigmoid(x) violates homogeneity, unlike a fixed scalar linear map.
    dynamic_error=abs(2*sigmoid(2)-2*sigmoid(1))
    assert dynamic_error>.1
    print(json.dumps({'python':platform.python_version(),'scope':'algebra and constructed examples; no training, sinks, or performance measurements','tolerance':1e-12,'errors':errors,'projection_cancellation_example':{'before':before,'after':after},'value_vs_output_gate_difference':abs(value_gate-output_gate),'dynamic_gate_homogeneity_error':dynamic_error,'status':'passed'},indent=2))


if __name__=='__main__':
    main()
