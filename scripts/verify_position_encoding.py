#!/usr/bin/env python3
"""Chapter 3 algebra checks and controlled counterexamples; Python stdlib.
No model training, benchmark, or empirical context-length claim.
"""
import math
import random
import json
import platform
from verify_transformer_attention import attention, mask, matmul, softmax


def rotate(x, pos, frequencies):
    assert len(x)==2*len(frequencies)
    result=[]
    for r,theta in enumerate(frequencies):
        c,s=math.cos(pos*theta),math.sin(pos*theta)
        a,b=x[2*r:2*r+2]
        result.extend([c*a-s*b,s*a+c*b])
    return result


def dot(a,b):
    return math.fsum(x*y for x,y in zip(a,b))


def main():
    rng=random.Random(20261026)
    def matrix(n,d):return [[rng.uniform(-1,1) for _ in range(d)] for _ in range(n)]
    errors={};counterexamples={}
    def close(name,a,b):
        assert len(a)==len(b)
        error=max(abs(x-y) for x,y in zip(a,b))
        assert error<1e-11,(name,error)
        errors[name]=error
    def flat(a):return [x for row in a for x in row]
    x=matrix(5,4); wq,wk,wv=matrix(4,4),matrix(4,4),matrix(4,4)
    q,k,v=matmul(x,wq),matmul(x,wk),matmul(x,wv)
    perm=[2,4,0,3,1]
    def reorder(a):return [a[i] for i in perm]
    close('unmasked_permutation_equivariance',flat(attention(reorder(q),reorder(k),reorder(v))),flat(reorder(attention(q,k,v))))
    ca=attention(reorder(q),reorder(k),reorder(v),mask(5,5));cb=reorder(attention(q,k,v,mask(5,5)))
    counterexamples['fixed_causal_mask_breaks_permutation_equivariance']=max(abs(a-b) for a,b in zip(flat(ca),flat(cb)))
    theta=.13;i=7;delta=11
    original=[math.sin(i*theta),math.cos(i*theta)]
    transformed=[math.cos(delta*theta)*original[0]+math.sin(delta*theta)*original[1],-math.sin(delta*theta)*original[0]+math.cos(delta*theta)*original[1]]
    close('sinusoidal_shift',transformed,[math.sin((i+delta)*theta),math.cos((i+delta)*theta)])
    f=[10000**(-2*r/4) for r in range(2)];qi,kj=q[2],k[1]
    close('rope_relative_dot',[dot(rotate(qi,2,f),rotate(kj,7,f))],[dot(qi,rotate(kj,5,f))])
    close('rope_joint_shift',[dot(rotate(qi,12,f),rotate(kj,17,f))],[dot(rotate(qi,2,f),rotate(kj,7,f))])
    close('rope_norm',[dot(rotate(qi,7,f),rotate(qi,7,f))],[dot(qi,qi)])
    close('rope_illustration',[dot(rotate([1,0],1,[math.pi/6]),rotate([1,0],3,[math.pi/6])),dot(rotate([1,0],4,[math.pi/6]),rotate([1,0],6,[math.pi/6]))],[.5,.5])
    counterexamples['rope_nonmonotonic_dot_increase']=math.cos(12*math.pi/6)-math.cos(6*math.pi/6)
    rho=4
    close('pi_position_vs_frequency',rotate(qi,7/rho,f),rotate(qi,7,[t/rho for t in f]))
    d=64;b=10000;bp=b*rho**(d/(d-2));ratios=[bp**(-2*r/d)/b**(-2*r/d) for r in range(d//2)]
    close('ntk_endpoints',[ratios[0],ratios[-1]],[1,1/rho])
    close('ntk_base_vs_ratio',ratios,[rho**(-2*r/(d-2)) for r in range(d//2)])
    # Equivalent coordinate layouts: adjacent-pair versus split-half.
    xr=[qi[0],qi[2],qi[1],qi[3]];out=[]
    for r,t in enumerate(f):out.append(xr[r]*math.cos(7*t)-xr[r+2]*math.sin(7*t))
    for r,t in enumerate(f):out.append(xr[r]*math.sin(7*t)+xr[r+2]*math.cos(7*t))
    target=rotate(qi,7,f);close('consistent_pair_layout_permutation',out,[target[0],target[2],target[1],target[3]])
    tau=.7
    close('temperature_vs_qk_scaling',[dot([a/math.sqrt(tau) for a in qi],[a/math.sqrt(tau) for a in kj])],[dot(qi,kj)/tau])
    for m in [.25,.5]:
        bias=[-m*(4-j) for j in range(5)]
        assert bias==sorted(bias) and bias[-1]==0
        weights=softmax(bias);assert weights==sorted(weights)
    qr=[rotate(row,i,f) for i,row in enumerate(q)];kr=[rotate(row,i,f) for i,row in enumerate(k)]
    full=attention(qr,kr,v,mask(5,5))
    inc=[attention([qr[t]],kr[:t+1],v[:t+1])[0] for t in range(5)]
    close('fixed_rope_full_vs_cache',flat(full),flat(inc))
    nf=[theta/4 for theta in f]
    nqr=[rotate(row,i,nf) for i,row in enumerate(q)];nkr=[rotate(row,i,nf) for i,row in enumerate(k)]
    mixed=attention([nqr[-1]],kr[:-1]+[nkr[-1]],v)
    updated=attention([nqr[-1]],nkr,v)
    counterexamples['mixed_frequency_cache_error']=max(abs(a-b) for a,b in zip(mixed[0],updated[0]))
    recovered=[rotate(rotate(row,-i,f),i,nf) for i,row in enumerate(kr)]
    close('single_layer_key_rerotation',flat(recovered),flat(nkr))
    # Two attention-only layers: upper-layer hidden states depend on lower RoPE.
    old_hidden=full;new_hidden=attention(nqr,nkr,v,mask(5,5))
    upper_q=matmul(new_hidden,wq);upper_k=matmul(new_hidden,wk);upper_v=matmul(new_hidden,wv)
    uq=[rotate(row,i,nf) for i,row in enumerate(upper_q)];uk=[rotate(row,i,nf) for i,row in enumerate(upper_k)]
    old_upper_k=matmul(old_hidden,wk);old_upper_v=matmul(old_hidden,wv)
    stale_k=[rotate(row,i,nf) for i,row in enumerate(old_upper_k[:-1])]+[uk[-1]]
    stale_v=old_upper_v[:-1]+[upper_v[-1]]
    reference=attention([uq[-1]],uk,upper_v)[0];stale=attention([uq[-1]],stale_k,stale_v)[0]
    counterexamples['two_layer_rerotation_keeps_stale_hidden_states']=max(abs(a-b) for a,b in zip(reference,stale))
    assert all(x>1e-6 for x in counterexamples.values()),counterexamples
    print(json.dumps({'python':platform.python_version(),'seed':20261026,'tolerance':1e-11,
        'scope':'algebra and toy attention layers only; no language-model quality evaluation',
        'maximum_absolute_errors':errors,'counterexample_differences':counterexamples,'alibi_bias_and_equal_content_weight_order':'passed','status':'passed'},indent=2))


if __name__=='__main__':main()
