"""Method adapters for FlashLog recovery and rule inference.

The native sampler, source folds, candidate budgets, Train gate, output order,
and exact graph executor are retained. DRUM uses the native functions unchanged.
"""
import importlib
import itertools
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'rules')]
import numpy as np
import torch
from rules import extract as recovery
from rules import evaluate as evaluation

NATIVE_ATTENTION=recovery.attention_batch
NATIVE_WEIGHTS=recovery.attention_rank_weights
NATIVE_SAMPLE=recovery.ground_head_batch_source_folds_sampled_capped_counts
NATIVE_SCORE=evaluation.score_rule_batch


def instantiate(method,checkpoint,num_relations,num_entities,emb_size,device,tau_1=1,tau_2=1,L=3,N=3,**unused):
    Model=importlib.import_module('model_'+method+'_kernel_sp').Model
    model=Model(n=2*num_relations+1,T=L,L=N,E=num_entities,N=0,emb_size=emb_size,
                tau_1=tau_1,tau_2=tau_2,use_gpu=device.type=='cuda',dropout=0,c1=1000,
                use_soft=False,use_topk=True,c2=400)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def compile_neurallp_attention(probabilities,memory):
    """Nonempty temporal chains, with exact memory coefficients.

    This is an attention-polynomial decomposition, not a claim that NeuralLP's
    query-dependent intermediate normalizers can be represented by scalars.
    The rule language excludes empty bodies. No weight cutoff is introduced.
    """
    B,T,V=probabilities.shape
    assert len(memory)==T+1
    subsets=tuple(s for n in range(1,T+1) for s in itertools.combinations(range(T),n))
    compiled=np.zeros((B,T,len(subsets),V+1),dtype=np.float64)
    for rank,subset in enumerate(subsets):
        coefficient=np.ones(B,dtype=np.float64)
        previous=0
        for t in subset:
            coefficient*=memory[t][:,previous].astype(np.float64)
            previous=t+1
        coefficient*=memory[T][:,previous].astype(np.float64)
        for t in range(T):
            if t in subset:
                compiled[:,t,rank,:V]=probabilities[:,t,:]
            else:
                compiled[:,t,rank,V]=1.
        compiled[:,0,rank,:]*=coefficient[:,None]
    return compiled


@torch.inference_mode()
def neurallp_attention(model,heads):
    embedding=model.emb(heads)
    inputs=torch.stack([embedding]*(model.T+1),dim=1)
    inputs[:,-1,:]=model.emb(torch.full_like(heads,model.n))
    hidden=model.lstm(inputs.transpose(1,0))[0].transpose(1,0)
    probabilities=torch.softmax(model.linear(hidden[:,:-1,:])/model.tau_1,dim=-1)
    memory=[torch.softmax(torch.einsum('bd,btd->bt',hidden[:,t],hidden[:,:t+1]),dim=-1).cpu().numpy()
            for t in range(model.T+1)]
    return compile_neurallp_attention(probabilities.cpu().numpy(),memory)


def max_identity_weights(attention,rows,identity):
    """smDRUM and mmDRUM collapse alternative identity placements by MAX."""
    return recovery.collapsed_rule_weights(attention, rows, identity, maximum=True)


def neural_weights(attention,rows,identity):
    # NeuralLP has one output channel: sum temporal branches BEFORE normalizing.
    return NATIVE_WEIGHTS(attention,rows,identity).sum(axis=1,keepdims=True)


def neural_sample(attentions,*args,**kwargs):
    # Each sampled rank gets equal requested walks in the native sampler. A
    # scalar memory coefficient cancels in its within-rank transition proposal.
    # Normalize before float32 conversion to avoid losing tiny positive chains.
    from rules.neural_sampling import sampling_proposal
    return NATIVE_SAMPLE(sampling_proposal(attentions),*args,**kwargs)


def method_score(method,sources,rules,csr,num_relations,num_entities,rule_chunk_size,aggregations):
    if method not in ('smDRUM','mmDRUM') or tuple(aggregations)!=('routed_normalized',) or not rules:
        return NATIVE_SCORE(sources,rules,csr,num_relations,num_entities,rule_chunk_size,
                            aggregations)
    compact=evaluation.compact_routed_rules(rules)
    from rules.max_inference import score
    output=score(evaluation,sources,compact,csr,num_relations,num_entities,
                 rule_chunk_size,final_max=method=='mmDRUM')
    return {'routed_normalized':output}


def install(method):
    assert method in ('DRUM','smDRUM','mmDRUM','NeuralLP')
    recovery.FALLBACK_CHANNEL_AGGREGATION = "max" if method == "mmDRUM" else "sum"
    recovery.instantiate_model=lambda *a,**kw: instantiate(method,*a,**kw)
    recovery.attention_batch=neurallp_attention if method=='NeuralLP' else NATIVE_ATTENTION
    recovery.attention_rank_weights=(max_identity_weights if method in ('smDRUM','mmDRUM') else
                                    neural_weights if method=='NeuralLP' else NATIVE_WEIGHTS)
    recovery.ground_head_batch_source_folds_sampled_capped_counts=neural_sample if method=='NeuralLP' else NATIVE_SAMPLE
    evaluation.score_rule_batch=lambda *a,**kw: method_score(method,*a,**kw)
