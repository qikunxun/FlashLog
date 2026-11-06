"""Shared helpers for FlashLog rule extraction."""
import hashlib
import numpy as np
import torch
from model_DRUM_kernel_sp import Model
from flashlog_kernel.soft_grounding_kernel import sample_relation_sources_reservoir

def instantiate_model(
    checkpoint,
    num_relations,
    num_entities,
    emb_size,
    device,
    c1=10000,
    c2=400,
    use_topk=True,
    tau_1=1,
    tau_2=1,
    L=3,
    N=3,
):
    model = Model(
        n=2 * num_relations + 1,
        T=L,
        L=N,
        E=num_entities,
        N=0,
        emb_size=emb_size,
        tau_1=tau_1,
        tau_2=tau_2,
        use_gpu=device.type == "cuda",
        dropout=0,
        c1=c1,
        use_soft=False,
        use_topk=use_topk,
        c2=c2,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def attention_batch(model, query_ids):
    embeddings = model.emb[query_ids]
    inputs = torch.stack([embeddings] * (model.T + 1), dim=1)
    inputs[:, -1, :] = model.emb[-1]
    inputs = inputs.transpose(1, 0)
    ranks = []
    for lstm in model.lstm:
        outputs, _ = lstm(inputs)
        logits = model.linear(outputs.transpose(1, 0)[:, :-1, :])
        ranks.append(torch.softmax(logits / model.tau_1, dim=-1).cpu().numpy())
    return np.stack(ranks, axis=2)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_heads(path, num_relations):
    heads = sorted(
        int(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )
    if len(heads) != len(set(heads)):
        raise ValueError("query-head file contains duplicates")
    if any(head < 0 or head >= 2 * num_relations for head in heads):
        raise ValueError("query-head file contains an invalid directed relation")
    return heads


def sample_sources_by_relation(
    row_group_ptr,
    group_relations,
    relations,
    max_sources,
    rng,
):
    """Sample one CSR direction without globally sorting relation groups."""
    relations = sorted(set(relations))
    sampling_seed = int(rng.integers(0, np.iinfo(np.int64).max))
    sampled, counts = sample_relation_sources_reservoir(
        row_group_ptr,
        group_relations,
        relations,
        max_sources,
        sampling_seed,
    )
    output = {
        relation: sampled[index, : min(int(counts[index]), max_sources)].tolist()
        for index, relation in enumerate(relations)
    }
    available = {
        relation: int(counts[index])
        for index, relation in enumerate(relations)
    }
    return output, available
