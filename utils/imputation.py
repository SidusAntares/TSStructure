"""Date geometry and selection for visible-observation feature imputation."""
import math
import torch
import torch.nn.functional as F


def _valid(dates, valid):
    return torch.ones_like(dates, dtype=torch.bool) if valid is None else valid.bool()


def _ordered(dates, valid, b):
    indices = torch.where(valid[b])[0]
    return indices[torch.argsort(dates[b, indices], stable=True)]


def linear_interpolation(features, dates, queries, valid=None, query_valid=None):
    """Interpolate from visible bracketing observations; never extrapolate."""
    valid, query_valid = _valid(dates, valid), _valid(queries, query_valid)
    output = features.new_zeros(*queries.shape, features.shape[-1])
    for b in range(features.shape[0]):
        indices = _ordered(dates, valid, b)
        for q in torch.where(query_valid[b])[0]:
            day = queries[b,q]
            left = indices[dates[b,indices] <= day]
            right = indices[dates[b,indices] >= day]
            if not len(left) or not len(right):
                raise ValueError('Query requires visible bracketing observations')
            l,r = left[-1],right[0]
            gap = dates[b,r] - dates[b,l]
            weight = torch.where(gap > 0, (day-dates[b,l])/gap.clamp_min(1e-12), gap.new_zeros(()))
            output[b,q] = features[b,l] * (1-weight) + features[b,r] * weight
    return output


def probe_attention(model, features, dates, shift=0, valid=None):
    """Probe raw temporal attention without dropout, gradients or state updates."""
    encoder = getattr(model, 'temporal_encoder', model)
    modes = [(module,module.training) for module in encoder.modules()]
    try:
        encoder.eval()
        with torch.no_grad():
            result = encoder(features, (dates + shift).long(),
                             return_attention_scores=True, valid_time=valid)
            return result[1].detach()
    finally:
        for module, mode in modes:
            module.training = mode


def _top_union(attention, indices, fraction, heads=None):
    if not len(indices):
        return indices
    k = max(1, math.ceil(fraction * len(indices)))
    values = attention[:,indices] if heads is None else attention[heads][:,indices]
    return indices[torch.unique(values.topk(min(k,len(indices)), dim=-1).indices.flatten())]


def _randperm(n, device, generator):
    # CPU generators also work when training tensors live on CUDA.
    rng_device = getattr(generator, 'device', device) if generator is not None else device
    return torch.randperm(n, generator=generator, device=rng_device).to(device)


def select_gap_queries(dates, attention, valid=None, top_fraction=.25, max_anchors=2, generator=None):
    valid = _valid(dates,valid)
    anchors = torch.zeros_like(valid)
    rows, gaps = [], []
    for b in range(len(dates)):
        indices = _ordered(dates,valid,b)
        top = _top_union(attention[b],indices,top_fraction)
        # Union heads by acquisition date, including duplicate acquisitions.
        selected_dates = set(dates[b,top].tolist())
        candidates, seen = [], set()
        for j,i in enumerate(indices.tolist()):
            day = dates[b,i].item()
            neighbors = []
            if j: neighbors.append((dates[b,i]-dates[b,indices[j-1]]).item())
            if j+1<len(indices): neighbors.append((dates[b,indices[j+1]]-dates[b,i]).item())
            if day in selected_dates and day not in seen and any(g>20 for g in neighbors):
                candidates.append(j); seen.add(day)
        chosen = _randperm(len(candidates), dates.device,generator)[:min(max_anchors,2)].tolist()
        qdays, qgaps = {}, {}
        for c in chosen:
            j = candidates[c]; i = indices[j]; anchors[b,i] = True
            for left,right in ((j-1,j),(j,j+1)):
                if left < 0 or right>=len(indices): continue
                l,r=dates[b,indices[left]].item(),dates[b,indices[right]].item()
                gap=r-l
                fractions=(1/3,2/3) if gap>40 else ((.5,) if gap>20 else ())
                for fraction in fractions:
                    day=l+fraction*gap
                    qdays[day]=day; qgaps[day]=gap
        ordered=sorted(qdays)[:8]
        rows.append(ordered); gaps.append([qgaps[day] for day in ordered])
    width=max(map(len,rows),default=0)
    q=dates.new_zeros((len(dates),width),dtype=torch.float32)
    qvalid=torch.zeros_like(q,dtype=torch.bool); gap_lengths=torch.zeros_like(q)
    for b,row in enumerate(rows):
        q[b,:len(row)]=q.new_tensor(row); qvalid[b,:len(row)]=True
        gap_lengths[b,:len(row)]=q.new_tensor(gaps[b])
    return dict(dates=q,valid=qvalid,anchors=anchors,anchor_count=anchors.sum(-1),query_count=qvalid.sum(-1),gap_lengths=gap_lengths)


def select_recovery_mask(dates, attention, valid=None, generator=None):
    valid=_valid(dates,valid); hidden=torch.zeros_like(valid)
    for b in range(len(dates)):
        indices=_ordered(dates,valid,b)
        if len(indices)<5: continue
        heads=_randperm(attention.shape[1],dates.device,generator)
        nheads=1+int(_randperm(2,dates.device,generator)[0])
        top=_top_union(attention[b],indices,.25,heads[:nheads])
        candidates=[]
        for j in range(2,len(indices)-2):
            if bool((top==indices[j]).any()):
                gap=max((dates[b,indices[j]]-dates[b,indices[j-1]]).item(),(dates[b,indices[j+1]]-dates[b,indices[j]]).item())
                candidates.append((gap,j))
        if not candidates: continue
        # Randomize ties while preferring the sparsest attended anchor.
        candidates=[candidates[i] for i in _randperm(len(candidates),dates.device,generator).tolist()]
        j=max(candidates,key=lambda item:item[0])[1]
        chosen=[j-1,j+1]
        outer=[k for k in (j-2,j+2) if 0<k<len(indices)-1]
        if outer and int(_randperm(2,dates.device,generator)[0]):
            chosen.append(min(outer,key=lambda k:abs((dates[b,indices[k]]-dates[b,indices[j]]).item())))
        hidden[b,indices[chosen]]=True
    return hidden


def reconstruction_loss(prediction, target, log_variance, nll_weight=.01):
    if prediction.numel()==0:
        return prediction.sum()*0
    variance=log_variance.clamp(-6,4).unsqueeze(-1)
    nll=.5*(variance+(prediction-target).square()*torch.exp(-variance))
    return F.huber_loss(prediction,target)+nll_weight*nll.mean()
