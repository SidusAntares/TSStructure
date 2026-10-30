"""TimePerceiver-inspired latent feature imputer using visible date geometry.

Architecture reference: https://github.com/efficient-learning-lab/TimePerceiver
This adaptation uses observation tokens, continuous dates, anchor conditioning,
residual PSE predictions and uncertainty, rather than the original patch model.
"""
import math
import torch
from torch import nn
from utils.imputation import linear_interpolation, _valid, _ordered


class _CrossBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.query_norm=nn.LayerNorm(width)
        self.context_norm=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
        self.norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*width,width),nn.Dropout(dropout))
        self.dropout=nn.Dropout(dropout)

    def forward(self, query, context, valid=None):
        result=self.attention(self.query_norm(query),self.context_norm(context),self.context_norm(context),
                              key_padding_mask=None if valid is None else ~valid,need_weights=False)[0]
        query=query+self.dropout(result)
        return query+self.ffn(self.norm(query))


class _Round(nn.Module):
    def __init__(self,width,heads,dropout,self_blocks):
        super().__init__()
        self.read=_CrossBlock(width,heads,dropout)
        self.blocks=nn.ModuleList([_CrossBlock(width,heads,dropout) for _ in range(self_blocks)])
        self.write=_CrossBlock(width,heads,dropout)

    def forward(self,latents,tokens,valid):
        latents=self.read(latents,tokens,valid)
        for block in self.blocks:
            latents=block(latents,latents)
        tokens=self.write(tokens,latents)
        return latents,tokens.masked_fill(~valid[...,None],0)


class TimePerceiverImputer(nn.Module):
    def __init__(self,feature_dim=128,d_model=256,num_latents=32,rounds=2,self_blocks=3,heads=8,dropout=.1):
        super().__init__()
        self.feature_dim=feature_dim; self.d_model=d_model
        self.input_projection=nn.Linear(feature_dim,d_model)
        self.gap_projection=nn.Sequential(nn.Linear(2,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.anchor_embedding=nn.Embedding(2,d_model)
        self.latents=nn.Parameter(torch.randn(num_latents,d_model)*.02)
        self.rounds=nn.ModuleList([_Round(d_model,heads,dropout,self_blocks) for _ in range(rounds)])
        self.query_context=nn.Linear(2*feature_dim,d_model)
        self.query_distances=nn.Sequential(nn.Linear(2,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.query_cross=_CrossBlock(d_model,heads,dropout)
        self.anchor_cross=_CrossBlock(d_model,heads,dropout)
        self.output=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,feature_dim))
        self.variance=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,1))
        frequency=torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*(-math.log(1000.)/d_model))
        self.register_buffer('frequency',frequency)

    def _position(self,days):
        angles=days[...,None]*self.frequency
        output=angles.new_zeros(*days.shape,self.d_model)
        output[...,0::2]=angles.sin()
        output[...,1::2]=angles[...,:self.d_model//2].cos()
        return output

    def _anchor_context(self,query,tokens,dates,queries,valid,query_valid,anchors):
        """Condition each query on at most its two nearest visible anchors."""
        contexts=tokens.new_zeros(*queries.shape,2,self.d_model)
        context_valid=torch.zeros(*queries.shape,2,dtype=torch.bool,device=tokens.device)
        for b in range(len(dates)):
            observed=_ordered(dates,valid,b)
            selected=observed[anchors[b,observed]]
            for q in range(queries.shape[1]):
                if not bool(query_valid[b,q]):
                    # One harmless key avoids all-masked attention for padding.
                    context_valid[b,q,0]=True
                    continue
                day=queries[b,q]
                if len(selected):
                    order=torch.argsort((dates[b,selected]-day).abs(),stable=True)
                    local=selected[order[:2]]
                else:
                    left=observed[dates[b,observed]<=day][-1]
                    right=observed[dates[b,observed]>=day][0]
                    local=torch.unique(torch.stack([left,right]))
                contexts[b,q,:len(local)]=tokens[b,local]
                context_valid[b,q,:len(local)]=True
        conditioned=self.anchor_cross(query.reshape(-1,1,self.d_model),
                                      contexts.reshape(-1,2,self.d_model),
                                      context_valid.reshape(-1,2)).reshape_as(query)
        return conditioned.masked_fill(~query_valid[...,None],0)

    def forward(self,features,dates,queries,valid=None,query_valid=None,anchors=None):
        valid=_valid(dates,valid); query_valid=_valid(queries,query_valid)
        if not bool(valid.any(-1).all()):
            raise ValueError('Every sample requires a valid observed date')
        baseline=linear_interpolation(features,dates,queries,valid,query_valid)
        if queries.shape[1]==0:
            return dict(prediction=baseline,linear=baseline,log_variance=features.new_zeros(queries.shape))
        safe_features=features.masked_fill(~valid[...,None],0)
        safe_dates=dates.masked_fill(~valid,0).to(features.dtype)
        safe_queries=queries.masked_fill(~query_valid,0).to(features.dtype)
        gaps=features.new_zeros(*dates.shape,2)
        context=features.new_zeros(*queries.shape,2*self.feature_dim)
        distances=features.new_zeros(*queries.shape,2)
        for b in range(features.shape[0]):
            indices=_ordered(dates,valid,b)
            delta=(safe_dates[b,indices[1:]]-safe_dates[b,indices[:-1]])
            gaps[b,indices[1:],0]=delta; gaps[b,indices[:-1],1]=delta
            for q in torch.where(query_valid[b])[0]:
                day=queries[b,q]
                left=indices[dates[b,indices]<=day][-1]
                right=indices[dates[b,indices]>=day][0]
                context[b,q]=torch.cat([safe_features[b,left],safe_features[b,right]])
                distances[b,q,0]=day-dates[b,left]; distances[b,q,1]=dates[b,right]-day
        flags=torch.zeros_like(valid) if anchors is None else anchors.bool() & valid
        tokens=self.input_projection(safe_features)+self._position(safe_dates)+self.gap_projection(torch.log1p(gaps.clamp_min(0)))+self.anchor_embedding(flags.long())
        tokens=tokens.masked_fill(~valid[...,None],0)
        latents=self.latents.unsqueeze(0).expand(features.shape[0],-1,-1)
        for round_block in self.rounds:
            latents,tokens=round_block(latents,tokens,valid)
        query=self._position(safe_queries)+self.query_context(context)+self.query_distances(torch.log1p(distances.clamp_min(0)))
        query=self._anchor_context(query,tokens,dates,safe_queries,valid,query_valid,flags)
        query=self.query_cross(query,tokens,valid)
        prediction=(baseline+self.output(query)).masked_fill(~query_valid[...,None],0)
        log_variance=self.variance(query).squeeze(-1).clamp(-6,4).masked_fill(~query_valid,0)
        return dict(prediction=prediction,log_variance=log_variance,linear=baseline)
