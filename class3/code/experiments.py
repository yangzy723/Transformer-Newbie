"""PPT、Notebook 与运行记录共用的实际计算结果。"""
import json
import math
from pathlib import Path

import torch
from attention import TOKENS,X,W_Q,W_K,W_V,project,attention,apply_rope
from review import build_review


def serializable(value):
    if isinstance(value,torch.Tensor):
        return serializable(value.detach().cpu().tolist())
    if isinstance(value,dict):
        return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [serializable(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):
        return '-inf' if value<0 else 'inf'
    return value


def build_report(destination=None):
    q,k,v=project(X)
    base=attention(q,k,v)
    changed_v=v.clone(); changed_v[0]+=torch.tensor([2.,0.],dtype=X.dtype)
    rv=attention(q,k,changed_v)
    changed_k=k.clone(); changed_k[0]+=torch.tensor([1.,0.],dtype=X.dtype)
    rk=attention(q,changed_k,v)
    def comparison(result):
        return dict(weights_before=base['weights'],weights_after=result['weights'],
                    output_before=base['output'],output_after=result['output'],
                    delta_output=result['output']-base['output'])
    value=dict(**comparison(rv),v_before=v,v_after=changed_v,
               predicted_delta_output=base['weights'][:,0:1]*torch.tensor([2.,0.],dtype=X.dtype))
    key=dict(**comparison(rk),k_before=k,k_after=changed_k,
             scores_before=base['scores'],scores_after=rk['scores'])
    changed_x=X.clone(); changed_x[-1]+=torch.tensor([3.,-2.],dtype=X.dtype)
    q2,k2,v2=project(changed_x)
    masked_after=attention(q2,k2,v2)['output']
    plain=attention(q,k,v,causal=False)['output']
    plain_after=attention(q2,k2,v2,causal=False)['output']
    causality=dict(x_before=X,x_after=changed_x,masked_before=base['output'],
                   masked_after=masked_after,unmasked_before=plain,unmasked_after=plain_after,
                   prefix_max_diff_masked=(masked_after[:-1]-base['output'][:-1]).abs().max().item(),
                   prefix_max_diff_unmasked=(plain_after[:-1]-plain[:-1]).abs().max().item())
    positions=torch.arange(3,dtype=X.dtype)
    qr,kr=apply_rope(q,positions),apply_rope(k,positions)
    rope=attention(qr,kr,v)
    shifted_q,shifted_k=apply_rope(q,positions+7),apply_rope(k,positions+7)
    rope_after=attention(apply_rope(q2,positions),apply_rope(k2,positions),v2)['output']
    position=dict(positions=positions,q_rotated=qr,k_rotated=kr,v_unchanged=v,
                  weights=rope['weights'],output=rope['output'],future_output=rope_after,
                  norm_max_diff=(q.norm(dim=-1)-qr.norm(dim=-1)).abs().max().item(),
                  score_max_diff_shifted=(qr@kr.T-shifted_q@shifted_k.T).abs().max().item(),
                  causality_prefix_max_diff=(rope_after[:-1]-rope['output'][:-1]).abs().max().item())
    report=serializable(dict(base=dict(tokens=TOKENS,x=X,w_q=W_Q,w_k=W_K,w_v=W_V,
                                       q=q,k=k,v=v,**base),
                             interventions=dict(value=value,key=key,causality=causality),
                             position=position,review=build_review(),
                             scale=dict(unscaled_weights=(q@k.T).softmax(-1),
                                        scaled_weights=(q@k.T/math.sqrt(2)).softmax(-1))))
    if destination:
        destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def main():
    path=Path(__file__).resolve().parents[1]/'assets/lesson_report.json'
    r=build_report(path)
    print('第一课旧权重：',r['review']['first_lesson']['weights'])
    print('第一课酒/牛奶概率：',r['review']['first_lesson']['lm_probabilities'])
    print('BPE编码：',r['review']['bpe']['encoded_tokens'])
    print('第三课权重：',r['base']['weights'])
    print('第三课输出：',r['base']['output'])
    for name in ('value','key'):
        print(name,'实验后输出：',r['interventions'][name]['output_after'])
    c=r['interventions']['causality']
    print('修改未来输入，早位置最大变化：mask开=',c['prefix_max_diff_masked'],
          'mask关=',c['prefix_max_diff_unmasked'])
    print('RoPE：',r['position'])


if __name__=='__main__':
    main()
