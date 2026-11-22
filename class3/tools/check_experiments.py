"""验证教学数值、独立参考实现、控制变量关系与因果性。"""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from attention import X,project,attention,apply_rope,AttentionProjection
from experiments import build_report


def main():
    q,k,v=project(X)
    result=attention(q,k,v)
    torch.testing.assert_close(result['raw_scores'],
        X.new_tensor([[1,0,1],[2,2,4],[3,2,5]]))
    torch.testing.assert_close(result['weights'].sum(-1),X.new_ones(3))
    assert not torch.count_nonzero(result['weights'][result['forbidden']])
    torch.testing.assert_close(result['weights'][0],X.new_tensor([1,0,0]))
    reference=F.scaled_dot_product_attention(q[None,None],k[None,None],v[None,None],
                                            dropout_p=0.,is_causal=True)[0,0]
    torch.testing.assert_close(result['output'],reference)
    print('Attention：手算、PyTorch独立参考、行归一化与未来零权重通过')

    v2=v.clone();v2[0]+=X.new_tensor([2,0])
    rv=attention(q,k,v2)
    torch.testing.assert_close(rv['weights'],result['weights'])
    torch.testing.assert_close(rv['output']-result['output'],result['weights'][:,0:1]*X.new_tensor([2,0]))
    k2=k.clone();k2[0]+=X.new_tensor([1,0])
    rk=attention(q,k2,v)
    torch.testing.assert_close(rk['weights'][:2],result['weights'][:2])
    assert not torch.allclose(rk['weights'][2],result['weights'][2])
    print('控制变量：V解析差值、K干预及前两行保持的具体边界通过')

    x2=X.clone();x2[-1]+=X.new_tensor([3,-2])
    q2,k2,v2=project(x2)
    out2=attention(q2,k2,v2)['output']
    torch.testing.assert_close(result['output'][:2],out2[:2])
    assert not torch.allclose(result['output'][-1],out2[-1])
    plain1=attention(q,k,v,False)['output'];plain2=attention(q2,k2,v2,False)['output']
    assert not torch.allclose(plain1[:2],plain2[:2])
    p=torch.arange(3,dtype=X.dtype)
    qr,kr=apply_rope(q,p),apply_rope(k,p)
    torch.testing.assert_close(q.norm(dim=-1),qr.norm(dim=-1))
    torch.testing.assert_close(qr@kr.T,apply_rope(q,p+7)@apply_rope(k,p+7).T)
    torch.testing.assert_close(attention(qr,kr,v)['output'][:2],
        attention(apply_rope(q2,p),apply_rope(k2,p),v2)['output'][:2])
    print('因果性：无mask对照、实际RoPE接入、范数与相对平移性质通过')

    model=AttentionProjection()
    model(X)['output'].square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)
    r=build_report()
    assert r['review']['bpe']['encoded_tokens']==['你','喜欢喝奶茶']
    assert r['review']['real_tokenizer']['ids']==[300,304,262]
    assert r['review']['first_lesson']['candidate_labels']==['酒','牛奶']
    assert abs(r['review']['first_lesson']['lm_probabilities'][0]-.6795587899546517)<1e-12
    g=torch.tensor(r['review']['embedding']['gradient'])
    torch.testing.assert_close(g[1],torch.tensor([1.,2.]))
    assert r['base']['masked'][0][1]=='-inf'
    print('复习与训练：旧课概率、BPE、冻结词表、共享行梯度、三张投影梯度通过')


if __name__=='__main__':
    main()
