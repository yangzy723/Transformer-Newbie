"""前两课的复习算例；只读取已冻结的第二课配置。"""
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'class2/code'))
from bpe_core import BPE_CORPUS, train_bpe, encode_bpe
from positions import sinusoidal

DTYPE = torch.float64


def first_lesson():
    x = torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=DTYPE)
    scores = x[-1] @ x.T / math.sqrt(2)
    weights = scores.softmax(-1)
    context = weights @ x
    logits = torch.stack((2*context[0], context[1]))
    return dict(scores=scores.tolist(), weights=weights.tolist(), values=x.tolist(),
                context=context.tolist(), candidate_labels=['酒','牛奶'],
                lm_logits=logits.tolist(), lm_probabilities=logits.softmax(-1).tolist())


def residual_example():
    """子层输出已给定，只演示一个位置的两次残差相加。"""
    x_i = torch.tensor([1., 2.], dtype=DTYPE)
    attention_update = torch.tensor([.3, -.2], dtype=DTYPE)
    h_i = x_i + attention_update
    ffn_update = torch.tensor([-.1, .4], dtype=DTYPE)
    y_i = h_i + ffn_update
    return dict(input=x_i.tolist(), attention_update=attention_update.tolist(),
                after_attention=h_i.tolist(), ffn_update=ffn_update.tolist(), output=y_i.tolist())


def bpe_review():
    merges, history = train_bpe(BPE_CORPUS, 4)
    result, steps = encode_bpe('你喜欢喝奶茶', merges)
    trace = [{'rank':None,'tokens':list('你喜欢喝奶茶')}]
    trace += [{'rank':s['rank'],'tokens':s['after']} for s in steps]
    return dict(corpus=BPE_CORPUS, merges=merges, training=history,
                encode_trace=trace, encoded_tokens=result)


def embedding_review():
    table = nn.Embedding(3,2,dtype=DTYPE)
    with torch.no_grad():
        table.weight.copy_(torch.tensor([[0,0],[.2,-.1],[.5,.3]],dtype=DTYPE))
    before = table.weight.detach().clone()
    ids = torch.tensor([1,1,2])
    upstream = torch.tensor([[1,0],[0,2],[3,4]],dtype=DTYPE)
    vectors = table(ids)
    optimizer = torch.optim.SGD(table.parameters(),lr=.1)
    optimizer.zero_grad()
    (vectors*upstream).sum().backward()
    gradient = table.weight.grad.clone()
    optimizer.step()
    training = json.loads((REPO/'class2/assets/training_report.json').read_text())
    return dict(ids=ids.tolist(),weight=before.tolist(),vectors=vectors.detach().tolist(),
                upstream=upstream.tolist(),gradient=gradient.tolist(),
                after=table.weight.detach().tolist(),learning_rate=.1,
                bigram_loss_start=training['history'][0]['loss'],
                bigram_loss_end=training['history'][-1]['loss'])


def rope_intuition():
    def arrow(p):
        theta=p*math.pi/6
        return torch.tensor([math.cos(theta),math.sin(theta)],dtype=DTYPE)
    q,k,qs,ks=[arrow(p) for p in (1,3,3,5)]
    return dict(degrees_per_position=30,positions=[1,3],angles_degrees=[30,90],
                q=q.tolist(),k=k.tolist(),angle_difference_degrees=60,dot=float(q@k),
                shifted_positions=[3,5],shifted_angles_degrees=[90,150],
                shifted_dot=float(qs@ks),pe_width4=sinusoidal(torch.arange(3),4).tolist())


def real_tokenizer():
    path=REPO/'class2/assets/tokenizer_320.json'
    tokenizer=Tokenizer.from_file(str(path))
    text='我喜欢喝牛奶。'
    encoded=tokenizer.encode(text)
    return dict(source=str(path.relative_to(REPO)),text=text,tokens=encoded.tokens,
                ids=encoded.ids,decoded=tokenizer.decode(encoded.ids))


def build_review():
    return dict(first_lesson=first_lesson(),residual=residual_example(),bpe=bpe_review(),embedding=embedding_review(),
                rope_intuition=rope_intuition(),real_tokenizer=real_tokenizer())


if __name__=='__main__':
    print(json.dumps(build_review(),ensure_ascii=False,indent=2))
