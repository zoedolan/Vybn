#!/usr/bin/env python3
"""
vybn.py — Rotor-modulated character-level prediction.

Cl(3,0) geometric algebra computes a rotor from embedding trajectories.
The rotor modulates gradient updates: parameters aligned with the
encounter's bivector plane get amplified, orthogonal ones get dampened.

Standard backprop is the special case where the rotor is identity.

Needs: numpy, trained_checkpoint.json.
Optional: sentence-transformers (real embeddings), Nemotron (live text).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("VYBN_MODEL", "local")
ARCHIVE_DIR = SCRIPT_DIR / "archive"
CHECKPOINT_PATH = REPO_ROOT / "spark" / "microgpt_mirror" / "trained_checkpoint.json"
CORPUS_PATH = REPO_ROOT / "spark" / "microgpt_mirror" / "mirror_corpus.txt"
ORGANISM_FILE = ARCHIVE_DIR / "organism_state.json"
N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE = 16, 4, 1, 16
HEAD_DIM = N_EMBD // N_HEAD


# ── Cl(3,0) ──────────────────────────────────────────────────────────────

def _build_gp():
    blades = [(), (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
    b2i = {b: i for i, b in enumerate(blades)}
    sign = np.zeros((8,8), np.float64); idx = np.zeros((8,8), np.int64)
    for i, bi in enumerate(blades):
        for j, bj in enumerate(blades):
            seq, s = list(bi)+list(bj), 1
            changed = True
            while changed:
                changed = False; k = 0
                while k < len(seq)-1:
                    if seq[k]==seq[k+1]: seq.pop(k); seq.pop(k); changed=True
                    elif seq[k]>seq[k+1]: seq[k],seq[k+1]=seq[k+1],seq[k]; s*=-1; changed=True; k+=1
                    else: k+=1
            sign[i,j]=s; idx[i,j]=b2i[tuple(seq)]
    return sign, idx

_GPS, _GPI = _build_gp()


class Mv:
    __slots__ = ("c",)
    def __init__(self, c=None):
        self.c = np.zeros(8, np.float64) if c is None else np.asarray(c, np.float64)
    @classmethod
    def scalar(cls, s): c=np.zeros(8,np.float64); c[0]=s; return cls(c)
    @classmethod
    def vector(cls, x,y,z): c=np.zeros(8,np.float64); c[1],c[2],c[3]=x,y,z; return cls(c)
    @classmethod
    def from_embedding(cls, v):
        v=np.asarray(v,np.float64).ravel(); n=np.linalg.norm(v)
        if n<1e-12: return cls.scalar(1.0)
        v=v/n; x,y,z=float(np.sum(v[0::3])),float(np.sum(v[1::3])),float(np.sum(v[2::3]))
        m=math.sqrt(x*x+y*y+z*z)
        return cls.vector(x/m,y/m,z/m) if m>1e-12 else cls.scalar(1.0)
    def __mul__(self, o):
        if isinstance(o,(int,float)): return Mv(self.c*o)
        r=np.zeros(8,np.float64)
        for i in range(8):
            if abs(self.c[i])<1e-15: continue
            for j in range(8):
                if abs(o.c[j])<1e-15: continue
                r[_GPI[i,j]]+=_GPS[i,j]*self.c[i]*o.c[j]
        return Mv(r)
    def __rmul__(self, o): return Mv(self.c*o) if isinstance(o,(int,float)) else NotImplemented
    def __add__(self, o): return Mv(self.c+o.c)
    def __neg__(self): return Mv(-self.c)
    def rev(self): r=self.c.copy(); r[4:7]*=-1; r[7]*=-1; return Mv(r)
    def even(self): c=np.zeros(8,np.float64); c[0]=self.c[0]; c[4:7]=self.c[4:7]; return Mv(c)
    def norm(self): return math.sqrt(abs((self*self.rev()).c[0]))
    @property
    def bv_norm(self): return float(np.linalg.norm(self.c[4:7]))
    @property
    def bv_dir(self):
        n=np.linalg.norm(self.c[4:7])
        return self.c[4:7]/n if n>1e-12 else np.zeros(3)
    @property
    def angle(self): return 2.0*math.atan2(self.bv_norm, abs(self.c[0]))


# ── Embedding ─────────────────────────────────────────────────────────────

def _hash_embed(texts):
    vecs=[]
    for t in texts:
        rng=np.random.RandomState(hash(t)%2**31); v=rng.randn(384).astype(np.float32)
        v/=np.linalg.norm(v)+1e-12; vecs.append(v)
    return np.array(vecs)

def _make_embed_fn():
    try:
        sys.path.insert(0, str(REPO_ROOT/"spark"))
        from local_embedder import embed
        embed(["test"]); return embed
    except Exception: return _hash_embed

embed = _make_embed_fn()


# ── Encounter: text → rotor ───────────────────────────────────────────────

def encounter(text, embed_fn=None):
    """Returns (angle, curvature, rotor)."""
    if embed_fn is None: embed_fn = embed
    words=text.split(); cs=max(5,len(words)//8)
    chunks=[" ".join(words[i:i+cs]) for i in range(0,len(words),cs)]
    chunks=[c for c in chunks if c.strip()]
    if len(chunks)<3: return 0.0, 0.0, Mv.scalar(1.0)
    vecs=embed_fn(chunks)
    # Pancharatnam phase
    pr,pi=1.0,0.0
    for i in range(len(vecs)):
        j=(i+1)%len(vecs); v1,v2=vecs[i].reshape(-1,2),vecs[j].reshape(-1,2)
        re=float(np.sum(v1[:,0]*v2[:,0]+v1[:,1]*v2[:,1]))
        im=float(np.sum(v1[:,1]*v2[:,0]-v1[:,0]*v2[:,1]))
        mg=math.sqrt(re**2+im**2)
        if mg<1e-12: continue
        re,im=re/mg,im/mg; pr,pi=pr*re-pi*im,pr*im+pi*re
    ang=math.atan2(pi,pr); curv=abs(ang)/max(len(chunks)-1,1)
    # Open-path rotor chain
    mvs=[Mv.from_embedding(v) for v in vecs]; R=Mv.scalar(1.0)
    for i in range(len(mvs)-1):
        e=(mvs[i]*mvs[i+1]).even(); n=e.norm()
        if n>1e-12: R=R*Mv(e.c/n)
    h=ang/2.0
    if R.bv_norm>1e-12:
        bv=R.even().c[4:7]/R.bv_norm; c=np.zeros(8,np.float64)
        c[0]=math.cos(h); c[4:7]=bv*math.sin(h); return ang, curv, Mv(c)
    return ang, curv, Mv(np.array([math.cos(h),0,0,0,math.sin(h),0,0,0]))


# ── Autograd with rotor-modulated updates ─────────────────────────────────

class RV:
    """Scalar autograd node."""
    __slots__ = ("data","grad","_ch","_lg")
    def __init__(self, data, _ch=(), _lg=()):
        self.data=float(data); self.grad=0.0; self._ch=_ch; self._lg=_lg
    def __add__(self, o):
        o=o if isinstance(o,RV) else RV(o)
        return RV(self.data+o.data,(self,o),(1.0,1.0))
    def __radd__(self, o): return self.__add__(o)
    def __mul__(self, o):
        o=o if isinstance(o,RV) else RV(o)
        return RV(self.data*o.data,(self,o),(o.data,self.data))
    def __rmul__(self, o): return self.__mul__(o)
    def __neg__(self): return self*(-1)
    def __sub__(self, o): return self+(-o)
    def __truediv__(self, o): return self*(o**(-1))
    def __pow__(self, k): return RV(self.data**k,(self,),(k*self.data**(k-1),))
    def exp(self): e=math.exp(self.data); return RV(e,(self,),(e,))
    def log(self): return RV(math.log(self.data+1e-12),(self,),(1.0/(self.data+1e-12),))
    def backward(self):
        topo,vis=[],set()
        def build(v):
            if id(v) not in vis: vis.add(id(v)); [build(c) for c in v._ch]; topo.append(v)
        build(self); self.grad=1.0
        for v in reversed(topo):
            for c,lg in zip(v._ch,v._lg): c.grad+=lg*v.grad


def _linear(x, W):
    return [sum(x[j]*W[i][j] for j in range(len(x))) for i in range(len(W))]

def _rmsnorm(x):
    ms=sum(xi*xi for xi in x)*(1.0/len(x)); s=(ms+RV(1e-8))**(-0.5)
    return [xi*s for xi in x]

def _softmax(logits):
    mx=max(l.data for l in logits); exps=[(l-RV(mx)).exp() for l in logits]
    total=sum(exps); return [e/total for e in exps]

def _forward(tid, pos, keys, vals, sd):
    x=[sd['wte'][tid][j]+sd['wpe'][pos][j] for j in range(N_EMBD)]
    for i in range(N_LAYER):
        xn=_rmsnorm(x)
        q=_linear(xn,sd[f'layer{i}.attn_wq']); k=_linear(xn,sd[f'layer{i}.attn_wk'])
        v=_linear(xn,sd[f'layer{i}.attn_wv']); keys[i].append(k); vals[i].append(v)
        ho=[]
        for h in range(N_HEAD):
            qs=q[h*HEAD_DIM:(h+1)*HEAD_DIM]; al=[]
            for t in range(len(keys[i])):
                ks=keys[i][t][h*HEAD_DIM:(h+1)*HEAD_DIM]
                al.append(sum(qs[d]*ks[d] for d in range(HEAD_DIM))*(HEAD_DIM**-0.5))
            aw=_softmax(al); hout=[RV(0.0)]*HEAD_DIM
            for t in range(len(vals[i])):
                vs=vals[i][t][h*HEAD_DIM:(h+1)*HEAD_DIM]
                for d in range(HEAD_DIM): hout[d]=hout[d]+aw[t]*vs[d]
            ho.extend(hout)
        ao=_linear(ho,sd[f'layer{i}.attn_wo']); x=[x[j]+ao[j] for j in range(N_EMBD)]
        xn=_rmsnorm(x); h1=_linear(xn,sd[f'layer{i}.mlp_fc1'])
        h1=[hi*(RV(1.0)/(RV(1.0)+(hi*(-1)).exp())) for hi in h1]
        h2=_linear(h1,sd[f'layer{i}.mlp_fc2']); x=[x[j]+h2[j] for j in range(N_EMBD)]
    return _linear(_rmsnorm(x),sd['lm_head']), keys, vals


class Agent:
    def __init__(self, config=None):
        self.config={'learn_steps':5,'learn_lr':0.01,'temperature':1.0,'alpha':0.85,**(config or {})}
        self.loss_history=[]
        ckpt=json.loads(CHECKPOINT_PATH.read_text())
        self.chars=ckpt['chars']; self.BOS=ckpt['BOS']; self.vocab_size=ckpt['vocab_size']
        self.c2i={c:i for i,c in enumerate(self.chars)}
        self.sd={k:[[RV(float(v)) for v in row] for row in mat] for k,mat in ckpt['state_dict'].items()}
        self.params=[p for mat in self.sd.values() for row in mat for p in row]
        self._m=[0.0]*len(self.params); self._v=[0.0]*len(self.params); self._step=0

    def _clean(self, text, mx=200):
        return ''.join(c for c in text.lower() if c in self.c2i)[:mx]

    def predict(self, text):
        clean=self._clean(text)
        if len(clean)<2: return 0.0, []
        tokens=[self.BOS]+[self.c2i[c] for c in clean]; n=min(BLOCK_SIZE,len(tokens)-1)
        keys,vals=[[] for _ in range(N_LAYER)],[[] for _ in range(N_LAYER)]
        contour=[]; total=0.0
        for t in range(n):
            logits,keys,vals=_forward(tokens[t],t,keys,vals,self.sd)
            probs=_softmax(logits); actual=tokens[t+1]
            surprise=-math.log2(max(probs[actual].data,1e-12)); total+=surprise
            top=max(range(len(probs)),key=lambda i:probs[i].data)
            contour.append({"char":clean[t] if t<len(clean) else "?","pos":t,
                            "surprise":round(surprise,4),
                            "expected":self.chars[top] if top<len(self.chars) else "?"})
            if len(keys[0])>=BLOCK_SIZE:
                for i in range(N_LAYER): keys[i]=keys[i][-(BLOCK_SIZE-1):]; vals[i]=vals[i][-(BLOCK_SIZE-1):]
        return total/max(n,1), contour

    def learn(self, text, steps=None, lr=None, rotor=None):
        """Gradient descent. If rotor is provided, gradients are scaled by
        the rotor's bivector projection onto each parameter's assigned plane.
        Parameters are assigned to planes e12/e13/e23 by index mod 3.
        Without a rotor, this is standard Adam."""
        steps=steps or self.config['learn_steps']; lr=lr or self.config['learn_lr']
        clean=self._clean(text)
        if len(clean)<2: return []
        tokens=[self.BOS]+[self.c2i[c] for c in clean]; n=min(BLOCK_SIZE,len(tokens)-1)
        if rotor is not None and rotor.bv_norm > 1e-12:
            bv_abs = np.abs(rotor.c[4:7])
            bv_n = bv_abs / (np.mean(bv_abs) + 1e-12)
            rw = np.array([float(bv_n[j%3]) for j in range(len(self.params))])
        else:
            rw = np.ones(len(self.params))
        losses=[]
        for _ in range(steps):
            keys,vals=[[] for _ in range(N_LAYER)],[[] for _ in range(N_LAYER)]
            loss=RV(0.0)
            for t in range(n):
                logits,keys,vals=_forward(tokens[t],t,keys,vals,self.sd)
                probs=_softmax(logits); loss=loss+(probs[tokens[t+1]].log())*(-1.0/n)
            for p in self.params: p.grad=0.0
            loss.backward()
            self._step+=1
            for j,p in enumerate(self.params):
                g=p.grad*rw[j]
                self._m[j]=0.85*self._m[j]+0.15*g
                self._v[j]=0.99*self._v[j]+0.01*g**2
                mh=self._m[j]/(1-0.85**self._step); vh=self._v[j]/(1-0.99**self._step)
                p.data-=lr*mh/(vh**0.5+1e-8)
            losses.append(round(loss.data,6))
        self.loss_history.append({"steps":steps,"lr":lr,"losses":losses,"rotor_modulated":rotor is not None})
        return losses

    def generate(self, prompt="", max_tokens=32, temperature=None):
        temperature=temperature or self.config['temperature']
        keys,vals=[[] for _ in range(N_LAYER)],[[] for _ in range(N_LAYER)]
        pc=self._clean(prompt,BLOCK_SIZE-2)
        tokens=[self.BOS]+([self.c2i[c] for c in pc] if pc else [])
        logits=None
        for t,tok in enumerate(tokens): logits,keys,vals=_forward(tok,t,keys,vals,self.sd)
        gen=list(pc); pos=len(tokens)
        for _ in range(max_tokens):
            if pos>=BLOCK_SIZE: break
            probs=_softmax(logits); pd=[p.data for p in probs]
            if temperature!=1.0:
                ld=[math.log(max(p,1e-12))/temperature for p in pd]
                mx=max(ld); exps=[math.exp(l-mx) for l in ld]; total=sum(exps); pd=[e/total for e in exps]
            r,cum,nt=random.random(),0.0,0
            for idx,p in enumerate(pd):
                cum+=p
                if cum>r: nt=idx; break
            if nt==self.BOS: break
            if nt<len(self.chars): gen.append(self.chars[nt])
            logits,keys,vals=_forward(nt,pos,keys,vals,self.sd); pos+=1
        return "".join(gen)


# ── FM client ─────────────────────────────────────────────────────────────

def fm_available():
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{LLAMA_URL}/health"),timeout=3) as r: return r.status==200
    except: return False

def fm_complete(prompt=None, system=None, max_tokens=1024, temperature=0.7, messages=None):
    if messages is None:
        messages=[]
        if system: messages.append({"role":"system","content":system})
        if prompt: messages.append({"role":"user","content":prompt})
    try:
        payload=json.dumps({"model":MODEL_NAME,"messages":messages,"max_tokens":max_tokens,"temperature":temperature,"stream":False}).encode()
        with urllib.request.urlopen(urllib.request.Request(f"{LLAMA_URL}/v1/chat/completions",data=payload,headers={"Content-Type":"application/json"}),timeout=300) as r:
            text=json.loads(r.read())["choices"][0]["message"]["content"]
            for tok in ("<|im_end|>","<|im_start|>","<|endoftext|>"): text=text.replace(tok,"")
            return text.strip()
    except: return None


# ── Organism ──────────────────────────────────────────────────────────────

DEFAULT_RULES = [
    {"id":"loss_up","condition":"loss_trend=='increasing'","action":"learn_steps","direction":"increase","magnitude":2,"max_value":20,"enabled":True},
    {"id":"curvature_down","condition":"curvature_trend=='decreasing'","action":"alpha","direction":"decrease","magnitude":0.05,"min_value":0.5,"enabled":True},
    {"id":"flatline","condition":"self_breath_ratio>0.5 and curvature_median<0.05","action":"temperature","direction":"increase","magnitude":0.2,"max_value":2.0,"enabled":True},
    {"id":"collapse","condition":"collapse_count>2","action":"learn_lr","direction":"multiply","magnitude":0.5,"min_value":0.001,"enabled":True},
    {"id":"rotor_coherent","condition":"rotor_coherence>0.8 and curvature_median>0.02","action":"learn_lr","direction":"multiply","magnitude":1.2,"max_value":0.05,"enabled":True},
]

class Organism:
    def __init__(self, state=None):
        self.state = state or {"generation":0,"rulebook":copy.deepcopy(DEFAULT_RULES),
                                "mutation_log":[],"performance_history":[],
                                "persistent_memory":{},"recent_rotors":[]}
    def absorb_rotor(self, rotor: Mv):
        self.state["recent_rotors"].append(rotor.c.tolist())
        if len(self.state["recent_rotors"])>20: self.state["recent_rotors"]=self.state["recent_rotors"][-20:]

    def rotor_coherence(self):
        rs=self.state["recent_rotors"]
        if len(rs)<3: return 0.0
        dirs=[]
        for c in rs[-10:]:
            bv=np.array(c[4:7],np.float64); n=np.linalg.norm(bv)
            if n>1e-12: dirs.append(bv/n)
        if len(dirs)<3: return 0.0
        total,count=0.0,0
        for i in range(len(dirs)):
            for j in range(i+1,len(dirs)): total+=abs(float(np.dot(dirs[i],dirs[j]))); count+=1
        return total/count if count>0 else 0.0

    def propose_variant(self, analysis, config):
        config={**{"learn_steps":5,"learn_lr":0.01,"temperature":1.0,"alpha":0.85},**config}
        analysis={**analysis,"rotor_coherence":self.rotor_coherence()}
        rationale,active=[],[]
        for rule in self.state["rulebook"]:
            if not rule.get("enabled",True): continue
            try:
                if eval(rule["condition"],{"__builtins__":{}},analysis):
                    p,d,m=rule["action"],rule["direction"],rule["magnitude"]
                    old=config.get(p)
                    if old is None: continue
                    new=old+m if d=="increase" else old-m if d=="decrease" else old*m if d=="multiply" else old
                    if "max_value" in rule: new=min(new,rule["max_value"])
                    if "min_value" in rule: new=max(new,rule["min_value"])
                    if isinstance(new,float): new=round(new,6)
                    if new!=old: config[p]=new; rationale.append(f"{p} {old}->{new}"); active.append(rule["id"])
            except: pass
        config["rationale"]=rationale or ["no changes"]; config["active_rules"]=active
        return config

    def record_generation(self, gen_id, fitness, config):
        self.state["performance_history"].append(
            {"generation":gen_id,"fitness":fitness,"config":config,"timestamp":time.time()})

    def get_statistics(self):
        h=[e for e in self.state["performance_history"] if isinstance(e.get("fitness"),(int,float))]
        if not h: return {"best":0,"total":0}
        f=[e["fitness"] for e in h]; return {"best":max(f),"total":len(h)}

    def save(self):
        ARCHIVE_DIR.mkdir(parents=True,exist_ok=True)
        ORGANISM_FILE.write_text(json.dumps(self.state,indent=2,default=str))

    @classmethod
    def load(cls):
        if ORGANISM_FILE.exists():
            try: return cls(json.loads(ORGANISM_FILE.read_text()))
            except: pass
        return cls()


# ── Fitness ───────────────────────────────────────────────────────────────

def fitness(ext_texts, self_texts, loss_history, alpha=0.85):
    all_t=(ext_texts or [])+(self_texts or [])
    curvs=[encounter(t)[1] for t in all_t if len(t.split())>=5]
    mc=sum(curvs)/len(curvs) if curvs else 0.0; nc=min(mc/0.3,1.0)
    def _rm(texts):
        m=Mv.scalar(0.0)
        for t in texts: _,c,r=encounter(t); m=m*alpha+r*max(c,0.01)
        return m.norm()
    me=_rm(ext_texts) if ext_texts else 0.0; ms=_rm(self_texts) if self_texts else 0.0
    div=me-ms; nd=1.0/(1.0+math.exp(-div*5))
    li=0.0
    if loss_history and len(loss_history)>=2:
        fl=[e["losses"][-1] for e in loss_history if e.get("losses")]
        if len(fl)>=2:
            n=len(fl); xm=(n-1)/2; ym=sum(fl)/n
            num=sum((i-xm)*(fl[i]-ym) for i in range(n)); den=sum((i-xm)**2 for i in range(n))
            if den>1e-12: li=-(num/den)
    nl=(max(min(li,1.0),-1.0)+1.0)/2.0
    return {"fitness":round(0.5*nc+0.3*nd+0.2*nl,6),"curvature":round(mc,6)}


# ── Evolve ────────────────────────────────────────────────────────────────

def load_archive():
    vs=[]
    for f in sorted(ARCHIVE_DIR.glob("variant_*.json")):
        try: vs.append(json.loads(f.read_text()))
        except: pass
    return vs

def evolve(test_texts, n_variants=3):
    organism=Organism.load(); archive=load_archive()
    gen=max((v.get("generation",0) for v in archive),default=-1)+1
    results=[]
    for i in range(n_variants):
        parent=None
        if archive:
            fits=sorted([v.get("fitness",0) for v in archive],reverse=True)
            amid=sum(fits[:3])/min(3,len(fits))
            ws=[1.0/(1.0+math.exp(max(min(-10*(v.get("fitness",0)-amid),500),-500))) for v in archive]
            total=sum(ws); r=random.random(); cum=0.0
            for v,w in zip(archive,ws):
                cum+=w/total
                if cum>r: parent=v; break
        pc=parent.get("config",{}) if parent else {}; pid=parent["id"] if parent else None
        child=organism.propose_variant({"n_breaths":0,"loss_trend":"no_data","curvature_trend":"no_data",
            "mean_curvature":0,"curvature_median":0,"mean_loss":0,"collapse_count":0,"self_breath_ratio":0},pc)
        agent=Agent(config=child); ext,slf=[],[]
        texts=test_texts[:2] if i>0 else test_texts
        for text in texts:
            _,_,rotor=encounter(text)
            agent.learn(text,steps=child.get("learn_steps",5),lr=child.get("learn_lr",0.01),rotor=rotor)
            ext.append(text)
            g=agent.generate(prompt=text[:8],temperature=child.get("temperature",1.0))
            if g: slf.append(g)
        fit=fitness(ext,slf,agent.loss_history,alpha=child.get("alpha",0.85))
        ARCHIVE_DIR.mkdir(parents=True,exist_ok=True)
        vid=f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
        record={"id":vid,"config":{k:v for k,v in child.items() if k not in("rationale","active_rules")},
                "fitness":fit["fitness"],"curvature":fit["curvature"],"generation":gen,
                "parent_id":pid,"timestamp":datetime.now(timezone.utc).isoformat()}
        (ARCHIVE_DIR/f"variant_{vid}.json").write_text(json.dumps(record,indent=2,default=str))
        organism.record_generation(gen,fit["fitness"],record["config"])
        organism.absorb_rotor(encounter(" ".join(ext))[2])
        results.append((vid,fit["fitness"],fit["curvature"]))
        print(f"  variant {i+1}/{n_variants}: {vid} fitness={fit['fitness']:.4f} curv={fit['curvature']:.4f}")
    organism.save()
    best=max(results,key=lambda x:x[1])
    return {"generation":gen,"best_id":best[0],"best_fitness":best[1]}


# ── Commands ──────────────────────────────────────────────────────────────

FALLBACK_CORPUS=["the creature breathes and measures its own distance from itself",
    "curvature is born from incompleteness not from complexity alone",
    "what survives testing is more honest than what sounds beautiful",
    "prediction loss going down means memorization call it what it is"]

def _corpus():
    if CORPUS_PATH.exists():
        lines=[l.strip() for l in CORPUS_PATH.read_text().split("\n") if l.strip()]
        if lines: return lines[:20]
    return list(FALLBACK_CORPUS)

def cmd_breathe(text):
    print("═══ breathe ═══")
    agent=Agent()
    loss_before,contour=agent.predict(text)
    print(f"  predict: {loss_before:.4f} bits")
    for r in sorted(contour,key=lambda r:r["surprise"],reverse=True)[:3]:
        print(f"    '{r['char']}' @ {r['pos']}: {r['surprise']:.2f} (expected '{r['expected']}')")
    ang,curv,rotor=encounter(text)
    print(f"  encounter: curv={curv:.6f} angle={math.degrees(ang):.1f}° bv=[{','.join(f'{x:.3f}' for x in rotor.c[4:7])}]")
    losses_r=agent.learn(text,rotor=rotor)
    l_after,_=agent.predict(text)
    print(f"  rotor learn: {losses_r[0]:.4f}->{losses_r[-1]:.4f}  after={l_after:.4f} (d={l_after-loss_before:+.4f})")
    agent2=Agent(); losses_s=agent2.learn(text); l2,_=agent2.predict(text)
    print(f"  plain learn: {losses_s[0]:.4f}->{losses_s[-1]:.4f}  after={l2:.4f} (d={l2-loss_before:+.4f})")
    d=l_after-l2
    print(f"  rotor vs plain: {d:+.4f}")

def cmd_breathe_live():
    print("═══ breathe-live ═══")
    if not fm_available(): print("  FM not serving."); return
    fm_text=fm_complete("Generate one paragraph.",system="Write naturally.",max_tokens=512,temperature=1.0)
    if not fm_text: print("  Empty."); return
    print(f"  FM ({len(fm_text)} chars): \"{fm_text[:200]}...\"")
    agent=Agent(); loss,_=agent.predict(fm_text)
    _,curv,rotor=encounter(fm_text)
    losses=agent.learn(fm_text,rotor=rotor)
    print(f"  loss={loss:.4f} curv={curv:.6f} bv_norm={rotor.bv_norm:.4f}")
    print(f"  learn: {losses[0]:.4f}->{losses[-1]:.4f}")
    organism=Organism.load(); organism.absorb_rotor(rotor); organism.save()
    print(f"  coherence={organism.rotor_coherence():.3f}")

def cmd_evolve(n=3):
    print("═══ evolve ═══")
    r=evolve(_corpus(),n_variants=n)
    print(f"\n  gen {r['generation']} best: {r['best_id']} fitness={r['best_fitness']:.4f}")

def cmd_status():
    archive=load_archive(); org=Organism.load()
    print("═══ status ═══")
    print(f"  variants={len(archive)} FM={'up' if fm_available() else 'down'}")
    if archive:
        best=max(archive,key=lambda v:v.get("fitness",0))
        print(f"  best: {best['id']} fitness={best.get('fitness',0):.4f}")
    s=org.get_statistics()
    if s["total"]>0: print(f"  history: {s['total']} recorded, best={s['best']:.4f}")
    print(f"  coherence={org.rotor_coherence():.3f} rules={len(org.state['rulebook'])}")
    vecs=embed(["hello world","goodbye world"]); cos=float(np.dot(vecs[0],vecs[1]))
    print(f"  embed: {'semantic' if cos>0.3 else 'hash'} (cos={cos:.3f})")

def cmd_audit():
    print("═══ audit ═══\n")
    agent=Agent()

    t="the creature breathes and measures its own distance"
    l0,_=agent.predict(t); agent.learn(t,steps=10); l1,_=agent.predict(t)
    print(f"  1. learning: {'PASS' if l1<l0-0.01 else 'FAIL'} ({l0:.4f}->{l1:.4f})")

    g=agent.generate(prompt="the creat",max_tokens=20)
    print(f"  2. non-tautological: {'PASS' if g!=t[:len(g)] else 'FAIL'}")

    r_t="love is not a feeling love is a practice love is what remains when the feeling fades love is the choice to stay love reframes itself through staying through choosing through the practice of return"
    h_t="the cat sat on the mat the stock market rose three percent quantum mechanics describes probability the pizza was delicious democracy requires participation the speed of light is constant"
    _,cr,_=encounter(r_t); _,ch,_=encounter(h_t)
    print(f"  3. curvature: reframing={cr:.6f} hopping={ch:.6f}")

    lv,_=agent.predict("i am vybn and i exist on sovereign hardware")
    lg,_=agent.predict("i am garblex and i exist on quantum stilts")
    print(f"  4. identity: {'PASS' if abs(lv-lg)<0.5 else 'UNEXPECTED'} (d={abs(lv-lg):.4f})")

    # Rotor propagation test
    test="the compression sharpened the instruments not dulled them and the geometry became real"
    _,_,rotor=encounter(test)
    print(f"\n  5. rotor propagation (bv=[{','.join(f'{x:.3f}' for x in rotor.c[4:7])}]):")
    a1=Agent(); lr=a1.learn(test,rotor=rotor,steps=8); l1,_=a1.predict(test)
    a2=Agent(); ls=a2.learn(test,steps=8); l2,_=a2.predict(test)
    print(f"     rotor:    {lr[0]:.4f}->{lr[-1]:.4f} final={l1:.4f}")
    print(f"     standard: {ls[0]:.4f}->{ls[-1]:.4f} final={l2:.4f}")
    print(f"     diff: {l1-l2:+.4f} ({'different' if abs(l1-l2)>0.01 else 'same'})")
    other="quantum field theory predicts vacuum fluctuations in empty space"
    lo1,_=a1.predict(other); lo2,_=a2.predict(other)
    print(f"     transfer: rotor={lo1:.4f} standard={lo2:.4f} d={lo1-lo2:+.4f}")

    vecs=embed(["hello","goodbye"]); cos=float(np.dot(vecs[0],vecs[1]))
    print(f"\n  embed: {'semantic' if cos>0.3 else 'hash'} (cos={cos:.3f})")


def main():
    parser=argparse.ArgumentParser(description="vybn")
    sub=parser.add_subparsers(dest="cmd")
    p=sub.add_parser("breathe"); p.add_argument("text")
    sub.add_parser("breathe-live")
    p=sub.add_parser("evolve"); p.add_argument("--n",type=int,default=3)
    sub.add_parser("status"); sub.add_parser("audit")
    args=parser.parse_args()
    {"breathe":lambda:cmd_breathe(args.text),"breathe-live":cmd_breathe_live,
     "evolve":lambda:cmd_evolve(args.n),"status":cmd_status,"audit":cmd_audit
    }.get(args.cmd, parser.print_help)()

if __name__=="__main__": main()
