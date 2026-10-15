"""Toy cross-layer controller study: tabular Q-learning vs rule-based, hourly simulator.
Weather/grid data are SYNTHETIC (no NASA POWER / PVGIS access in sandbox)."""
import numpy as np, json, sys
IT_PEAK=5.0; LAES_CAP=10.0; LAES_P=5.0; ETA=np.sqrt(0.60); BACKLOG_CAP=4.0
ALPHA,BETA,GAMMA=0.4,0.4,2.0   # Eq.10 weights; thermal weight raised from 0.2 to 2.0 so the SLA acts as a near-hard constraint
T_THR=62.0

def make_year(rng, pv_mw):
    d=np.arange(365)[:,None]; h=np.arange(24)[None,:]
    Tm=25.4+9*np.sin(2*np.pi*(d-105)/365)
    T=Tm+6*np.sin(2*np.pi*(h-9)/24)+rng.normal(0,1.5,(365,1))
    T=T+0.0
    cloud=np.clip(0.80-0.30*np.exp(-((d-215)/35.0)**2)+rng.normal(0,0.18,(365,1)),0.25,1.0)
    sun=np.clip(np.sin(np.pi*(h-6)/12),0,None)
    pv=pv_mw*0.78*cloud*sun/0.78*0.78  # peak fraction ~0.78*cloud
    prof=np.where((h>=8)&(h<20),0.9,0.5)*np.ones((365,1))
    ci=np.where((h>=10)&(h<16),0.60,np.where((h>=18)&(h<23),0.82,0.72))*np.ones((365,1))
    tar=np.where((h>=18)&(h<22),0.13,np.where((h>=23)|(h<6),0.07,0.10))*np.ones((365,1))
    f=lambda x:x.reshape(-1)
    return dict(T=f(T),pv=f(pv),prof=f(prof),ci=f(ci),tar=f(tar))

class Env:
    def __init__(s,W): s.W=W; s.n=len(W['T'])
    def reset(s,t0=0): s.t=t0; s.soc=0.0; s.bl=0.0; return s.obs()
    def obs(s):
        W,t=s.W,min(s.t,s.n-1); T=W['T'][t]; h=t%24
        Lnom=IT_PEAK*W['prof'][t]*1.19; net=W['pv'][t]-Lnom
        return (h//2, int(np.digitize(T,[30,38,43])), int(np.digitize(net,[-3,0,3])),
                int(np.digitize(s.soc/LAES_CAP,[0.15,0.5,0.85])), int(np.digitize(s.bl,[0.01,2.0])))
    def step(s,a):
        W,t=s.W,s.t; laes=a%3; wl=(a//3)%3; boost=a//9
        T=W['T'][t]; it_nom=IT_PEAK*W['prof'][t]
        if wl==1 and s.bl+0.2*it_nom<=BACKLOG_CAP: served=0.8*it_nom; s.bl+=0.2*it_nom
        elif wl==2 and s.bl>0:
            cu=min(s.bl,IT_PEAK-it_nom); served=it_nom+cu; s.bl-=cu
        else: served=it_nom
        o_imm=0.045+(0.04 if boost else 0.015)+0.032*max(0,T-40)
        o_cool=0.6*o_imm+0.4*0.20
        L=served*(1+o_cool+0.07); pue=L/max(served,1e-6)
        Tmax=T+(6 if boost else 12)+8*served/IT_PEAK; viol=max(0,Tmax-T_THR)
        r=W['pv'][t]; pch=pd=0.0
        if laes==1: pch=min(LAES_P,(LAES_CAP-s.soc)/ETA); s.soc+=pch*ETA
        elif laes==2: pd=min(LAES_P,s.soc*ETA); s.soc-=pd/ETA
        grid=max(0.0,L+pch-r-pd); curt=max(0.0,r-L-pch)
        em=grid*W['ci'][t]; cost=grid*W['tar'][t]*1000
        R=-ALPHA*pue-BETA*W['ci'][t]*grid/IT_PEAK-GAMMA*viol-0.02*s.bl
        s.t+=1; done=s.t>=s.n
        return s.obs(),R,done,dict(grid=grid,em=em,cost=cost,viol=viol>0,curt=curt,L=L,served=served,pue=pue,bl=s.bl,pv=min(r,L))

def rule(env,o,W,t):
    h=t%24; T=W['T'][t]; net=W['pv'][t]-IT_PEAK*W['prof'][t]*1.19
    laes=1 if net>0 else (2 if 18<=h<23 else 0)
    wl=1 if 18<=h<22 else (2 if 10<=h<16 else 0)
    boost=1 if T>=38 else 0
    return laes+3*wl+9*boost

NA=18; SH=(12,4,4,4,3); DISC=0.8
def train(pv_mw,episodes=80000,seed=0,days=2):
    rng=np.random.default_rng(seed); Q=np.zeros(SH+(NA,)); N=np.zeros(SH+(NA,))
    Ws=[make_year(rng,pv_mw) for _ in range(8)]; ret=[]
    for ep in range(episodes):
        eps=max(0.03,1-ep/(0.5*episodes))
        W=Ws[ep%8]; t0=int(rng.integers(0,365-days))*24
        e=Env({k:v[t0:t0+24*days] for k,v in W.items()}); e.reset()
        e.soc=rng.random()*LAES_CAP*(rng.random()<0.7); e.bl=rng.random()*3*(rng.random()<0.4); o=e.obs(); G=0
        while True:
            a=int(rng.integers(NA)) if rng.random()<eps else int(np.argmax(Q[o]))
            o2,R,done,_=e.step(a); G+=R
            N[o+(a,)]+=1; lr=max(0.02,1/(1+0.05*N[o+(a,)]))
            Q[o+(a,)]+=lr*(R+(0 if done else DISC*Q[o2].max())-Q[o+(a,)]); o=o2
            if done: break
        ret.append(G)
    return Q,np.array(ret)

def evaluate(policy,W):
    e=Env(W); o=e.reset(); S=dict(grid=0,em=0,cost=0,viol=0,curt=0,L=0,served=0,pv=0,R=0)
    while True:
        t=e.t; a=policy(o,t)
        o,R,done,i=e.step(a)
        for k in ('grid','em','cost','viol','curt','L','served','pv'): S[k]+=i[k]
        S['R']+=R
        if done: break
    S['pue']=S['L']/S['served']; S['ren']=S['pv']/S['L']; S['bl_end']=e.bl; S['em']/=1.0; return S

if __name__=="__main__":
    pv=float(sys.argv[1]); Q,ret=train(pv)
    np.save(f'learning_curve_pv{int(pv)}.npy',ret); np.save(f'Q_pv{int(pv)}.npy',Q)
    rows={'rule':[],'rl':[],'static':[]}
    for sd in range(100,105):
        W=make_year(np.random.default_rng(sd),pv)
        rows['rule'].append(evaluate(lambda o,t:rule(None,o,W,t),W))
        rows['rl'].append(evaluate(lambda o,t:int(np.argmax(Q[o])),W))
        rows['static'].append(evaluate(lambda o,t:9,W))
    out={k:{m:(float(np.mean([r[m] for r in v])),float(np.std([r[m] for r in v]))) for m in v[0]} for k,v in rows.items()}
    json.dump(out,open(f'results_pv{int(pv)}.json','w'),indent=1)
