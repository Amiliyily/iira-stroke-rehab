"""Read a bounded subset of the official CMU bulk ZIP by HTTP ranges.

No crawling of individual motion URLs; CRC32 validates each archive member.
This is real nonclinical motion used as a robot reference, not real patient
torque or FMA ground truth. Subject-number aliases preclude a clinical cohort.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import struct
import subprocess
import time
import zlib
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
import arm6dof as arm

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data_real/cmu_mocap'
URL='https://mocap.cs.cmu.edu/allasfamc.zip'

def read_amc(raw):
    text=raw.decode('ascii',errors='replace')
    if ':DEGREES' not in text:raise ValueError('AMC angles not marked DEGREES')
    frames=[];current={}
    joints=['rhumerus','rradius','rwrist','rhand']
    def append():
        if all(j in current for j in joints):
            frames.append(current['rhumerus'][:3]+[current[j][0] for j in joints[1:]])
    for line in text.splitlines():
        words=line.strip().split()
        if not words:continue
        if len(words)==1 and words[0].isdigit():
            append();current={}
        elif words[0] in joints:
            current[words[0]]=list(map(float,words[1:]))
    append()
    x=np.asarray(frames,float)
    if x.ndim!=2 or x.shape[1]!=6 or len(x)<80 or not np.isfinite(x).all():
        raise ValueError('Incomplete or short right-arm kinematics')
    return x

def fetch_entry(job):
    entry,end=job
    name=entry['name'];key=Path(name).stem
    path=DATA/'members'/f'{key}.bin'
    for attempt in range(3):
        if not path.exists():
            r=subprocess.run(['curl.exe','--silent','--show-error','--fail',
                              '--connect-timeout','15','--max-time','120',
                              '--range',f'{entry["offset"]}-{end}',URL,
                              '--output',str(path)],capture_output=True)
            if r.returncode:
                path.unlink(missing_ok=True)
                if attempt<2:continue
                return dict(name=name,status='download_failed',error=r.stderr.decode(errors='replace')[:150]),None
        try:
            block=path.read_bytes()
            h=struct.unpack_from('<4s5H3I2H',block)
            if h[0]!=b'PK\x03\x04':raise ValueError('HTTP range returned wrong offset')
            pos=30+h[-2]+h[-1]
            comp=block[pos:pos+entry['compressed']]
            raw=zlib.decompress(comp,-15) if entry['method']==8 else comp
            if zlib.crc32(raw)&0xffffffff!=entry['crc']:raise ValueError('CRC mismatch')
            x=read_amc(raw)
            return dict(name=name,status='ok',capture_block=name.split('/')[-2],
                        frames=len(x),compressed_bytes=entry['compressed'],
                        crc32=entry['crc'],sha256=hashlib.sha256(raw).hexdigest()),x
        except Exception as e:
            # Valid archive member but incompatible data: retained as a logged exclusion.
            return dict(name=name,status='excluded',error=str(e)),None

def make_ref(x):
    # A centre segment of up to 241 frames; shorter recordings are time-normalised.
    # Nominal source sampling is 120 Hz; actual temporal equivalence is not claimed.
    n=min(len(x),241);start=(len(x)-n)//2
    x=np.unwrap(np.deg2rad(x[start:start+n]),axis=0)
    t=np.linspace(0,2.,201)
    ang=np.column_stack([np.interp(t,np.linspace(0,2.,n),x[:,j]) for j in range(6)])
    ang=sosfiltfilt(butter(2,3.,fs=100,output='sos'),ang,axis=0)
    scale=np.array([.2,.2,.2,.1,.1,.1])
    excursion=(ang-ang[0])*scale
    limits=np.array([.6,.6,.6,.3,.3,.3])
    clipped_fraction=float(np.mean(np.abs(excursion)>limits))
    q=arm.Q0_HOME+np.clip(excursion,-limits,limits)
    qd=np.gradient(q,.01,axis=0,edge_order=2)
    qdd=np.gradient(qd,.01,axis=0,edge_order=2)
    p=np.array([arm.fkine_all(v)[0][6] for v in q])
    tau=np.array([arm.rne(q[k],qd[k],qdd[k])+arm.FV*qd[k]+arm.FC*np.tanh(qd[k]/.01)
                  for k in range(len(t))])
    amplitude=max(float(np.max(np.linalg.norm(p-p[0],axis=1))),.01)
    return dict(t=t,q_ref=q,qd_ref=qd,qdd_ref=qdd,p_ref=p,tau_ff=tau,
                dt=.01,T=2.,d_actual=amplitude),clipped_fraction

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args();t0=time.perf_counter()
    (DATA/'members').mkdir(parents=True,exist_ok=True)
    entries=json.loads((DATA/'zip_index.json').read_text(encoding='utf-8'))
    byoffset=sorted(entries,key=lambda e:e['offset'])
    ends={e['name']:byoffset[i+1]['offset']-1 for i,e in enumerate(byoffset[:-1])}
    ends[byoffset[-1]['name']]=1083299307-1
    amc=[e for e in entries if e['name'].lower().endswith('.amc')]
    # Outcome-independent byte-budget rule: all <=100 kB plus smallest per capture block.
    selected={e['name']:e for e in amc if e['compressed']<=100000}
    for sid in sorted({e['name'].split('/')[-2] for e in amc}):
        e=min((e for e in amc if e['name'].split('/')[-2]==sid),key=lambda e:e['compressed'])
        selected[e['name']]=e
    metadata=dict(source=URL,homepage='https://mocap.cs.cmu.edu/',archive_entries=len(entries),
                  archive_amc=len(amc),archive_capture_blocks=112,
                  selection_rule='all AMC <=100000 compressed bytes, plus smallest per capture block',
                  requested_members=len(selected),raw_compressed_budget=sum(e['compressed'] for e in selected.values()),
                  clinically_scored=False,stroke_patients=False,
                  caveats=['Same person may have multiple subject numbers (official warning)',
                           'Real arm angles retargeted to UR5-shaped simulated plant; not human inverse kinematics',
                           'Temporal normalisation and amplitude clipping, no actual force or assistance measurements'])
    (DATA/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    records=[];arrays=[]
    with cf.ThreadPoolExecutor(args.workers) as pool:
        fs={pool.submit(fetch_entry,(e,ends[e['name']])):e['name'] for e in selected.values()}
        for k,f in enumerate(cf.as_completed(fs),1):
            rec,x=f.result();records.append(rec)
            if x is not None:
                ref,clipped=make_ref(x)
                arrays.append((rec['name'],ref));rec['clipped_fraction']=clipped
            if k%20==0 or k==len(fs):
                print(f'{k}/{len(fs)} members; valid={len(arrays)}; elapsed {(time.perf_counter()-t0)/60:.2f} min',flush=True)
                (DATA/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    arrays.sort(key=lambda pair:pair[0])
    names=[a[0] for a in arrays]
    packed={key:np.stack([a[1][key] for a in arrays]).astype(np.float32)
            for key in ['q_ref','qd_ref','qdd_ref','p_ref','tau_ff']}
    packed.update(names=np.array(names),capture_block=np.array([int(n.split('/')[-2]) for n in names]),
                  t=np.linspace(0,2.,201),dt=.01,T=2.,d_actual=np.array([a[1]['d_actual'] for a in arrays]))
    np.savez_compressed(DATA/'references.npz',**packed)
    metadata.update(valid_records=len(arrays),valid_capture_blocks=len(set(packed['capture_block'])),
                    runtime_min=(time.perf_counter()-t0)/60,
                    references_sha256=hashlib.sha256((DATA/'references.npz').read_bytes()).hexdigest())
    (DATA/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
