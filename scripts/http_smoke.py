"""Exercise a running real HTTP service; retains generated games/reports."""
import argparse
import json
from pathlib import Path
import time
import httpx


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    client=httpx.Client(base_url=args.url,timeout=60)
    def get(path):
        r=client.get('/api'+path);r.raise_for_status();return r.json()
    def post(path,body):
        r=client.post('/api'+path,json=body);r.raise_for_status();return r.json()
    checkpoints=get('/checkpoints')
    report=dict(health=get('/health'),games=[],analyses=[])
    for arch in ('cnn','vit'):
        ck=next(c for c in checkpoints if c['model_config']['architecture']==arch)
        game=post('/games',dict(checkpoint=ck['id'],human=1,simulations=8,size=9))
        game=post(f"/games/{game['id']}/move",dict(action=40,revision=0))
        game=post(f"/games/{game['id']}/ai",dict(revision=game['revision']))
        result=post('/analyze',dict(checkpoint=ck['id'],size=9,moves=game['moves'],simulations=8))
        assert len(result['top5'])==5
        assert all(c['action'] not in game['moves'] for c in result['top5'])
        branch=post(f"/games/{game['id']}/branch",dict(step=1,human=-1))
        assert branch['moves']==[40] and branch['parent']['game']==game['id']
        exported=get(f"/games/{game['id']}/export")
        imported=post('/games/import',exported)
        assert imported['moves']==game['moves']
        report['games'].append(dict(architecture=arch,checkpoint=ck['id'],game=game,branch=branch['id'],imported=imported['id']))
        report['analyses'].append(dict(architecture=arch,top5=result['top5'],elapsed_ms=result['elapsed_ms']))
    job=post('/evaluations',dict(checkpoint=checkpoints[0]['id'],opponent='random',pairs=1,simulations=8,seeds=[42]))
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        result=get('/jobs/'+job['id'])
        if result['status'] in ('completed','failed'):
            break
        time.sleep(.2)
    assert result['status']=='completed',result
    report['evaluation_job']=result
    Path('docs/http-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('HTTP games, AI, replay, branches, export/import, legal Top 5, async evaluation: passed')


if __name__=='__main__':
    main()
