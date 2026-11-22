"""Read saved public records only: no simulator session or network connection."""
from pathlib import Path
import csv,hashlib,json,math,re,sys
from statistics_helpers import summarize
ROOT=Path(__file__).resolve().parents[2]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=2e-5):assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,(a,b)
def replay(path,expected,formal=False):
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    seen={};clock=0.;position=(0.,0.);channel=1;cleared=set();entered=False;exited=False;max_error=0.;actions=0;stamps={};observed=0.;max_step_error=0.
    for row in rows:
        assert row.get('http_status')==200 and not row.get('error') and not row.get('transport_failure') and not row.get('http_error')
        request,reply=row['request'],row['response'];assert reply.get('accepted') is True
        rid=request['request_id'];fingerprint=json.dumps([row['path'],request,reply],sort_keys=True)
        if rid in seen:assert seen[rid]==fingerprint;continue
        seen[rid]=fingerprint;action=row['path'];before=clock
        if action=='/enter':
            assert not entered and not exited;entered=True;close(reply['virtual_time_s'],0);stamps['enter']=reply['real_timestamp_ms']
        elif action=='/exit':
            assert entered and not exited;exited=True;stamps['exit']=reply['real_timestamp_ms']
        else:
            assert entered and not exited and action in ('/measure','/clear');ch=request['channel'];assert type(ch) is int and 1<=ch<=20
            point=(request['position']['x'],request['position']['y']);assert all(math.isfinite(x) for x in point)
            clock+=math.dist(position,point)/5;position=point;actions+=1
            if action=='/measure':
                clock+=5+(ch!=channel);channel=ch;assert reply['measure_result'] in ('no_signal','direction','near')
                if reply['measure_result']=='direction':assert 0<=reply['svd_deg']<360
            else:
                assert reply['clear_result'] in ('success','no_target_in_range');success=reply['clear_result']=='success';clock+=3+2*success
                if success:assert ch not in cleared;cleared.add(ch)
        # Public timestamps have six decimal places. A difference between two
        # rounded timestamps has at most 1e-6 s quantization uncertainty.
        # Check every increment and accumulate that explicit rounding budget.
        step_error=abs(reply['virtual_time_s']-observed-(clock-before))
        close(reply['virtual_time_s']-observed,clock-before,1.0001e-6)
        max_step_error=max(max_step_error,step_error)
        observed=reply['virtual_time_s']
        max_error=max(max_error,abs(observed-clock));close(observed,clock,actions*1.0001e-6+1e-9)
    assert entered and exited and len(cleared)==expected['cleared'];close(observed,expected['virtual_time_s'] if formal else expected['total'],1e-9)
    if formal:close((stamps['exit']-stamps['enter'])/1000,expected['runtime_s'],1e-9)
    else:assert len(cleared)==expected['N']
    return actions,max_error,max_step_error

def main():
    evidence=load(ROOT/'evidence/verified_results.json');all_ids=set();checks=[];actions=0;max_error=0.;max_step_error=0.
    for q,expected_n in [('Q3',306),('Q4',303)]:
        rows=evidence['practice'][q];assert len(rows)==expected_n
        for r in rows:
            assert r['case_id'] not in all_ids;all_ids.add(r['case_id']);s=load(ROOT/r['source']);assert sha(ROOT/r['source'])==r['sha256']
            assert s['case_id']==r['case_id'] and s['practice_only'] and not s['official_result'] and s['all_cleared']
            assert s['true_source_count']==r['N']==s['cleared_channel_count']==r['cleared']
            close(s['virtual_time_s'],r['total'],1e-9);close(r['value'],r['total']/r['N'],1e-9)
            ui=load(ROOT/r['public_completion']);assert r['case_id'] in ui['case_ids'] and ui['test_ended']
            counts=[]
            for line in ui['source_texts']:
                found=re.search(r'(?:共|本次案例含干扰源)\s*(\d+).*?全向\s*(\d+).*?定向\s*(\d+)',line)
                if found:counts.append(tuple(map(int,found.groups())))
            assert counts and all(x[0]==r['N'] and x[1]+x[2]==x[0] for x in counts)
            a,e,step=replay(ROOT/r['client'],r);actions+=a;max_error=max(max_error,e);max_step_error=max(max_step_error,step)
        actual=summarize(rows);saved=evidence['summary'][q]
        for key in ['n','mean','sd','sources','cleared','pooled','median','p95','min','max']:close(actual[key],saved[key],1e-8)
        for a,b in zip(actual['ci95'],saved['ci95']):close(a,b,1e-8)
        with (ROOT/'results'/q/'practice/all_practice_cases.csv').open(encoding='utf-8-sig',newline='') as f:csv_rows=list(csv.DictReader(f))
        assert [r['case_id'] for r in csv_rows]==[r['case_id'] for r in rows]
        checks.append({'question':q,'practice_cases':len(rows),'cleared_sources':actual['sources'],'mean_s_per_source':actual['mean']})
    assert len(evidence['formal'])==6
    for r in evidence['formal']:
        assert r['case_id'] not in all_ids;all_ids.add(r['case_id']);s=load(ROOT/r['source'])
        assert s['formal_session_recorded'] and not s.get('practice_only',False) and s['true_source_count'] is None and r['N_public'] is None
        assert sha(ROOT/r['source'])==r['sha256']
        close(s['virtual_time_s'],r['virtual_time_s'],1e-9);assert s['cleared_channel_count']==r['cleared'];close(r['value'],r['virtual_time_s']/r['cleared'],1e-9)
        a,e,step=replay(ROOT/r['client'],r,True);actions+=a;max_error=max(max_error,e);max_step_error=max(max_step_error,step)
        assert len(r['export_sha256']) == 64  # Historical digest of the formal export.
    for item in evidence['bound_files']:
        if not item['path'].startswith('formal_logs/'):
            assert sha(ROOT/item['path'])==item['sha256'],item['path']
    print(json.dumps({'status':'PASS','scope':'saved evidence only; clock reconstruction is not a new simulator test','practice':checks,'formal_cases':6,'unique_case_ids':len(all_ids),'actions_reconstructed':actions,'max_action_clock_error_s':max_error,'max_increment_error_s':max_step_error,'increment_rounding_tolerance_s':1.0001e-6,'package_bindings_checked':len(evidence['bound_files'])},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
