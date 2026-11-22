"""Finish the already-recorded Q3 case after a simulator window restart."""
import shutil
import sys
from pathlib import Path
import q3_practice_campaign as c

folder = c.OUT/'trials'/'b05_r02'
journal = c.load(folder/'journal.json')
summary = c.load(folder/'summary.json')
assert journal['stage']=='recorded' and journal['case_id']=='JW8C-JJVQ-XR7S-68XY'
assert summary['exited'] and summary['case_id']==journal['case_id']
state=c.ui('read')
assert journal['case_id'] in state['case_ids'], 'The same case must be in the public history'
assert not state['waiting_for_enter'] and not state['test_ended']
shutil.copyfile(c.UI/'window.png',folder/'recovery_practice_list.png')
c.save(folder/'ui_recovery.json', {'at':c.now(),'reason':'Simulator window process changed after completion evidence was saved, before confirmation/export. Same case observed at top of practice history.', 'original_journal':journal,'public_history_case_present':True,'new_trials_started':0})
c.ui('export',journal['case_id'])
c.command([c.PWSH,'-NoProfile','-File',str(c.UI/'save_export_dialog.ps1')])
exports=list((c.ROOT/'results/official_exports').glob('*'+journal['case_id']+'.jlog'))
assert len(exports)==1 and exports[0].stat().st_size>8
summary.update(export_path=exports[0].relative_to(c.ROOT).as_posix(),export_sha256=c.sha(exports[0]))
c.save(folder/'summary.json',summary)
journal.update(stage='exported',completed_at=c.now(),recovery='ui_recovery.json')
c.save(folder/'journal.json',journal)
c.report()
print('Same completed case recovered and exported; no replacement case.')
