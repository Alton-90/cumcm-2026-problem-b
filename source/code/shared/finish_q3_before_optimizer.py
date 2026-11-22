"""Close out the thirtieth completed practice case before changing the plan."""
import re
import shutil
import sys
import q3_practice_campaign as c

folder=c.OUT/'trials/b06_r05'
journal=c.load(folder/'journal.json')
summary=c.load(folder/'summary.json')
assert journal['stage']=='executed' and journal['case_id']=='5P8M-WAFM-WHAN-Z8TK' and summary['exited']
end=c.ui('read')
assert end['case_ids']==[journal['case_id']] and end['test_ended'] and end['completion_titles']==['问题3演练测试完成']
counts=set()
for text in end['source_texts']:
    m=re.search(r'(?:共|本次案例含干扰源)\s*(\d+).*?全向\s*(\d+).*?定向\s*(\d+)',text)
    if m:counts.add(tuple(map(int,m.groups())))
assert counts=={(11,11,0)}
c.save(folder/'public_ui_after_exit.json',end)
c.command([sys.executable,'-X','utf8',str(c.UI/'capture_window.py')])
shutil.copyfile(c.UI/'window.png',folder/'completion.png')
summary.update(true_source_count=11,true_omnidirectional_count=11,true_directional_count=0,
               s_per_true_source=summary['virtual_time_s']/11,all_cleared=summary['status']=='completed' and summary['cleared_channel_count']==11,
               source_count_observed_after_exit=True,completion_screenshot_sha256=c.sha(folder/'completion.png'))
c.save(folder/'summary.json',summary)
c.ui('confirm');c.ui('return');c.ui('export',journal['case_id'])
c.command([c.PWSH,'-NoProfile','-File',str(c.UI/'save_export_dialog.ps1')])
exports=list((c.ROOT/'results/official_exports').glob('*'+journal['case_id']+'.jlog'))
assert len(exports)==1 and exports[0].stat().st_size>8
summary.update(export_path=exports[0].relative_to(c.ROOT).as_posix(),export_sha256=c.sha(exports[0]))
c.save(folder/'summary.json',summary)
journal.update(stage='exported',completed_at=c.now(),paused_after_this_case=True)
c.save(folder/'journal.json',journal)
print(c.report())
