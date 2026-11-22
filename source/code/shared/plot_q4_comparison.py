from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
s=json.loads((ROOT/"results/Q4/experiments/round2/comparison.json").read_text(encoding="utf-8"))
rows=s["case_comparisons"]
assert len(rows)==36
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
for split,marker,label in [("development","o","Development (12)"),("holdout","^","Holdout (24)")]:
 rr=[x for x in rows if x["split"]==split]
 axes[0].scatter([x["v1_time_s"]/3600 for x in rr],[x["recovery_time_s"]/3600 for x in rr],marker=marker,label=label,color="#1A6FC4",alpha=.8)
hi=max(max(x["v1_time_s"],x["recovery_time_s"])/3600 for x in rows)*1.06
axes[0].plot([0,hi],[0,hi],"--",color="#767676",label="Equal time")
axes[0].set(xlim=(0,hi),ylim=(0,hi),xlabel="Original main policy: virtual time (h)",ylabel="Recovery policy: virtual time (h)",title="Same-case comparison")
axes[0].legend(fontsize=8)
values=sorted(100*(1-x["ratio"]) for x in rows)
axes[1].bar(range(1,37),values,color=["#1A6FC4" if v>=0 else "#E53935" for v in values])
axes[1].axhline(0,color="#767676",lw=.8)
axes[1].set(xlabel="Cases sorted by time reduction",ylabel="Virtual time reduction (%)",title="All cases retained, including regressions")
fig.suptitle("Q4 synthetic paired evaluation — not formal-test results",fontsize=12)
f=ROOT/"results/Q4/experiments/round2/figures";f.mkdir(exist_ok=True)
fig.savefig(f/"paired_comparison.png",dpi=200)
fig.savefig(f/"paired_comparison.svg")
plt.close(fig)
print(f/"paired_comparison.png")
