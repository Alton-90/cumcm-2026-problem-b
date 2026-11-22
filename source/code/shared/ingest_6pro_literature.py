from pathlib import Path
import hashlib,json,shutil
import fitz
ROOT=Path(__file__).resolve().parents[2]
src=ROOT/'literature_inputs'
out=ROOT/'planning/literature/6pro_20260911';out.mkdir(parents=True,exist_ok=True)
files=[
('MD','B题_文献研读成果_交总控.md'),
('P1','无人机无线电监测混合定位算法.pdf'),
('P2','tokekar2013asensor.pdf'),
('P3','RSS-Based_Localization_of_Multiple_Directional_Sources_With_Unknown_Transmit_Powers_and_Orientations.pdf'),
('P4a','Journal of Field Robotics - 2014 - Vander Hook - Cautious Greedy Strategy for Bearing‐only Active Localization  Analysis.pdf'),
('P4b','JoshV_JFR_2013_Localization.pdf'),
('P5','IEEE_TRO_2012.pdf')]
records=[]
for ident,name in files:
    source=src/name;target=out/name
    assert source.is_file(),str(source)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    else:shutil.copyfile(source,target)
    item={'id':ident,'name':name,'original_path':str(source),'workspace_path':str(target.relative_to(ROOT)).replace('\\','/'),'sha256':digest}
    if name.endswith('.pdf'):
        doc=fitz.open(target);item['pages']=len(doc)
        text='\n\n'.join(f'===== PDF PAGE {i+1} =====\n'+page.get_text(sort=True) for i,page in enumerate(doc))
        (out/f'{ident}_extracted.txt').write_text(text,encoding='utf-8')
        item['text_chars']=len(text)
    records.append(item)
(out/'source_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2))
