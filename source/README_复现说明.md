# B题支撑材料复现说明

公开包包含四问代码、可复算实验记录、正式场次汇总及论文 LaTeX。原始竞赛存档中的加密导出日志保留在队内。此版复核正式场次汇总和公开回执，记录内保留历史导出文件的 SHA-256。

## 当前论文对应的数据

| 问题与方法 | 演练批次 | 演练场数 | 清除数/真实源总数 | 演练均值（秒/源） | 均值95%区间 | 正式测试 | 实际运行合计 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Q3 有限费用前瞻（strict84） | 5+80+100+121 | 306 | 3977/3977 | 246.2936 | [242.4459,250.1413] | 3 | 309 |
| Q4 冗余扫描删除（FREE） | 15+100+188 | 303 | 3896/3896 | 511.0703 | [501.1629,520.9776] | 3 | 306 |

均值先计算每场总虚拟时间/真实源数，再按场次等权平均；区间使用Student-t方法。模拟器的独立抽样机制及持续监测影响未另行校正，区间用于描述均值不确定性。两问共609场演练，另有6场正式测试；离线比较、日志复算和其他版本不计入模拟器次数。

正式测试不公开真实源数，各场只报告已清除数、总虚拟时间/已清除数和接口现实时间戳之差。Q3正式场次均值266.4584秒/源，Q4为484.5980秒/源，不并入演练均值，不作为另行公布的官方评分。

## 四问入口

| 问题 | 评委入口 | 当前模型实现 |
| --- | --- | --- |
| Q1 | `final_code/q1_entry.py` | `code/Q1/python/q1_geometry.py` |
| Q2 | `final_code/q2_entry.py` | `code/Q2/python/select_second_detection_point.py` |
| Q3 | `final_code/q3_entry.py` | `code/Q3/python/q3_final_main.py`及其锁定共享依赖 |
| Q4 | `final_code/q4_entry.py` | `code/Q4/python/q4_free_tasks_v1/selected_entry.py`及其锁定依赖 |

Q2保留已经存在的候选保留修复。Q3/Q4算法、参数和提交版代码锁沿用原包；此前已整理证据、复算统计和更新论文；本次仅精简附录与更新打包说明。原包15个历史脚本的路径匿名化记录仍见`evidence/anonymization_changes.json`。历史文件角色见`archive/README_历史文件索引.md`，代码没有因重新分类而移动。

## 离线复现

运行环境为Python 3.12及以上、NumPy。先按`requirements.txt`安装依赖，再在本目录执行：

```bash
python -B -X utf8 run_all_checks.py
```

该命令检查环境、四问入口、核心算例、既有单元测试、615份已有公开案例记录、论文数字及文件完整性。只使用本地已有文件，不发起模拟器会话。当前实际验证环境和范围见`复现验证记录.md`；历史验证环境不冒充本次重跑结果。

Q1/Q2可分别运行：

```bash
python -B -X utf8 final_code/q1_entry.py examples/q1_two_bearings.json --output q1_result.json
python -B -X utf8 final_code/q2_entry.py --first-x 0 --first-y 0 --bearing-deg 0 --budget-m 600 --output q2_result.json
```

Q3/Q4入口直接运行仅离线验锁，也向模拟器调用方提供`run_search(client)`。历史会话、正式测试与演练脚本均不在默认离线复现命令中。

## 实验材料与来源

- `results/Q3/practice/all_practice_cases.csv`与Q4对应文件：全部演练逐场表，可按案例编码去重和分层。
- 两问对应的`all_practice_statistics.json`：合并统计、批次统计、源数分层统计；Q4另保留原始批次报告的正态近似区间用于追溯，论文使用Student-t区间。
- `results/Q3/practice/`与`results/Q4/practice/`：609场演练的`summary.json`、`client.jsonl`、`public_ui_after_exit.json`，以及有关历史审核结果。
- `results/Q3/formal/`与`results/Q4/formal/`：6场正式案例的公开汇总、动作回执、结束及上传记录。
- `evidence/verified_results.json`：当前论文采用的609场演练及6场正式案例索引、统计值和包内文件绑定。
- `evidence/simulator_source_provenance.json`：原始来源摘要与匿名提交副本摘要分开登记。复制前核对1389项原始绑定；包内匿名副本另行核对，不要求历史审核的原始摘要等于匿名副本摘要。
- `evidence/paper_results_snapshot.json`：更新前306+115场演练的历史论文快照，仅供追溯，当前论文和复算入口不读取它。
- `evidence/q12_verification.json`：原包Q1/Q2证据；`evidence/qa.json`：本次限定范围的整理与复核记录，不代表对四问所有研究结论重新做最终验收。

公开回执已去除机器人编号等身份字段，旧电脑路径改为相对路径，位置、频道、请求编号、响应、时间及动作顺序保持不变。609份演练加密导出和截图保留在原始实验目录，不重复装入限额20MB的提交ZIP；已在复制前核对其来源摘要。包内公开逐场材料支持独立复算，并非重新运行模拟器。公开包提供六场正式测试的公开回执与汇总。

## 论文LaTeX与自动生成

`paper_snapshot/`现为完整可编辑LaTeX源码，包含正文、附录、参考文献及所有正文图表。安装Matplotlib后，从支撑材料根目录执行：

```bash
python -B -X utf8 code/shared/generate_practice_assets.py
```

会从当前逐场JSON生成`run_counts.tex`、`practice_batches.tex`、`practice_statistics.tex`、`q3_formal.tex`、`q4_formal.tex`、`auto_practice_numbers.tex`及两幅演练图。`--tables-only`只重建文本表格，无需Matplotlib。旧的离线比较图和Q1/Q2图保留现有数据口径。

在`paper_snapshot/`下依次执行（需要XeLaTeX、BibTeX及Windows宋体、黑体、楷体、Times New Roman和Consolas）：

```bash
xelatex -interaction=nonstopmode -halt-on-error B_q12_focus_20260913.tex
bibtex B_q12_focus_20260913
xelatex -interaction=nonstopmode -halt-on-error B_q12_focus_20260913.tex
xelatex -interaction=nonstopmode -halt-on-error B_q12_focus_20260913.tex
```

附录A以1页目录表概括全部支撑文件，公开版逐文件路径及摘要见根目录`SHA256.json`。附录B按四问、共享模块、统计制图排列35份完整源码，共用模块只列一次。代码清单使用浅灰底色、细边框、语法高亮和框外行号，保留原始源码的4547行内容、空行及缩进。每份文件从第1行编号，排版折行不增加行号。模型原始源码、实验记录及来源记录未改变。

`build_appendix.py`使用Pygments生成高亮LaTeX，并逐文件确认词法分析后的字符序列与原始源码完全一致。代码清单直接写入`appendix_submission.tex`，不再保留上一版35份纯文本排印副本。`appendix_listing_map.json`记录原始文件SHA-256、行数及其在LaTeX清单中的位置。

修改源码后，先安装`requirements-analysis.txt`中的Pygments依赖，再从支撑材料根目录执行：

```bash
python -B -X utf8 paper_snapshot/build_appendix.py
```

随后按上面的XeLaTeX命令编译。直接编译随包提供的高亮LaTeX不需要调用Pygments，也不需要开启shell-escape。

当前PDF共108页：摘要1页，正文及参考文献28页，附录79页。摘要参照示例采用“总体思路—各问方法与结果”的结构，并补充正文已有的模型与求解说明；摘要和正文沿用统一行距、段距。重点方法与结论适量加粗。赛题参考文献及其正文引用已删除，保留5条文献。请保留完整支撑材料目录进行复现。

`SHA256.json`覆盖发布时所有文件（自身除外）。重新生成、编译或另存输出后，包内容已经改变，应另存为工作副本；提交原包用于完整性复核。

## 压缩格式

本次以通用Deflate ZIP格式打包，并逐项核对解压内容。当前环境没有WinRAR，未调用GUI；规范中“使用WinRAR软件”的指定工具步骤未在本机执行。ZIP可由WinRAR读取。如需逐字遵循指定压缩软件条款，可在WinRAR中对本目录重新压缩，保持文件内容和目录结构即可。

## 许可

本包内原创源码采用 MIT 许可证，见同目录的 LICENSE。论文、竞赛日志和外部数据按各自的权利说明提供查阅。

