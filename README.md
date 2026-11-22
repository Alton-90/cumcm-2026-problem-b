# 2026 全国大学生数学建模竞赛 B 题 · 上海赛区三等奖

论文《无线电干扰源的鲁棒定位与快速搜索清除》讨论竞赛模拟器中的搜索问题。给定有误差的测向信息，先确定单个干扰源可能出现的范围，再选择后续检测位置。多源场景下则要同时考虑覆盖、搜索顺序以及移动和检测成本。 [参赛论文 PDF](paper/paper.pdf) · [组委会赛题与附件](https://www.mcm.edu.cn/html_cn/node/27b6e148f8113f09b0269f64a02629fb.html)

## 四问的做法

| 问题 | 做法 | 入口 |
| --- | --- | --- |
| Q1 初次定位 | 将测向角区写成半平面约束，求交会区域，计算区域直径并用枚举复核。 | [Q1 入口](source/final_code/q1_entry.py) |
| Q2 第二检测点 | 利用首次成功接收给第二检测点划出可行域，再按最坏情况下的定位范围比较候选。 | [Q2 入口](source/final_code/q2_entry.py) |
| Q3 多源搜索 | 把覆盖点与已发现的源纳入同一条搜索路线，以有限费用前瞻比较后续动作。 | [Q3 入口](source/final_code/q3_entry.py) |
| Q4 FREE 场景 | 调整任务与搜索次序，删除能由既有信息排除的冗余扫描，并检查方案在演练中的表现。 | [Q4 入口](source/final_code/q4_entry.py) |

论文中 Q3 的演练记录有 306 场，Q4 有 303 场。均值按场次等权汇总，分别为 246.29 和 511.07 秒/源，两个数值对应各自的题设。另有 6 场正式测试，按独立口径列在支撑材料中。 [统计口径与正式测试记录](source/README_复现说明.md)

## 离线复现

下载[公开复现包](reproduction/support-material-public.zip)并解压，进入 `B-support/`，使用 Python 3.12 或 3.13 执行以下命令。

```bash
python -m pip install -r requirements.txt
python -B -X utf8 run_all_checks.py
```

核查使用包内文件，包含四问入口、27 项单元测试、615 份已有案例与论文数字复算。包内还有完整 LaTeX 源码、公开回执和文件摘要。加密原始日志由队伍归档。若只想浏览源码，可以直接打开 [`source/code/`](source/code)。

赛前使用的 Skill 及其版本变更见 [B 题 Skill 与强化训练](https://github.com/Alton-90/cumcm-b-skill)。

## 许可

本仓库及公开复现包内的原创源码采用 [MIT 许可证](LICENSE)。参赛论文、竞赛日志、赛题和外部数据按各自权利说明提供查阅。
