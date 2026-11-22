# 2026 全国大学生数学建模竞赛 B 题 · 上海赛区三等奖

参赛论文《无线电干扰源的鲁棒定位与快速搜索清除》解决有界噪声下的几何可行域定位、第二检测点选择、多源覆盖与联合路径规划。获奖级别由参赛团队提供。

| 资料 | 内容 |
| --- | --- |
| [论文 PDF](paper/paper.pdf) | 108 页参赛论文与代码附录 |
| [公开复现 ZIP](reproduction/support-material-public.zip) | 四问代码、算例、论文 LaTeX、逐场公开回执与汇总 |
| [四问代码入口](source/final_code) · [源码](source/code) | 可直接浏览的代码 |
| [复现说明](source/README_复现说明.md) | 算法、口径、输入与工具 |
| [2026 年赛题及官方附件](https://www.mcm.edu.cn/html_cn/node/27b6e148f8113f09b0269f64a02629fb.html) | 全国大学生数学建模竞赛组委会发布 |

复验时解压完整 ZIP，进入 `B-support/`，按其中的 `requirements.txt` 安装依赖后执行 `python -B -X utf8 run_all_checks.py`。公开包使用脱敏路径，保留六场正式测试的公开回执和汇总。加密原始导出日志留在队内归档。

公开版在 Python 3.13.7 和 NumPy 2.3.3 下通过四问入口、27 项单元测试、615 份已有案例核查及论文数字复算。

共用的 Skill 与训练记录见 [数学建模 B 题 Skill 与强化训练](https://github.com/Alton-90/cumcm-b-skill)。
