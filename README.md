# 人民日报切分语料 n-gram 语言模型实验

本项目使用人民日报 1998 年上半年分词语料，比较 n=1–8 模型的训练时间、模型规模和困惑度，并在四元模型上比较 Jelinek–Mercer、Katz 与 Modified Kneser–Ney 平滑。完整实验结果见[实验报告](experiment_report.md)。

## 环境

- Python 3.10 或更新版本
- Matplotlib 3.8.4

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 数据准备

将课程提供的六个月原始语料放入项目根目录下的 `raw_data/`。脚本需要六个 GBK 编码的 `.txt` 文件，文件名应以 `1998-01` 至 `1998-06` 开头，课程原始文件名可以保留。脚本会按文件名排序并检查月份顺序。完整实验数据也保存在仓库中，公开发布前应确认课程语料的再分发授权。

## 运行

在项目根目录执行：

```powershell
python run_ngram_experiment.py
```

脚本按 1–4 月训练、5 月验证、6 月测试，执行语料清理、模型训练、评估、文本续写和绘图。每次运行会覆盖 `ngram_experiment/` 下本实验生成的语料、结果和图表；完成时会在终端显示这些输出的位置。

运行后生成：

- `ngram_experiment/data/`：按月保存的清理后语料
- `ngram_experiment/results/experiment_results.json`：实验指标与生成样例
- `ngram_experiment/results/training.log`：清理和训练过程记录
- `ngram_experiment/figures/`：训练时间与困惑度、模型规模、平滑方法对比图

训练时间为单次测量，可能随运行环境和系统负载变化。脚本保存统计结果和图表，不保存序列化模型文件。

## 文件结构

```text
.
├── README.md
├── requirements.txt
├── run_ngram_experiment.py
├── experiment_report.md
└── ngram_experiment/
    ├── data/       # 运行后生成，默认忽略
    ├── figures/
    └── results/
```

仓库同时保留实验报告、结构化结果、训练日志、图表、原始课程语料和清理后的文本，便于直接复现实验。
