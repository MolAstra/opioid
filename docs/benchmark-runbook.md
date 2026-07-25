# DPBench 审稿 benchmark 运行手册

本手册定义 `μORScreen` 当前正式 reviewer 评测流程。`label=1` 始终表示
antagonistic；所有候选都参与比较，不自动选择 Top1。

## 1. 环境与固定数据

主训练环境使用 Python 3.12 并执行 `pip install -e .`。DPBench/DataSAIL 只在
`molm` 环境运行；Chemprop 只使用 GPU。

正式数据由 `workflow/10_build_data` 所有：

```text
workflow/10_build_data/data/
├── source/opioid.csv
└── dpbench/
    ├── dpbench_runtime.json
    └── task_pool/muor_antagonism/processed_<protocol>/
```

三个协议是 scaffold、固定种子 random、DataSAIL。每个协议包含一个 fixed test、五个
train/valid fold、assignments、diagnostics 和 manifest。DPBench 去重后保留 982 条记录。

验证已提交数据或明确重建：

```bash
./workflow/10_build_data/run.sh --protocol all
./workflow/10_build_data/run.sh --protocol all --replace
```

默认 dpnet 命令为：

```text
conda run --no-capture-output -n molm dpnet
```

源码输入或 task metadata 改变时，只允许 `--protocol all --replace` 整体替换。若需要在
runtime snapshot 中记录 DataSAIL checkout，显式传入 `--datasail-source <path>`；代码
不保存机器绝对默认路径。

## 2. 切分审计

```bash
./workflow/20_audit_splits/run.sh --protocol all
```

主审计调用 DPNet 0.14+ 正式 `dpnet analyze`，结果写入
`results/<protocol>/dpnet_analysis/`：离线 HTML、JSON/CSV、analysis manifest 和
checksums，包含数据质量、标签/分子量、scaffold evidence、精确 development/test
ECFP4/Tanimoto、DataSAIL-compatible leakage 及 adapter diagnostics。已有报告只在传入
`--replace` 时原子替换。

DPNet 对 fixed-test CV 有意将五个互斥 validation 合并为唯一 development，因此项目在
`fold_diagnostics/` 额外报告五个 train/valid 的最近邻与全 pair 分布；可用
`--no-fold-diagnostics` 跳过。scaffold 零重叠硬断言来自官方 report.json。random 和
DataSAIL 仅作描述性报告，不宣称 0.70 Tanimoto 硬阈值。

## 3. 模型矩阵

按照不访问 test 的 CV 阶段开始，再运行完整 shared-test 阶段：

```bash
./workflow/30_benchmark_models/run.sh \
  --protocol all --suite traditional --no-test

CUDA_VISIBLE_DEVICES=<gpu-id> ./workflow/30_benchmark_models/run.sh \
  --protocol all --suite chemprop --no-test

CUDA_VISIBLE_DEVICES=<gpu-id> ./workflow/30_benchmark_models/run.sh \
  --protocol all --suite traditional --suite chemprop
```

传统矩阵为 11 个算法 preset × 5 个分子表征，共 55 个候选；另有三个 GPU Chemprop
候选。Chemprop 每 fold 只接收 train/valid，按 `val_loss` early stopping，不使用 test
训练，也不进行 full-development 第六次重训。

每个候选输出五个 validation 和五个 shared-test 分数：AUROC、AUPRC、Accuracy、F1、
MCC。汇总包含 mean、sample SD 和 Student-t 95% CI。validation 指标排序只控制绘图
顺序，不构成选择。

结果位于：

```text
workflow/30_benchmark_models/results/<protocol>/benchmark/
├── fold_metrics.csv
├── cv_summary_metrics.csv
├── matrix_test_metrics.csv
├── run_manifest.json
├── status.json
├── test_predictions/
└── chemprop/
```

该步骤不写 deployment runtime。Git 发布指标、预测、超参数和 manifest；逐折
Chemprop checkpoint、PT、日志及恢复状态只保留为本地可再生运行工件。

## 4. 模型评比图与正式报告

```bash
./workflow/30_benchmark_models/run_full.sh
./workflow/40_report_results/run.sh all --replace
```

步骤 40 只读步骤 30，并原子替换自身的单层 `results/` 结果包。全部 58 个候选采用
完全一致的绘图样式；表格使用独立的 `model_name`/`model_type` 列，图片使用相同内容
的两行标签，内部 `candidate_id` 不变。正式主图为 validation 与 shared-test 两张 AUROC
跨划分热图：`figures/roc_valid.png` 和 `figures/roc_test.png`，数值保留三位小数。

`results/` 仅包含三个 CSV、Markdown、HTML、manifest 和一个含 21 张 PNG 的
`figures/` 目录。正式 report 必须满足：三个协议、58 个候选、每候选五折 validation、
每候选五折 shared-test、完整 matrix-test manifest。CV-only 或缺失候选时必须失败；
结果包不进行排名或自动模型选择。

## 5. 全数据 RF 可解释性

```bash
./workflow/50_explain_rf/run.sh all --replace
```

步骤 50 固定使用 `rf__ecfp_2048`。该候选由用户在 benchmark 结束后根据三协议
validation AUROC 平均值选为解释对象；这不会回写步骤 30/40 的全候选评测合同。

训练前必须从 scaffold、random、DataSAIL 三套正式分区分别重建完整语料，并确认三者
都是相同的 982 个 `sample_id/smiles/label`（754 个 label 0、228 个 label 1）。随后使用
全部 982 条训练一个 RF 工件，生成 class-1 TreeSHAP、Morgan bit 的全部实际原子环境、
hash collision 记录和 12 个训练语料局部 SAR 案例。

该模型已经包含原 fixed test，因此不能再报告新的独立 test 指标。性能证据仍来自步骤
30；步骤 50 的 RF score 未校准，只能用于解释和后续候选库的相对优先级排序。局部案例
也只能描述训练语料上的模型行为，不能标记为 TP/FP 或泛化错误。

## 6. DataSAIL 两阶段筛选模型选择

Workflow 60 只读取 Workflow 40 的正式完整矩阵。它先在 DataSAIL 协议内，对 SVM、RF、
XGBoost、LightGBM、TabPFN、Chemprop、KNN 和 LR 八个家族分别以 validation mean
AUROC 固定一个完整候选；再以这些候选的 shared-test mean AUROC 排序并取前三个家族。
当前固定结果为：

```text
lgbm__ecfp_1024_rdkit2d_normalized_200
tabpfn__ecfp_2048
rf__ecfp_2048
```

Workflow 60 在训练前重新计算并核验该选择，保存八家族完整选择证据和 Workflow 40
输入 SHA256。最终组合属于 `test_informed_screening_ensemble`，shared test 不能再作为
该组合的独立验证。三个 full-data 分数均未校准，主筛选规则保持全部三个分数
`>=0.5`。

## 7. 文献外部案例

```bash
OPIOID_EXTERNAL_GPU=5 ./workflow/70_external_validation/run.sh all --replace
```

Workflow 70 使用 Workflow 60 已冻结的 LightGBM/TabPFN/RF full-data 工件，不重新选模型、
调整阈值或训练。固定源表包含 20 条文献案例（10 条 antagonistic、10 条
non-antagonistic）；按保留立体化学的 canonical SMILES 对 982 条语料审计后，PZM21、
cebranopadol 和 endomorphin-1 三条 non-antagonistic 案例为训练重叠。结果同时报告全部
20 条和排除重叠后的 17 条。

该案例集规模小、人工平衡且非随机，只报告 accuracy、balanced accuracy 与原始
TP/FP/TN/FN（同时给出 sensitivity/specificity 以便审计），不报告 AUROC/AUPRC。HTML
报告逐条列出名称、DOI、训练重叠状态和模型预测；SMILES 保留在预测 CSV 中，避免表格
过宽。它属于探索性外部案例分析，不应表述为代表性 prospective validation。主共识规则
与 Workflow 60 一致：三个未校准 class-1 score 全部 `>=0.5`。

## 8. 单次执行记录

每次正式运行在 `docs/execution-summary.md` 追加：

- 日期、时区、Git SHA 和 dirty 状态；
- 完整命令和 `CUDA_VISIBLE_DEVICES`；
- dpnet、DataSAIL、Chemprop 和主要包版本；
- 三个 partition manifest SHA256 与 DPBench validate 结果；
- benchmark/output root 和三个 run manifest；
- 失败、恢复、跳过或人工中断项；
- 明确确认没有自动 Top1、test-driven tuning 或部署晋升。
- 若运行步骤 50，额外记录 full-data manifest SHA256，并明确其没有独立评测含义。

全部数据与非模型结果由 Git 发布，统一校验入口为：

```bash
python workflow/verify_release.py verify
```

模型二进制、checkpoint、日志和恢复状态不发布。新 clone 必须先重跑 Workflow 50/60，
才能执行 Workflow 70 或启动 Workflow 80。迁移前历史结果只保存在本机
`tmp/archive/`，不作为任何 Workflow 的输入。
