# Protein-Only Beam Search Postprocess

## 背景

当前 EM3DFold 默认启用的是基于 `flood_fill_with_edge_dual(...)` 的后处理 tracing 流程。这个流程目前有几个特点：

- 默认主流程保持不变，仍然是生产路径
- tracing 主要依赖 `C-N` 几何关系和 `pred_edge_existence_dict`
- 现有 greedy / flood-fill 风格在局部歧义区域只能走出一条主分支
- 如果某些真实相邻残基的 `C-N` 预测偏差较大，即使 `CA-CA` 距离接近 `3.8A`，也可能完全不进入候选列表

因此，这次新增了一条完全并行的实验性 protein-only beam-search 后处理支线，用来：

- 在不影响默认输出的前提下，提高蛋白 tracing 的候选召回
- 通过 beam search 枚举多个 plausible tracing 分支，而不是只保留贪心单解
- 利用 `CA-CA` 距离作为 `C-N` 失败时的 rescue recall 通道
- 先把尽可能多的候选路径导出出来，便于观察和分析


## 原始计划

### 总体目标

新增一条完全独立、并行于当前默认 postprocess 的实验性后处理分支：

- 只处理蛋白质残基
- 完全忽略核酸
- 不改动原始 `final_results_align_to_sequence(...)`
- 不改动原始 `flood_fill_with_edge_dual(...)`
- 不覆盖任何当前默认输出
- 所有新输出统一写到单独目录：
  - `output_dir/postprocess_beam_protein/`

### 计划要求

#### 1. 保持现有主流程不变

默认生产流程必须继续保持：

- `final_results_align_to_sequence(...)` 行为不变
- `flood_fill_with_edge_dual(...)` 行为不变
- 现有 `output.cif`
- 现有 `before_prune.cif`
- 现有 `after_prune.cif`
- 现有 protein / NA 输出文件

实验 beam 分支只能作为额外的 sibling branch 存在。

#### 2. 新增可选开关

增加新的 opt-in 运行参数：

- `--extra-protein-trace-backend {none,beam}`

默认值：

- `none`

行为：

- `none`：只跑原有默认流程
- `beam`：先跑原有默认流程，再额外跑新的 protein-only beam 分支

#### 3. 新增 protein-only beam tracing backend

在 `infer/flood_fill.py` 中新增独立 backend：

- `beam_trace_with_edge_dual_protein(...)`

beam state 在 v1 中应包含：

- 当前 ordered protein node path
- used-node mask / set
- current tail node
- cumulative score
- score breakdown：
  - edge score
  - `C-N` geometry score
  - `CA-CA` geometry score
  - optional confidence score
- per-hop edge provenance：
  - `cn_primary`
  - `ca_rescue`
  - `both`

beam 行为：

- 只做单端 forward growth
- path 内部禁止重复使用节点
- 不做 bidirectional growth
- 每轮只保留 top `beam_width` 活跃状态
- 无法继续扩展或达到最大长度时，输出 candidate path

#### 4. 加入 `CA-CA rescue`

在新 beam backend 中，候选邻居来自两个通道：

- primary：现有 `C-N` recall
- rescue：基于 `CA-CA ≈ 3.8A` 的额外候选扩增

规则：

- 使用 `cn_candidates ∪ ca_candidates`
- 按节点去重
- 保留 provenance 标签
- `C-N` 仍然是主信号
- `CA-CA` 只作为 recall expansion，不替代 `C-N`

#### 5. 先导出所有 candidate

第一目标不是最终最优链，而是“候选可见性”。

需要导出的 candidate 信息包括：

- candidate id
- ordered node indices
- path length
- total score
- score breakdown
- per-edge provenance tags
- rescue edge 数量
- overlap metadata placeholder

目标输出文件：

- `postprocess_beam_protein/protein_candidates.json`
- `postprocess_beam_protein/protein_candidates_topk.cif`
- `postprocess_beam_protein/beam_summary.json`

#### 6. 在新分支内部做二阶段处理

raw candidate 导出后，在新分支内部继续做：

1. dedup
2. overlap/conflict graph
3. HMM scoring / reranking
4. optional anchor selection
5. optional iterative retracing

v1 的 conflict policy：

- 两条 protein candidate 只要共享任意 node，就认为冲突

#### 7. HMM rerank 与 iterative refine

复用已有蛋白 HMM machinery，对 candidate 进行注释和 rerank：

- 从 `final_results` 取每个 candidate 的 aa logits
- 对 protein sequence 跑 HMM alignment
- 记录：
  - match score
  - matched sequence id
  - matched residue span
  - aligned coverage
  - exists-in-sequence-mask summary

后续再支持：

- `--beam-protein-trace-iterative-refine`
- 用高置信 candidate 作为 anchors
- 冻结 anchor nodes
- 对剩余未解决节点重新做 beam tracing
- 最多若干轮迭代


## 本次实际改动

### 1. 新增 CLI 开关

文件：

- [src/em3dfold/infer/inferlm_common.py](E:/projects/EM3DFold/src/em3dfold/infer/inferlm_common.py:139)

新增参数：

- `--extra-protein-trace-backend {none,beam}`
- `--beam-width-protein`
- `--trace-max-candidates`
- `--cn-radius-protein`
- `--ca-rescue-radius-protein`
- `--no-beam-protein-trace-dump-candidates`
- `--beam-protein-trace-hmm-rerank`
- `--beam-protein-trace-iterative-refine`
- `--beam-protein-trace-max-refine-rounds`

默认值：

- `extra_protein_trace_backend = none`
- `beam_width_protein = 16`
- `trace_max_candidates = 5000`
- `cn_radius_protein = 2.1`
- `ca_rescue_radius_protein = 4.8`
- candidate dump 默认开启
- HMM rerank 默认关闭
- iterative refine 默认关闭

### 2. 默认推理流程后新增 opt-in beam 支线调用

文件：

- [src/em3dfold/infer/inferlm_common.py](E:/projects/EM3DFold/src/em3dfold/infer/inferlm_common.py:575)

行为：

- 仍然先执行原有 `final_results_align_to_sequence(...)`
- 只有当 `--extra-protein-trace-backend beam` 时，才额外执行：
  - `final_results_align_to_sequence_beam_protein(...)`
- beam 支线返回的信息会附加到 `output_info["beam_protein_output_info"]`
- 原先的 canonical outputs 不会被这个新分支覆盖

### 3. 新增 protein-only beam tracing backend

文件：

- [src/em3dfold/infer/flood_fill.py](E:/projects/EM3DFold/src/em3dfold/infer/flood_fill.py:606)

新增函数：

- `beam_trace_with_edge_dual_protein(...)`

当前实现的核心逻辑：

- 输入为 protein residue dummy atom positions
- 用 residue-level confidence 作为种子和路径打分的一部分
- 使用 `C-N` 邻接半径检索 primary candidates
- 使用 `CA-CA` 半径检索 rescue candidates
- 两类候选取并集并按 provenance 标记：
  - `cn_primary`
  - `ca_rescue`
  - `both`
- 单端 forward growth
- path 内不允许 node reuse
- beam 每轮保留 top `beam_width`
- 达到最大长度或无可扩展节点时输出 candidate

当前 score 由几部分相加组成：

- `log(edge_probability)`
- `C-N` 距离高斯型打分
- `CA-CA` 距离高斯型打分
- local confidence 加分

额外辅助函数也一并加入：

- `_safe_edge_probability(...)`
- `_log_gaussian_score(...)`
- `_collect_protein_beam_candidates(...)`

### 4. 新增 protein-only beam postprocess 分支

文件：

- [src/em3dfold/infer/denovo.py](E:/projects/EM3DFold/src/em3dfold/infer/denovo.py:1049)

新增函数：

- `final_results_align_to_sequence_beam_protein(...)`

这个函数的职责：

- 从同一份 `final_results` 出发
- 复用默认后处理中的必要预处理步骤：
  - residue confidence 计算
  - dummy atom reconstruction
  - protein existence filter
- 只保留蛋白节点
- 完全忽略核酸
- 调用新的 `beam_trace_with_edge_dual_protein(...)`
- 将输出写入：
  - `output_dir/postprocess_beam_protein/`

### 5. 候选导出

当前已经实现以下候选导出：

- `protein_candidates.json`
- `protein_candidates_topk.cif`
- `beam_summary.json`

其中：

- `protein_candidates.json`
  - 保存 raw candidates
  - 保存 processed candidates
  - 保存 dedup removed mapping
  - 保存 conflict map
- `protein_candidates_topk.cif`
  - 将 raw candidates 中分数较高的一部分导出为结构文件，方便可视化
- `beam_summary.json`
  - 汇总 candidate 数量、直方图、rescue 比例、round 信息等

### 6. 已实现的 candidate metadata

当前 candidate 中已包含：

- `candidate_uid`
- `source_round_idx`
- `source_candidate_id`
- `node_indices_local`
- `node_indices_filtered`
- `node_indices_original`
- `path_length`
- `tail_idx_local`
- `tail_idx_filtered`
- `tail_idx_original`
- `score_total`
- `score_breakdown`
- `edge_sources`
- `edge_scores`
- `num_ca_rescue_edges`
- `num_both_edges`
- `overlap_metadata`

`overlap_metadata` 当前包括：

- `duplicate_of`
- `dedup_removed_candidate_ids`
- `conflict_candidate_ids`

### 7. 已实现的二阶段后处理

文件：

- [src/em3dfold/infer/denovo.py](E:/projects/EM3DFold/src/em3dfold/infer/denovo.py:744)

新增了若干 helper，用于 beam candidate 的后处理：

- `_serializable_candidate(...)`
- `_beam_candidate_overlap(...)`
- `_deduplicate_protein_beam_candidates(...)`
- `_attach_protein_beam_conflicts(...)`
- `_summarize_protein_beam_candidates(...)`
- `_rank_protein_beam_candidates(...)`
- `_write_json(...)`

当前规则：

- strict 相同路径会被 dedup
- 高重叠候选可能被压缩
- 低分 subpath 在部分情况下会被折叠
- 任意共享 node 的 candidate 会被标记为 conflict

### 8. 已实现的 HMM candidate 注释与 rerank

文件：

- [src/em3dfold/infer/denovo.py](E:/projects/EM3DFold/src/em3dfold/infer/denovo.py:905)

新增函数：

- `_annotate_protein_beam_candidates_with_hmm(...)`

当前行为：

- 复用已有 `best_match_to_sequences(...)`
- 对每个 protein beam candidate 提取 aa logits
- 可选地附加 HMM 注释
- 支持按 HMM 分数 + coverage 重新排序 candidate

当前注释字段包括：

- `match_score`
- `matched_sequence_id`
- `matched_residue_span`
- `aligned_coverage`
- `exists_in_sequence_mask_summary`
- `hmm_output_match_sequence`

### 9. 已实现的 iterative refine 雏形

文件：

- [src/em3dfold/infer/denovo.py](E:/projects/EM3DFold/src/em3dfold/infer/denovo.py:996)

新增函数：

- `_select_anchor_candidates_for_iterative_refine(...)`

当前实现：

- iterative refine 仍然默认关闭
- 当开启时，可以从候选中选择一批 anchor
- anchor nodes 会被冻结
- 后续轮次只在未冻结节点上继续 beam tracing

当前 anchor 选择逻辑：

- 如果 candidate 有 HMM 注释，则优先用：
  - match score
  - aligned coverage
- 如果没有 HMM 注释，则退回用 beam score
- anchors 之间不能共享节点


## 当前行为总结

### 默认行为

不加新参数时：

- 只运行原有 tracing / HMM / prune 流程
- 所有当前 canonical 输出保持原样
- protein 和 nucleic acid 的现有路径不受影响

### 打开 beam 支线后的行为

加入：

```bash
--extra-protein-trace-backend beam
```

后：

- 先正常生成默认输出
- 再额外生成 `postprocess_beam_protein/`
- 只针对蛋白节点做 beam tracing
- 不会覆盖默认 `output.cif`、`before_prune.cif`、`after_prune.cif`
- 不会触碰 NA tracing 的已有逻辑


## 当前输出文件

beam 支线输出目录：

- `output_dir/postprocess_beam_protein/`

当前关键输出：

- `protein_candidates.json`
- `protein_candidates_topk.cif`
- `beam_summary.json`
- `hmm/`


## 已完成验证

### 1. 语法检查

已通过：

- `py_compile`

涉及文件：

- `src/em3dfold/infer/flood_fill.py`
- `src/em3dfold/infer/denovo.py`
- `src/em3dfold/infer/inferlm_common.py`

### 2. 合成 beam smoke

已在 `em3dfold` 环境下做过一个最小 synthetic import / run：

- 成功 import 新 backend
- 成功生成多条 candidate

### 3. CLI 参数检查

已确认新的 CLI 参数能出现在 help 中：

- `--extra-protein-trace-backend`


## 当前限制

这次实现是 v1，重点是“开出一条平行实验线并先看到候选”，还不是最终最优版本。

当前限制包括：

- 只支持 protein-only beam branch
- 核酸完全不参与这条新支线
- beam 扩展目前只做单端 forward growth
- 暂未做 bidirectional tracing
- dedup / overlap 规则仍然较简单
- raw candidates 仍然可能很多
- `protein_candidates_topk.cif` 只是便于观察，不代表最终筛选结论
- HMM rerank 已接通，但还没有用于强反馈控制 tracing
- iterative refine 只是第一版 anchor-freeze 机制，默认关闭


## 后续建议

### 1. 优先做真实案例验证

重点验证这些场景：

- `C-N` miss，但 `CA-CA ≈ 3.8A` 能 rescue 的蛋白连接
- greedy 只能给一条链、beam 能给多条 plausible 分支的歧义区域
- 高密度重叠候选区的 dedup / conflict 可解释性
- 短碎片 raw candidates 经 HMM 打分后自然掉到后面的情况

### 2. 强化 candidate 诊断信息

可以继续补充：

- per-candidate source seed
- more explicit overlap graph export
- node coverage summary
- rescue-only edge 详细统计
- score normalization / calibration

### 3. 再决定是否让 HMM 进入闭环

后续如果要增强：

- 用 HMM 高分 candidate 作为更强 anchors
- 做多轮 unresolved-node retracing
- 尝试 partial rollback 或 soft-freeze
- 将 chain-level consistency 反馈回 beam ranking


## 相关代码位置

- [src/em3dfold/infer/flood_fill.py](E:/projects/EM3DFold/src/em3dfold/infer/flood_fill.py:606)
- [src/em3dfold/infer/denovo.py](E:/projects/EM3DFold/src/em3dfold/infer/denovo.py:1049)
- [src/em3dfold/infer/inferlm_common.py](E:/projects/EM3DFold/src/em3dfold/infer/inferlm_common.py:139)
- [src/em3dfold/infer/inferlm_common.py](E:/projects/EM3DFold/src/em3dfold/infer/inferlm_common.py:575)


## 使用示例

最小启用方式：

```bash
python -m em3dfold.infer.inferlm_v3x2 \
  --map your_map.mrc \
  --polymer your_polymer.cif \
  --protein-seq your_protein.fasta \
  --output-dir output_beam \
  --extra-protein-trace-backend beam
```

带更多 beam 参数：

```bash
python -m em3dfold.infer.inferlm_v3x2 \
  --map your_map.mrc \
  --polymer your_polymer.cif \
  --protein-seq your_protein.fasta \
  --output-dir output_beam \
  --extra-protein-trace-backend beam \
  --beam-width-protein 32 \
  --trace-max-candidates 10000 \
  --cn-radius-protein 2.2 \
  --ca-rescue-radius-protein 4.8 \
  --beam-protein-trace-hmm-rerank
```


## 备注

本次实现遵循的原则是：

- 不改原有主流程
- 不破坏当前默认输出
- 先把候选空间看清楚
- 再逐步做 rerank、anchor、iterative retracing

也就是说，这个 beam branch 目前的定位是：

- 一个独立的、分析优先的实验性 protein tracing 支线
- 而不是替代当前生产 postprocess 的默认 backend
