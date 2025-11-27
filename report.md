# Pruner-Zero: Evolving Symbolic Pruning Metric from scratch for Large Language Models
GitHub: https://github.com/pprp/Pruner-Zero
## 1. 论文总结
尽管大型语言模型（LLM）具有卓越的功能，但由于其庞大的规模，它们面临着部署挑战。修剪方法会降低权重子集以加速，但其中许多方法需要重新训练，这非常昂贵且计算量大。最近，后训练修剪方法引入了新的度量，使得LLM的修剪无需重新训练。然而，为了有效地识别上级剪枝度量，作者开发了一个使用遗传编程搜索符号剪枝度量的自动框架。特别地，还设计了一个包含现有剪枝度量的精细搜索空间来发现潜在的符号剪枝度量，并提出了一种相反的操作简化策略来增加种群的多样性，这样，基于搜索结果，本文研究了符号剪枝度量与剪枝后性能之间的关系，总结了一些原则，并在LLaMA和LLaMA-2上进行了大量的语言建模和zero-shot任务的实验，实验结果表明，我们的PrunerZero比SOTA后训练剪枝方法具有上级性能。
## 2. 论文创新点
1. **自动化的符号化剪枝度量搜索框架**
 - **首创性**：Pruner-Zero是首个利用遗传编程（Genetic Programming, GP）从零开始自动搜索符号化剪枝度量（Symbolic Pruning Metric, SPM）的框架。它通过进化算法动态生成和优化剪枝度量，无需人工设计复杂的剪枝规则。
 - **全面的搜索空间**：该框架设计了一个统一且全面的搜索空间，涵盖了现有的剪枝度量（如权重、梯度等），并引入了多种基本操作（如加法、乘法、归一化等），能够重构和优化现有的剪枝方法。
2. **对立操作简化策略（Opposing Operation Simplification, OOS）**
- **优化搜索效率**：OOS策略通过识别和消除符号树中对立的操作（如exp和log、sqrt和sqr等），减少搜索空间中的冗余，提高搜索效率和剪枝度量的多样性。
- **提升剪枝性能**：该策略不仅简化了符号表达式，还通过减少冗余操作，使得最终搜索到的剪枝度量更加简洁且性能更优。
3. **无需重新训练的高效剪枝方法**
- **无需权重更新**：Pruner-Zero在剪枝过程中无需对模型权重进行更新或重新训练，显著降低了计算成本和资源需求，尤其适用于大规模语言模型。
- **快速评估**：通过在LLaMA-2-7B模型上进行快速后剪枝评估（每次评估耗时不到5分钟），Pruner-Zero能够在短时间内找到高效的剪枝度量。
4. **广泛的实验验证和性能提升**
- **超越现有方法**：在LLaMA和LLaMA-2模型上，Pruner-Zero在多种剪枝比例（如50%、4:8、2:4）下均优于现有的剪枝方法（如SparseGPT、Wanda等），在语言建模和零样本任务中表现出更低的困惑度（Perplexity）和更高的准确率。
- **适用于多种模型架构**：Pruner-Zero不仅在LLaMA系列模型上表现出色，还成功应用于其他模型（如Tiny-LLaMA和OPT），证明了其通用性和泛化能力。
## 3. 流程图
<p align="center">
<img src="https://raw.githubusercontent.com/pprp/Pruner-Zero/main/.github/images/pruner-zero-main-figure.png" width=100% height=100% 
class="center">
象征式搜索得到的树：
```json
{
  "data": "mul",
  "left": {
    "data": "abs",
    "left": {
      "data": "mul",
      "left": {"data": "W"},
      "right": {"data": "W"}
    }
  },
  "right": {
    "data": "mms",
    "left": {"data": "G"}
  }
}
```

公式：
\[
Score_i = |W_i|^2 \times \frac{G_i - \min(G)}{\max(G) - \min(G) + \varepsilon}
\]

对应代码：
- 加载符号树：`lib/gptree.py` (`GPTree.load_tree`)
- 剪枝实现：`lib/prune.py` 中 `prune_pruner_zero`
- 梯度生成：`lib/gradient_computation.py`
- 稀疏度检查：`check_sparsity(model)` (`lib/prune.py`)
- 主入口：`main.py` 行 85–87

## 2. 安装与环境

最小环境：
```bash
conda create -n pruner_zero python=3.9
conda activate pruner_zero
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.28.0 accelerate==0.18.0 datasets==2.11.0 \
            sentencepiece wandb bitsandbytes==0.42.0
# 可选：
pip install deepspeed peft==0.7.1
```

或使用完整：
```bash
pip install -r requirements.txt
```

## 3. 数据与校准

梯度校准：
```bash
CUDA_VISIBLE_DEVICES=0 python lib/gradient_computation.py \
  --nsamples 128 \
  --model meta-llama/Llama-2-7b-hf \
  --llama_version 2 \
  --task gradient \
  --save_path data/grad_llama2_7b.pt
```

评测数据：WikiText 用于困惑度，Zero-Shot 任务包括 boolq, rte, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa。

LoRA 微调数据：C4 数据集自动通过 datasets 加载。

## 4. 关键依赖说明

| 依赖 | 功能 |
|------|------|
| torch | 基础深度学习框架 |
| transformers | 加载与操作 LLM |
| accelerate | 多设备调度 |
| datasets | 数据加载（WikiText/C4 等） |
| sentencepiece | LLaMA 分词 |
| bitsandbytes | 低比特权重支持 |
| wandb | 实验日志 |
| deepspeed | 大模型分布式/推理优化 |
| peft | LoRA 微调 |

## 5. 剪枝与评测示例

非结构化 50%：
```bash
python main.py \
  --model decapoda-research/llama-7b-hf \
  --prune_method pruner-zero \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --nsamples 128 \
  --gradient_path data/grad_llama_7b.pt \
  --json_tree data/best_tree.json \
  --save out/llama_7b/unstructured/pruner-zero/
```

结构化 2:4：
```bash
python main.py \
  --model decapoda-research/llama-7b-hf \
  --prune_method pruner-zero \
  --sparsity_ratio 0.5 \
  --sparsity_type 2:4 \
  --gradient_path data/grad_llama_7b.pt \
  --json_tree data/best_tree.json \
  --save out/llama_7b/2-4/pruner-zero/
```

LLaMA-2：
```bash
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method pruner-zero \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --gradient_path data/grad_llama2_7b.pt \
  --json_tree data/best_tree.json \
  --save out/llama2_7b/unstructured/pruner-zero/
```

零样本评测：
```bash
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method pruner-zero \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --gradient_path data/grad_llama2_7b.pt \
  --json_tree data/best_tree.json \
  --eval_zero_shot \
  --save out/llama2_7b/eval/pruner-zero/
```

保存模型：
```bash
python main.py \
  --model decapoda-research/llama-7b-hf \
  --prune_method pruner-zero \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --save_model checkpoints/llama7b_pruned_50 \
  --save out/llama_7b/unstructured/pruner-zero/
```

LoRA 微调：
```bash
CUDA_VISIBLE_DEVICES=0 python lora_ft/finetune_lm.py \
  --model_name_or_path checkpoints/llama7b_pruned_50 \
  --config_name decapoda-research/llama-7b-hf \
  --dataset_name c4 \
  --num_train_epochs 1 \
  --block_size 1024 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --do_train --do_eval \
  --max_train_samples 30000 \
  --max_eval_samples 128 \
  --learning_rate 1e-4 \
  --overwrite_output_dir \
  --output_dir lora_out/llama7b_pruned_lora/
```

## 6. 常见问题

| 问题 | 解决 |
|------|------|
| Tokenizer 报错 | 确保 transformers=4.28.0 + sentencepiece 安装；参见官方 issue 建议。 |
| 显存不足 | 减少 nsamples；使用 8bit；降低 batch；不做零样本评测。 |
| 结构化断言失败 | 保持 `--sparsity_ratio 0.5` 与 `--sparsity_type` 二者匹配。 |

| LoRA 训练慢 | 降低 `max_train_samples` 或提升 batch（显存允许）。 |
