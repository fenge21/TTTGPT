# TTT 理论优势验证 - 最终报告

硬件: NVIDIA RTX 5060 Ti 16GB | PyTorch eager (fp32) | 全部可复现脚本在本目录

## 已验证的优势 (vs nanoGPT)

### 1. 时间复杂度 O(n) vs O(n²) -- 验证通过
log-log 拟合指数 (time ~ n^p):
- GPT attention: p = 1.33 -> 2 (小规模 Flash Attention 常数主导, 大 n 趋向平方)
- TTT-Linear:    p = 0.90 ~= 1 (纯 Python 循环下已呈线性)

增长倍率 (seq 4096->8192): attention 2.9x, TTT 1.9x

### 2. 常数流式推理状态 -- 验证通过
继续解码所需携带的状态:
- Attention: KV 缓存 = 2*L*n*d*2B, 1M 上下文 = 4 GB
- TTT: 固定 W = L*dh^2*4B = 64 KB (65536x 更小)

### 3. 长上下文利用 -- 方法论验证通过, 简化实现未达标
任务: 文档内唯一 motif 复现 (间隔 80-240 token),
度量: 已见 motif 召回 NLL 相对首见基线的增益。

最终等预算结果 (8000 iters, bs12, 最优检查点选择):
| 模型 | gain@80 | gain@240 | 备注 |
|------|---------|----------|------|
| GPT (attention) | +3.01 | +3.31 | 完美归纳头行为 (阳性对照) |
| LSTM (向量状态) | +0.12 | +0.08 | 固定向量无法任意复制, 符合理论 |
| TTT-Linear (简化版) | -0.09 | -0.04 | 未形成可迁移复制回路 |

## 关键教训 (复现 TTT 的实现陷阱)

1. 因果性: dual form 若先更新后输出, chunk 内早期位置泄漏后期信息
   (症状: IID 噪声 NLL=17, 不可能值) -> 必须先输出后更新
2. 训练/评估分布一致性: 任何只在训练开启的随机 mask 都会造成严重偏移
3. 记忆化捷径: 小型重复语料 (<=2500 docs) 下, 展开内循环的模型倾向
   "记忆文档身份"而非学习复制机制; 验证损失从 iter~1000 起单调恶化。
   官方用数十亿 token Pile 规避了此问题。
4. 归纳头形成是相变: 需要(重复频率 x 步数)超过阈值, 对超参极其敏感

## 本轮实现的优化 (ttt_layer_opt.py)

- 两阶段 chunked online gradient: 顺序快照状态 + 并行块内一阶修正
  y_t = W_c q_t + eta * sum_{j in c, j<t} v_j (k_j . q_t)
- 可学习初始状态 W0 (论文 Sec 2.7 强调的稳定性关键)
- 视图 LayerNorm + lr clamp + 平均梯度 (数值稳定三件套)
- causality_test.py: 因果性自动回归测试 (PASS)

## 与官方实现的差距 (未来工作)

官方 (ttt-lm-jax / 自定义内核) 具备而本实现缺少:
1. 精确逐 token 在线 GD 的闭式 dual form (非一阶近似)
2. 融合 CUDA/Triton 内核 (消除 Python 循环开销)
3. 每 token 学习率 eta(x) = eta_base * sigmoid(theta_lr . x)
4. 内循环 f 的 LN+f 残差结构
5. 十亿级独特 token 训练语料

结论: TTT 的复杂度/状态优势是架构固有的 (已验证);
质量优势依赖系统级优化兑现 (论文在 8k+ 上下文反超 Transformer)。
