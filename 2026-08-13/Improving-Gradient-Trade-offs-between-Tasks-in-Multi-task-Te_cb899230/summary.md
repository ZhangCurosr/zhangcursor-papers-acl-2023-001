---
title: "Improving-Gradient-Trade-offs-between-Tasks-in-Multi-task-Te"
source: https://aclanthology.org/2023.acl-long.144.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:24:56"
---

# 论文速读：Improving-Gradient-Trade-offs-between-Tasks-in-Multi-task-Te

## 一句话总结
本文提出 GetMTL 框架，通过多任务文本分类的主目标（平均损失）邻域约束寻找梯度更新方向，有效缓解任务间梯度冲突，实现所有任务性能的同步提升。

## 研究问题与动机
1. **多任务梯度冲突导致性能退化**：在 MTC 场景下，共享参数使不同任务的梯度方向不一致甚至相互抵消，引发部分任务过拟合或欠拟合，整体表现劣于独立学习（STL）。
2. **现有 Pareto 方法缺乏目标导向性**：MGDA、TchebycheffAdv 等方法仅在 Pareto 前沿上寻找任意均衡解，无法保证所有任务同时改善，可能牺牲主任务或平均性能。
3. **缺少针对平均损失的针对性优化**：现有方法忽略多任务文本分类的真实目标是最小化平均损失 $\mathcal{L}_0$，导致学到的权衡解在损失空间中分布发散，泛化能力受限。

## 核心贡献（创新点）
1. 提出 GetMTL 梯度权衡多任务学习框架，将更新方向搜索空间约束在主目标平均梯度 $g_0$ 的邻域锥内。与已有工作的本质区别在于从“追求任意 Pareto 解”转向“锚定主目标附近的特定可控权衡解”。
2. 将梯度冲突缓解形式化为极大极小（MaxMin）优化问题，并通过拉格朗日对偶与 Sion 定理推导得到闭式更新方向 $d^*$。与已有工作的本质区别在于利用对偶转换规避了高维非凸搜索，直接输出理论可解的最优更新向量。
3. 给出 GetMTL 在凸损失与 L-Lipschitz 梯度假设下的严格收敛性证明，并推导步长 $\mu$ 的上界。与已有工作的本质区别在于为该邻域约束优化提供了可验证的理论收敛保障，而非仅依赖经验调参。
4. 在 Amazon Review 与 Topic Classification 双基准上进行系统实验，并结合任务方差演化与权重轨迹可视化揭示冲突缓解机理。与已有工作的本质区别在于构建了“理论证明-数值验证-几何可视化”的完整证据链，证实邻域约束对全体任务的同时增益效应。

## 方法详解
- **基础架构**：采用硬参数共享（Hard Parameter Sharing）设计，共享编码器为 TextCNN（卷积核尺寸 3, 5, 7），各任务头为单层全连接网络+Softmax，参数记为 $\{\theta^{sh}, \theta^1, \dots, \theta^T\}$。
- **任务梯度与 Pareto 初始方向**：第 $i$ 个任务的梯度为 $g_i = \nabla \ell_i(\theta)$。首先利用 MGDA 在梯度凸包 $\mathcal{CH} = \{G\beta \mid \beta \in S^T\}$ 中求解最小范数点，得到下降方向 $d_{des} = \sum_{i=1}^T \beta_i g_i$，并生成缩放后的任务梯度 $g_{\beta_i} = \beta_i g_i$。
- **邻域约束的 MaxMin 问题**：定义平均梯度 $g_0 = \frac{1}{T}\sum_{i=1}^T g_i$ 作为主目标锚点。构造优化问题：在约束 $\|d - g_0\| \leq \varepsilon g_0^\top d$（锥约束，$\varepsilon \in (0,1]$ 控制搜索半径）下，最大化最坏任务方向的梯度收益 $\min_i \langle g_{\beta_i}, d \rangle$，同时要求 $-g_0^\top d \leq 0$ 保证下降性。
- **对偶求解与参数更新**：构造拉格朗日函数 $L(d, \lambda, w) = g_w^\top d - \lambda(\|d-g_0\|^2 - \varepsilon^2(g_0^\top d)^2)/2$，利用 Sion 极小极大定理交换 $\max_d$ 与 $\min_{\lambda,w}$ 顺序。对 $d$ 求导令其为零，得到最优方向闭式解 $d^* = \frac{g_w + \lambda^* g_0}{(1-\varepsilon^2 \|g_0\|^2)\lambda^*}$，其中 $\lambda
