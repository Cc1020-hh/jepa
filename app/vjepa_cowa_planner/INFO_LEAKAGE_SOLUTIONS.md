# 信息泄漏问题解决方案

## 问题回顾

当 `use_temporal=True` 时，Planner 的输入 `z_ar` 来自 Predictor 的自回归输出。Predictor 在每一步都接收 **actions** 和 **states**，因此 `z_ar` 隐含了这两类信息。

- **States 泄漏（主要问题）**：`states` 包含车辆在每时刻的绝对位置 (x, y, yaw)，而 GT 轨迹正是由这些 states 推导而来。Predictor 在预测第 k 步时已看到 `states[:, :k+1]`，因此其输出已编码了未来轨迹信息。
- **Actions 泄漏（次要）**：actions 是控制输入，理论上可作为合法条件；但在训练时若模型过度依赖 actions 推断轨迹，可能削弱从视觉中学习的能力。

---

## 方案一：使用 `use_z_context=True`（推荐，零代码改动）

**原理**：用 Encoder 输出 `z_context` 替代 Predictor 输出 `z_ar` 作为 Planner 输入。Encoder 只接收视频帧，不接收 actions/states，因此无信息泄漏。

**配置修改**（在 `planner` 段添加）：

```yaml
planner:
  use_planner: true
  use_temporal: true
  use_z_context: true   # 关键：使用 encoder 输出，避免泄漏
  # ... 其他配置
```

**优点**：配置即可，无需改代码  
**缺点**：失去 Predictor 的时序推理能力，Planner 仅依赖 Encoder 的视觉特征

---

## 方案二：Predictor 不传 States（需改 Predictor）

**原理**：在生成 `z_ar` 时，只传 actions，不传 states（或传零向量）。这样 Predictor 输出不再编码未来轨迹。

**实现步骤**：

1. 在 `ac_predictor.py` 中增加可选参数 `use_states`：

```python
def forward(self, x, actions, states, extrinsics=None, use_states=True):
    # ...
    if use_states:
        s = self.state_encoder(states).unsqueeze(2)
    else:
        s = torch.zeros(B, states.shape[1], 1, D, device=x.device, dtype=x.dtype)
    a = self.action_encoder(actions).unsqueeze(2)
    # ...
```

2. 在 `forward_predictions` 中，调用 predictor 时传入 `use_states=False`（仅用于生成 planner 的 `z_ar` 分支时）。

**注意**：Predictor 若在预训练时依赖 states，直接传零可能使输出质量下降，需要实验验证或重新训练一个「无 states」的 Predictor 变体。

---

## 方案三：仅移除 status_feature 中的 action（缓解 actions 泄漏）

若主要担心 actions 泄漏，可修改 `prepare_status_feature`，不把 actions 放入 status_feature：

```python
# 在 prepare_status_feature 中，将 action_feat 置零
action_feat = torch.zeros(B, 3, device=states.device, dtype=states.dtype)  # 不传 actions
return torch.cat([velocity, acceleration, yaw, xy, action_feat], dim=-1)
```

**注意**：这只影响 Planner 的 status 输入，不影响 `z_ar` 本身。若 `z_ar` 仍由带 states 的 Predictor 生成，states 泄漏依然存在。

---

## 方案四：分阶段训练

1. **阶段 A**：用 `use_z_context=True` 训练 Planner，避免泄漏。
2. **阶段 B**：可选地，在 Planner 收敛后，用 `use_z_context=False` 微调，此时可尝试方案二（Predictor 不传 states）以降低泄漏。

---

## 推荐配置

优先使用 **方案一**：

```yaml
planner:
  use_planner: true
  use_temporal: true
  use_z_context: true   # 关键：使用 encoder 输出
  use_spatial_tokens: true
  # ... 其他保持不变
```

若需要保留 Predictor 的时序信息，再考虑方案二，并配合实验验证效果。
