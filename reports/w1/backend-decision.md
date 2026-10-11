---
title: W1-09 融合后端审计与决策
created: 2026-10-11
type: report
status: decision
---

# W1-09 融合后端审计与决策

**决策：A1/A2 及第二个月 A_base 不复用 SVIn，改用 GTSAM 4.3.0 短窗口原始测量级离线后端。** OpenVINS 继续作为独立的 A0 参照，不作为 A1 的输入边。

## SVIn 审计（vendor/SVIn，main @ fcda546）

| 检查项 | 实际证据 | 结论 |
| --- | --- | --- |
| 分支与 ROS 版本 | 仅有 `main`（ROS2 Jazzy）；README 第 10–13 行写明 ROS2 下“Sonar and depth modes will not work and are disabled by default” | 声呐、深度功能在当前主分支不可用 |
| 配置 | `config/*.yaml` 全部 `isSonarUsed: false`、`isDepthUsed: false` | 没有可直接复现的声呐或深度配置 |
| 声呐观测模型 | `okvis_ceres/src/SonarError.cpp:146` 构造声呐点 `(range·cos(heading), range·sin(heading), 0)`；残差为测量距离与 `|T_WS 位置 − 附近视觉地标均值|` 之差 | 默认仰角为 0，而且只是单波束测距（数据话题 `/imagenex831l/range`，剖面声呐）；不是前视成像声呐的斜距方位模型 |
| 与详案的一致性 | A v2 第 2 节、B v2 第 3.2 节都要求仰角未观测、不把斜距当水平距离 | 直接冲突；修补需要重写因子和关联，等同于新建后端 |
| 深度因子 | 依赖同样被关闭的自定义消息路径 | 不计为可用 |

旧 `ros1` 分支本机没有检出；即使检出，声呐模型仍是上面的剖面声呐假设，结论不变。

## 选择 GTSAM 的依据

- `gtsam 4.3.0` 有 cp312 manylinux wheel（2026-10-11 `pip download` 成功），与本机 Python 3.12 一致；Ubuntu 仓库另有 `libgtsam-dev 4.2.0`，C++ 版本需要时再定。
- 原始 IMU 预积分、相机投影、先验和自定义因子都是现成组件；声呐斜距方位因子按 A v2 第 2 节自写，几何函数复用 `team/uwbench/uwbench/sonar.py`，与 B 的候选共用一套模型。
- 短窗口、批量离线求解足以支撑 M1 规格要求的白化残差、雅可比与信息矩阵导出；实时性不是 A1 的验收项。

## 约束与下一步

- 安装在独立的 `.venv-a`，不改动锁定的 `.venv-eval`；版本与哈希写入 `locks/`。
- 不把 OpenVINS 输出当视觉边再重复加入同一份 IMU（A v2 3.1）；需要状态级融合时单独命名为工程基线。
- W3-02 先做已知对应的因子检查：真值零残差、数值差分雅可比、缺测与离群；再接前端。
