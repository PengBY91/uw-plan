---
title: W1 OpenVINS 1× 回放漂移修复与回归验收
created: 2026-10-10
updated: 2026-10-10
type: plan
status: execution-record
tags: [acoustic-optic-fusion, slam, camera, calibration, rov]
---

# W1 OpenVINS 1× 回放漂移修复与回归验收

本页接续 [[plans/acoustic-optic-w1-openvins-v101-baseline-2026-10|r01–r04 公共基线记录]]，对应 [[plans/acoustic-optic-month1-execution-checklist-2026-10|首月 W1 执行清单]]。代码工作区为 `/home/steve/uw-ao-lab`，OpenVINS 源码为其 `ros_ws/src/open_vins`。工作日记为 2026-10-10，机器日志使用 UTC，跨至 2026-10-11。本页只验收公共双目惯性基线的工程修复，不代表水下融合精度、闭环 SLAM 或实机实时性能。

## 结论与证据边界

**1× 严重漂移已修复，r09/r10 两次完整运行通过预设工程回归。** 两轮均接收全部 29,120 条 IMU、全部 2,912 对双目并完成更新，零缺失、重复、乱序或双目错配；修正参考 APE RMSE 为 0.044367/0.044367 m，1 s 平移 RPE RMSE 为 0.012527/0.012527 m。两轮均输出 2,800 条 pose、覆盖 96.154%，无 >75 ms 的初始化后输出间断，干净退出且无 class_loader 警告。

定位到的主要问题是 IMU 输入丢失，而不是数据转换或标定数值被改坏。旧配置在 200 Hz IMU 上使用 best-effort、KEEP_LAST 深度 5，仅容纳约 25 ms 的到达消息；实际回调存在超过这一时间的停顿。修补线程生命周期后保留深度 5 的对照 r05 仍丢失 43.84% 的 IMU，APE 为 150.87 m；同一二进制仅加深队列到 1000 后，r06 的 APE 降至 0.04413 m。r06 仍有分散于序列中的 45 条 IMU 未进入算法，因此不能仅凭精度恢复宣告修复通过。

代码还发现异步线程按引用捕获回调局部 `message`、后台线程 detach、ImageTransport 生命周期过短，以及运行器向进程组发送 SIGINT 造成 launch 与子进程重复收信号的问题。本轮修复这些问题，但没有分别消融其精度影响，不能声称它们各自造成了多少漂移。接收缺口与队列对照直接测得；best-effort 下剩余缺口的具体 UDP 包丢失位置没有抓包定位。

## 固定条件与验收协议

在 r05/r06 之前将 `reports/w1/replay-fix-protocol.md` 提交。r06 未通过零丢样后，先提交协议补记，再开始可靠传输试验 r07；r07/r08 先后暴露旧触发时刻与图像晚到的尾帧问题，分别提交补记和修复，再运行 r09/r10。整个过程不放宽门槛、不删除失败运行。

| 固定项 | 执行口径 |
| --- | --- |
| 数据 | 已全量核验的 V1_01_easy ROS2 MCAP，算法只输入 cam0、cam1、imu0；没有输入 Vicon 或辅助 IMU |
| 输入总量 | 每路相机 2,912 条，双目时间戳完全对应；主 IMU 29,120 条、200 Hz |
| 算法与标定 | configs/euroc_v101 三份 YAML 哈希不变；未通过本次评分调估计器、噪声或标定参数 |
| 回放与环境 | 1.0×、header 时间；Cyclone DDS、lo、静态 peer、同一 XML、接收缓冲上限 16 MiB；无 RViz |
| 准确度 | 完整匹配轨迹 SE(3) 对齐，无尺度拟合；关联容差 10 ms、偏移 0；实际时间戳 1/5 s RPE，配对容差 25 ms |
| 参考 | OpenVINS 修正参考为主，ETH ASL 原始参考为辅助；二者始终同时保留 |
| 控制变量 | r05/r06 使用同一二进制，仅深度 5→1000；r07 新增可靠性开关后重编译，深度 1000 不变；r08 刷新已喂入 IMU 时间仍缺尾帧；r09 增加相机/IMU 双侧通知的持久工作线程，r09/r10 使用相同提交与二进制 |

工程回归门槛：两次完整 1× 回放均满足全部主 IMU、全部双目输入按源顺序进入算法，零重复、零错配、全部双目对进入更新；位姿输出覆盖率 ≥95%，首个输出后无 >75 ms 的间断；修正参考 APE RMSE ≤0.20 m、1 s 平移 RPE RMSE ≤0.08 m；所有进程干净退出、无 class_loader 警告。精度界限参照此前半速结果制定，是当前故障的回归门槛，不是产品承诺。

## 实际运行与输入审计

| 运行 | 1× 配置 | IMU 收到/应收 | 双目收到/更新 | 修正 APE RMSE / m | 1 s 平移 RPE / m | 验收 |
| --- | --- | --- | --- | ---: | ---: | --- |
| r05 | best-effort / 5 | 16,354/29,120 | 2,912/2,912 | 150.871559 | 6.410043 | 失败：12,766 条 IMU 缺失、严重漂移 |
| r06 | best-effort / 1000 | 29,075/29,120 | 2,912/2,912 | 0.044130 | 0.013805 | 失败：45 条 IMU 缺失 |
| r07 | reliable / 1000 | 29,120/29,120 | 2,912/2,911 | 0.051243 | 0.012809 | 失败：尾帧未更新，旧触发时刻 |
| r08 | 同 r07 + 最新 IMU 时间 | 29,120/29,120 | 2,912/2,911 | 0.044367 | 0.012527 | 失败：尾帧未更新，相机晚到 |
| r09 | 同 r08 + 双侧通知持久线程 | 29,120/29,120 | 2,912/2,912 | 0.044367 | 0.012527 | 全部通过 |
| r10 | 与 r09 相同 | 29,120/29,120 | 2,912/2,912 | 0.044367 | 0.012527 | 全部通过，重复验证 |


r05 中 IMU 回调墙钟间隔 P95 为 51.10 ms、最大 101.52 ms，超过 25 ms 的间隔有 2,923 个；输入样本时间最大间隔 55 ms。双目 2,912 对全部进入、全部处理，却只收到 16,354 条主 IMU。这解释了为什么“图像和位姿条数完整”仍可能伴随严重漂移。

r07 的最后一个更新线程由时间戳 1403715418812143000 ns 的 IMU 触发，先处理前一对相机。在这次视觉更新进行时，最后九条 IMU 已进入算法，但线程仍按启动时的时间戳判断下一对相机是否可处理，导致最后一对留在队列。最后一对图像与触发 IMU 同时间，存在正的在线相机–IMU 时间偏移，需要稍后的 IMU；它们实际已经收到，却没有被旧触发时刻的判断使用。修复后每处理一对相机前读取“已喂入 IMU 的最新时间”原子值，并重算相机时间界限；没有伪造尾部 IMU、追加真值或在评分时补轨迹。

r08 进一步验证：仅刷新时间上界仍不够。最后一对双目回调的 steady clock 为 3818938257428 ns，最后一条 IMU 回调为 3818937959144 ns，相机晚约 0.298 ms 进入回调；IMU 触发线程已经没有后续通知。因此最终改为一个持久、可 join 的工作线程，使用 condition variable 和互斥保护的 pending 标志，由 IMU 喂入与图像入队两侧通知；处理期间到达的通知保留下来，避免忙碌时直接丢弃唤醒。相机入队后释放队列锁再通知，工作线程等待时释放信号锁；shutdown 在信号锁内设置停止标志并唤醒、join。

上游 ROS2 订阅没有显式划分 callback group，因而落在默认互斥组。图像回调获取 `camera_queue_mtx`，而更新线程持有此锁进行视觉更新，存在阻塞同组 IMU 回调的路径。实际停顿可能还包含里程计可视化、系统调度等；本轮没有完成逐函数 profiler，不把所有停顿归因于某一函数。加大队列解决观测到的输入积压，不能保证任意负载下都不会溢出。

输入审计点嵌入**算法自身**的 IMU 回调、双目同步回调以及相机更新入口，不使用独立观察节点收到的数量代替算法输入。每条记录保留 header 整数纳秒与 steady clock；内存积累后在工作线程 join 后写 CSV，避免每条回调同步写盘。主 IMU 和双目 header 必须精确匹配；核心相机时间经过 double 秒转换，更新匹配允许 256 ns 的数值舍入误差，不移动源时间戳。

审计报告包含缺失时间戳、重复、乱序、错配、更新覆盖及等待时间。`callback_to_update_wait_s` 是双目回调入口到更新入口的墙钟等待，不含更新计算，也不是传感器采样到输出的端到端延迟。开启审计可能影响调度，本页结论限定于这一审计配置。

## 修复实现与兼容性

| 位置 | 改动及目的 |
| --- | --- |
| ROS2Visualizer.cpp / .h | IMU history depth 参数化；可选 reliable；为输入/更新增加轻量 trace |
| 同上 | 消除对回调局部变量的悬空引用；初版按值捕获，最终版逐帧读取已喂入 IMU 的时间上界，同时解决旧触发时刻遗漏尾帧 |
| 同上 | 拥有并 join 更新/图像线程；更新使用单个持久线程和 pending 条件变量，处理 IMU 与相机双侧通知，停止标记防止后台继续访问已销毁对象 |
| 同上 | ImageTransport 与发布者共同存活，按成员销毁次序先释放发布者再释放其 transport |
| run_subscribe_msckf.cpp | executor 退出后停止并 join 线程，再做最终输出、销毁对象和 ROS shutdown |
| subscribe.launch.py | 暴露 imu_history_depth 与 imu_reliable；裸 launch 保留 best-effort 默认，以兼容只提供 best-effort 的实机驱动 |
| scripts/openvins/run_openvins.py | 公共 bag 运行器明确默认 reliable、depth=1000；只给 launch 发送一次 SIGINT；记录代码提交、二进制和共享库哈希、参数、子进程异常 |
| 同上 | 团队仓库和 OpenVINS 源码均要求干净 Git 状态；唯一 run_id，禁止覆盖历史结果 |
| scripts/openvins/audit_input_trace.py | 逐条核对算法接收与更新，检查缺失、重复、顺序、双目错配 |
| scripts/openvins/check_replay_acceptance.py | 对已保存 run、输入审计和评分文件应用预设门槛，写 regression-acceptance.json；失败返回非零 |

可靠订阅要求发布端也提供 reliable。本次回放端实测 reliable、深度 10，算法端 reliable、深度 1000；实际端点信息保存在每次运行的 imu-qos.txt。实机驱动若仅 best-effort，必须先检查 QoS 兼容性，再决定驱动发布策略与订阅配置；不能直接照搬公共回放配置后把不匹配误判为设备无数据。

## 重复验证的精度与完整性

| 指标 | r09 修正参考 | r09 原始参考 | r10 修正参考 | r10 原始参考 |
| --- | ---: | ---: | ---: | ---: |
| APE RMSE / m | 0.044367 | 0.055103 | 0.044367 | 0.055103 |
| APE P95 / m | 0.069264 | 0.086551 | 0.069264 | 0.086551 |
| APE 最大值 / m | 0.092986 | 0.102006 | 0.092986 | 0.102006 |
| 1 s 平移 RPE RMSE / m | 0.012527 | 0.044151 | 0.012527 | 0.044151 |
| 1 s 旋转 RPE RMSE / degree | 0.176636 | 0.470972 | 0.176636 | 0.470972 |
| 5 s 平移 RPE RMSE / m | 0.037253 | 0.167271 | 0.037253 | 0.167271 |
| 5 s 旋转 RPE RMSE / degree | 0.452118 | 2.020336 | 0.452118 | 2.020336 |

| 完整性与运行时间 | r09 | r10 |
| --- | ---: | ---: |
| pose 输出数 | 2800 | 2800 |
| 相机槽覆盖率 | 96.154% | 96.154% |
| 首 pose 距首帧 / s | 5.600874 | 5.600874 |
| pose 最大间隔 / s | 0.050421 | 0.050421 |
| 回放墙钟 / s | 148.023818 | 148.025424 |
| 回调入口→更新入口等待 P95 / s | 0.004452 | 0.005201 |
| 回调入口→更新入口等待最大 / s | 0.033575 | 0.034429 |

回放墙钟含播放器启动与调度，不是单帧推理耗时；不能据此宣称实机端到端实时性。

![r09 完整 1× 轨迹与全程 APE](../../runs/w1_v101_a0_r09/evaluation/trajectory_ape.png)

![r10 完整 1× 轨迹与全程 APE](../../runs/w1_v101_a0_r10/evaluation/trajectory_ape.png)

修正参考由 OpenVINS 使用 IMU/Vicon 优化，不是与输入完全独立的水下测量真值。APE 全轨迹对齐也不能替代在线绝对定位准确度或端到端实时性测量。输出覆盖分母包含初始化前的相机槽，不通过删掉初始化阶段提高覆盖率。

## 复现步骤

以下在 WSL Ubuntu 内运行，假定已恢复并核验数据和项目本地 Cyclone。先按 `locks/repositories.json` checkout 原始上游提交，再按 `locks/openvins-replay-fix.json` 顺序应用 Jazzy 兼容补丁与回放修复补丁；提交本地源码后构建。两个补丁已在临时目录从原始上游重放，重建出的五个修改文件与实际提交逐字节一致。

```bash
cd /home/steve/uw-ao-lab
git status --short
git -C ros_ws/src/open_vins status --short
bash scripts/env/build_openvins.sh
sudo /usr/sbin/sysctl -w net.core.rmem_max=16777216
RUN=w1_v101_a0_r11
bash scripts/env/ros_cyclone_run.sh /usr/bin/python3 scripts/openvins/run_openvins.py --lab "$PWD" --bag "$PWD/data/public/euroc/V1_01_easy_ros2" --run-id "$RUN" --rate 1.0 --imu-history-depth 1000 --imu-reliability reliable
.venv-eval/bin/python scripts/openvins/audit_input_trace.py --bag "$PWD/data/public/euroc/V1_01_easy_ros2" --run "$PWD/runs/$RUN"
.venv-eval/bin/python scripts/openvins/evaluate_openvins.py --run "$PWD/runs/$RUN" --audit "$PWD/reports/w1/v101-data-audit/audit.json" --gt-original "$PWD/data/public/euroc/V1_01_easy_asl/mav0/state_groundtruth_estimate0/data.csv" --gt-corrected "$PWD/ros_ws/src/open_vins/ov_data/euroc_mav/V1_01_easy.csv"
.venv-eval/bin/python scripts/openvins/check_replay_acceptance.py --run "$PWD/runs/$RUN"
```

运行前选未使用的 run_id。原始数据、完整日志、trace、输出 bag、评分与图片均保留在 runs，不提交大文件。`status=scored` 只表示评分完成；只有完整输入审计及 regression-acceptance.json 全部通过，才算一次回归通过。此处 sysctl 为临时设置，重启 WSL 后需重新检查。

## Git 与交付证据

| 仓库/锁 | 分支与提交 | 含义 |
| --- | --- | --- |
| 团队根仓库 | codex/w1-public-baseline；执行提交 `f8116d608064745d3148f3b16e15b05118028afa` | r09/r10 实际脚本、配置、补丁与协议版本 |
| OpenVINS 源码 | codex/w1-ros2-replay-fix；最终 `ac543cd427ba880b00582abac45e01dbc51cc23b` | 队列、可靠性、线程生命周期、尾帧与唤醒修复 |
| 上游原始代码 | 69488123ed9362dd44b6f28e7f4680abbff1442b | 官方源码基准 |
| Jazzy 兼容提交 | 49d30a527af5fa580bde80e824278138ad2e2dfa | 头文件包含兼容 |
| locks/openvins-replay-fix.json | 完整补丁、应用顺序与 SHA256 | 从原始上游重建当前源码 |
| locks/openvins-replay-validated-build.json | 最终构建指纹 | 实际可执行程序、共享库与 YAML 哈希 |

实验后报告与精选结果另行提交；实验执行提交不随归档改写。团队根仓库尚未配置远程，本轮为本地 Git 提交，没有推送上游。

可执行程序及共享库哈希、各轮完整提交号、参数、冻结 YAML 哈希见各 run.json；精选指标与原始证据 SHA256 见版本化的 reports/w1/replay-fix-summary.json。团队根仓库用补丁与锁记录嵌套源码，OpenVINS checkout 自身也有独立提交；不是只在 wiki 保存脚本副本。

验证包含九项输入审计/实际时间 RPE 测试、修复后的 C++ 编译、原始上游加补丁的重建一致性，以及表中所有真实回放。验收脚本已拒绝 r05/r06/r07/r08，避免把“精度通过但输入有缺口或尾帧未更新”误当成功。

## W1 后续与参考

公共基线修复后，继续 [[plans/acoustic-optic-w1-missing-inputs-2026-10|D0、采购 SDK、时间模型与标定缺口]]，以及 [[plans/acoustic-optic-ab-first-eight-weeks-2026-10|A/B 八周计划]] 中尚未完成的 B 几何链、SVIn 后端审计和独立参考方案。W1-05 至 W1-09、G0 不因本次回放通过而自动勾选。

- [ROS2 Jazzy callback groups 官方文档源码](https://github.com/ros2/ros2_documentation/blob/jazzy/source/How-To-Guides/Using-callback-groups.rst)：默认互斥组、组内并发约束。
- [ROS2 Jazzy QoS 官方文档源码](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst)：KEEP_LAST、depth、reliable/best-effort 与兼容性。
- [OpenVINS 官方数据说明](https://docs.openvins.com/gs-datasets.html)：V1_01 修正参考背景；数据转换与标定核验证据见前序公共基线记录。
