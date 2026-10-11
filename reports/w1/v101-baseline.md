---
title: W1 EuRoC V1_01_easy 数据核验与 OpenVINS 公共基线执行记录
created: 2026-10-10
updated: 2026-10-10
type: plan
status: execution-record
tags: [acoustic-optic-fusion, slam, camera, calibration, rov]
---

# W1 EuRoC V1_01_easy 数据核验与 OpenVINS 公共基线执行记录

本页接续 [[plans/acoustic-optic-w1-environment-audit-2026-10|WSL 环境准备]]，记录 [[plans/acoustic-optic-month1-execution-checklist-2026-10|W1-02/03/04]] 的实际操作、失败与结果。运行工作区为 `/home/steve/uw-ao-lab`；日期沿用本轮工作日 2026-10-10，机器日志使用 UTC（运行跨至 2026-10-11 UTC）。本页是公共双目惯性里程计工程复现，不代表水下声光融合方法效果、闭环 SLAM 或产品验收。

本轮结论：**W1-02 空目录复建、W1-03 数据入库及 W1-04 公共 A0 的 R1 运行验收已完成。** 有效结果来自 0.5× 离线回放：修正参考下 APE RMSE 0.177510 m，1 s 平移 RPE RMSE 0.067166 m。1× 回放出现严重漂移，尚未通过实时或稳健性验收；本次仅有一次有效完整半速运行，不能称为多次复现实验均通过。W1-05 至 W1-09 与 G0 仍未整项完成。

## 数据来源与完整性

通过 [ETH EuRoC 官方入口](https://projects.asl.ethz.ch/datasets/euroc-mav/)定位 [Research Collection 数据记录](https://doi.org/10.3929/ethz-b-000690084)。来源 archive 为 vicon_room1.zip，元数据大小 6,042,263,426 字节。对 [官方 bitstream](https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content) 使用 HTTP Range 读取 ZIP64 目录，再提取 V1_01_easy 的 bag 与 ASL ZIP，未下载其他序列。每个响应验证 Content-Range，解压验证 archive member CRC32，计算本地 SHA256，复制进 WSL 后再次计算 SHA256。

| 文件 | 字节数 | 成员 CRC32 | 本地 SHA256 |
| --- | ---: | --- | --- |
| V1_01_easy.bag | 2,130,599,434 | 001e8ccc | 6dc6192fac63dd0a05ba745548b41fe8cae14724168a98865a81d37e681bbc81 |
| V1_01_easy.zip | 1,149,702,102 | 1abe6a0d | a920fe5b5e69a6ad19b32f1cfaf90ac2f59d45dfd1ca18cc9722e07684ba45fa |

ASL ZIP 的 5,856 个条目完成解压 CRC 检查，并检查路径不越出目标目录。本次没有取得整个外层 archive，因此**未核验外层整包 MD5**；本地 SHA256 是下载后生成的指纹，不冒充发布方提供的 SHA256。官方元数据、成员目录和下载回执保存在 data/public/euroc/provenance 与 Windows 临时 staging；正式登记为 manifests/euroc_v101.yaml。元数据使用权标记为 In Copyright - Non-Commercial Use Permitted，原始数据不随团队代码仓库提交。

### ROS1 → ROS2 转换与内容审计

使用 rosbags 0.11.6 转为 ROS2 Jazzy typestore、metadata version 9、MCAP。转换保留全部六个话题，实际算法回放仅选择双目与 /imu0；Vicon 和辅助飞控话题不输入算法。

| 话题 | 消息数 | 消息采样间隔中位数 |
| --- | ---: | ---: |
| /cam0/image_raw | 2,912 | 0.050 s |
| /cam1/image_raw | 2,912 | 0.050 s |
| /imu0 | 29,120 | 0.005 s |
| /fcu/imu | 29,123 | 约 0.005 s |
| /fcu/motor_speed | 29,123 | 约 0.005 s |
| /vicon/firefly_sbx/firefly_sbx | 14,629 | 约 0.010 s |

两路相机 header 从 1403715273262143000 到 1403715418812143000 ns，跨度 145.55 s；/imu0 结束于 1403715418857143000 ns，跨度 145.595 s。两路相机时间戳完全相同，三个算法输入话题均严格递增，无重复或倒序。

审计逐条读取转换前后消息，对记录时间、header 和数据内容生成按话题哈希并比较；ROS2 不存在的 ROS1 Header.seq 明确排除。两路图像均为 752×480 mono8；与 ASL 的首、中、尾共六幅 PNG 逐像素比较一致。ASL 全部相机、IMU 行按顺序与 bag 对应，时间戳最大差值 104 ns，采用不超过 128 ns 的舍入容差；主 IMU 六轴数值差为 0。此容差仅适用于 ASL CSV 对照，ROS1/ROS2 转换本身要求内容完全一致，没有修改数据时间戳。

转换产物 V1_01_easy_ros2.mcap 为 2,129,404,503 字节，SHA256 为 329eeff0797d5c3fcf89a4bf468bce8eca49abae80b538e89fccee2892dfdfb1。转换后又用 ROS 原生反序列化检查首条 IMU 与左右图像，和 rosbags 解码的六轴数值、图像字节一致。完整审计为 reports/w1/v101-data-audit/audit.json。

### 标定与真值

三份官方 EuRoC YAML 在运行前复制到 configs/euroc_v101 并提交冻结。左右相机内参、畸变、分辨率、T_imu_cam 与 ASL sensor.yaml 相符，数值检查容差 1e-12；IMU 200 Hz、噪声与随机游走参数也与原始标定一致。没有用本机水下标定替换公共参数。

[OpenVINS 官方说明](https://docs.openvins.com/gs-datasets.html)指出 V1_01 原始姿态参考不准确，作者用 IMU 和 Vicon 重新优化。因此在评分前约定：主参考使用锁定上游仓库 ov_data/euroc_mav/V1_01_easy.csv；辅助参考使用本次 ASL 的 state_groundtruth_estimate0/data.csv，二者均报告。修正参考使用了 IMU，属于社区复现参考，不作为与输入完全独立的水下精度真值。两份文件哈希见 manifest。

## 环境、构建与版本

Ubuntu 24.04.4 / ROS2 Jazzy / 系统 Python 3.12，离线评分使用 .venv-eval。OpenVINS 上游固定为 69488123ed9362dd44b6f28e7f4680abbff1442b；本地原有补丁仅在两份 ROS 头文件中增加四处 .hpp/.h 包含兼容，不修改估计器计算。锁定补丁位于 locks/openvins-jazzy.patch。

在新的 build_w1_clean、install_w1_clean 和 log_w1_clean 中，按顺序、并行度 2、RelWithDebInfo、DISABLE_MATPLOTLIB=ON 重建 ov_core/ov_init/ov_msckf/ov_eval，四包成功，耗时 8 分 11 秒；新 install 的可执行入口和 launch 参数查询通过。运行入口仍使用先前同源构建的 ros_ws/install；空目录复建作为 W1-02 的可重建验证，二者不混写。

团队代码根目录已建立独立 Git，分支 codex/w1-public-baseline，无远程地址。scripts、team 源码、冻结 YAML、DDS XML、依赖与补丁锁、数据 manifest 和 Markdown 报告均版本化；vendor、ros_ws 上游 checkout、数据、结果 bag、SDK、venv、构建与缓存不进入团队 Git。上游提交号与补丁组合保持可追溯。

## 运行与评分口径

官方 subscribe.launch.py，use_stereo=true、max_cameras=2、RViz 关闭；使用官方 KLT/在线标定参数，未以本序列评分调参。先确认输出发布者，再启动并确认 poseimu/odomimu 录制器，最后回放 cam0、cam1、imu0。采用 header 时间，不启用 /clock 或 use_sim_time，不向算法提供真值。每次运行有唯一目录，失败记录不覆盖。

位姿验收检查非空、有限值、单位四元数、严格递增时间、初始化、首末时间与间断。输出覆盖率按相机时间槽 ±25 ms 匹配，仅表示有输出；不能用它代替有精度门槛的定位可用率。

APE 使用完整匹配轨迹 SE(3) 对齐，禁止尺度拟合；真值关联阈值 10 ms、固定偏移 0。RPE 按实际输出时间选择 1 s 和 5 s 配对，容差 25 ms，使用 evo 的标准相对位姿误差公式，分别报告平移 m 与旋转 degree；保存所有配对索引、实际间隔和配对数。时间配对的五项测试通过，覆盖空轨迹、重复时间、缺帧与已知相对误差等情况。

当前 evo 1.37.1 的 delta_unit 不支持 s。原手册中的秒单位命令已更正；20 帧/100 帧只作为均匀 20 Hz 示例，本次真实秒间隔评分使用 scripts/evaluate_openvins.py，不把固定帧数误记为秒。

## 实际运行结果

| run_id 后缀 | 团队提交 | 回放 | 实际结果 | 判定 |
| --- | --- | --- | --- | --- |
| r01 | 2366bf0 | 未播放 | Fast DDS 发现/录制订阅未就绪，启动保护停止 | 失败，保留日志 |
| r02 | 4d7e246 | 1× | Cyclone DDS 可发现，IMU 可接收，图像未有效送达，无有效位姿 | 失败；播放结束不等于算法通过 |
| r03 | 363c797 | 1× | 图像与位姿恢复，2,800 个 pose，但修正参考 APE RMSE 232.502201 m | 精度失败；关闭时另有子进程 -11 |
| r04 | ca5f554 | 0.5× | 同一算法与通信配置，2,800 个 pose，完整评分成功 | R1 离线运行通过；保留实时与关闭问题 |

目录均为 runs/w1_v101_a0_rXX。r04 对 r03 的算法参数、数据、标定和 DDS 配置完全相同；ca5f554 仅新增原生解码核对脚本，正式运行的变化是回放速度。结果支持“存在回放速度/时序依赖”，还不能据此断言唯一根因是丢帧、CPU 性能、DDS 或上游线程竞态。

### r04 输出完整性

| 项目 | 实测值 |
| --- | --- |
| 回放墙钟时间 | 295.086 s，0.5×；不是算法实时耗时 |
| 初始化日志 | successful initialization；初始化计算约 0.0003 s（这是初始化求解时间，不是等待首个位姿时间） |
| 第一位姿相对首帧 | 5.600871 s |
| poseimu | 2,800 条；首 1403715278863013744 ns，末 1403715418812819719 ns |
| odomimu | 22,001 条；首 1403715278912142992 ns，末 1403715418857142925 ns |
| 数值及时间 | 两个输出话题均有限、四元数模长约 1、时间严格递增 |
| pose 最大相邻间隔 | 0.051832 s；首个位姿之后无超过 0.075 s 的间断 |
| 相机时间槽覆盖 | 2,800 / 2,912 = 96.154%；112 个前置槽未输出，不能删除初始化分母 |
| 尾部时间差 | 最后 pose 比最后相机 header 晚 0.677 ms；保留在线时间偏移效果 |
| 参考匹配 | 修正参考 2,783 / 2,800；原始参考 2,781 / 2,800；未匹配的尾端样本不进入精度计算，仍保留在输出覆盖统计 |

这里的输出间断不是精度失败区间。r03 同样有 96.154% 输出覆盖，却严重漂移，因此不能声称“无间断即未失效”。r04 后段 APE 存在明显峰值，下表和全程图均保留；W2 的精度门槛/失效定义冻结前，不追加事后成功率门槛。

### r04 精度：同时报告两份参考

| 指标 | OpenVINS 修正参考（主） | ETH ASL 原始参考（辅助） |
| --- | ---: | ---: |
| 平移 APE RMSE / m | 0.177510 | 0.187257 |
| 平移 APE 中位数 / m | 0.128300 | 0.143491 |
| 平移 APE P95 / m | 0.393691 | 0.394024 |
| 平移 APE 最大值 / m | 0.824121 | 0.819916 |
| 1 s 平移 RPE RMSE / m | 0.067166 | 0.080456 |
| 1 s 旋转 RPE RMSE / degree | 0.303290 | 0.537023 |
| 5 s 平移 RPE RMSE / m | 0.154566 | 0.237304 |
| 5 s 旋转 RPE RMSE / degree | 0.590398 | 2.085941 |
| 1 s / 5 s 配对数 | 2,763 / 2,683 | 2,761 / 2,681 |

主参考的实际估计时间间隔为 0.995387–1.001303 s 和 4.996495–5.001734 s；匹配真值间隔恰为 1/5 s。参考选择对旋转 RPE 影响明显，不能省略真值来源。evo CLI 的 APE 与 Python API 使用相同配对和 SE(3) 对齐协议，结果 ZIP 与配对明细随 run 保存。

![r04 半速基线：SE3 对齐轨迹与全程平移误差](../assets/w1-openvins-v101/trajectory-ape-r04.png)

图中保留了末段约 0.82 m 的误差峰值。该结果证明数据到轨迹、评分、证据归档的链路可运行，不证明达到 OpenVINS 论文精度或水下定位指标。

## 遇到的问题、处理与剩余风险

| 问题 | 证据与处理 | 当前状态 |
| --- | --- | --- |
| 老数据地址不可用、WSL 直连超时 | 经 Windows 访问迁移后的 ETH Research Collection，按官方 ZIP64 成员下载并双端校验 | V1_01 已解决；WSL 直连网络仍未修复 |
| ASL 文件名不能由 bag 时间戳直接拼接 | 首次审计失败；改为 CSV 按序匹配文件名，核验最大 104 ns 的差异及全部行数，容差 128 ns | 已解决，不重写时间戳 |
| 辅助 IMU 被误合并入主 IMU 审计数组 | 第二次审计出现长度不一致；改为只把 /imu0 与 ASL imu0 比较，辅助话题继续独立做全量转换核验 | 已解决；两次失败目录保留 |
| Fast DDS 本地发现不稳定 | r01 录制准备超时；LOCALHOST、静态对端、UDP/回环白名单对照仍不可靠 | 基线改用项目本地 Cyclone DDS；未声称已修复系统 Fast DDS |
| 能发现但收不到图像 | r02 可接收 IMU、无图像/位姿；采用 1200 B MaxMessageSize、1000 B FragmentSize、最小 4 MB socket 接收缓冲 | r03/r04 图像链路已恢复；几项一起调整，不能单独归因某一项 |
| WSL 接收缓冲上限太小 | 实测 net.core.rmem_max=212992；临时调整为 16777216，未写 sysctl 持久配置 | 重启后复现前检查；入口已增加不足时停止提示 |
| 1× 运行严重漂移 | r03 APE 232.502 m，0.5× r04 为 0.178 m；未调估计器参数 | 实时化未通过。后续记录算法实际接收的每个 IMU/双目时间戳，检查队列、线程及重复运行差异 |
| 关闭阶段警告/异常 | r04 在主动 SIGINT 后有 class_loader 警告、节点码 -2；r03 另出现 -11。ros2 launch 自身仍返回 0 | 尚未完成干净退出验收。运行器已追加解析节点退出码，-11 等异常不再被 launch 的 0 掩盖；历史 run.json 不回写 |
| evo 秒间隔命令不支持 | 当前版本不接受 delta_unit=s | 已修订手册并实现基于时间戳的 RPE，五项测试通过 |

Cyclone DDS 安装在 local/cyclone/opt/ros/jazzy，采用 rmw_cyclonedds_cpp 2.2.4（ROS 官方仓库构建 20260911.095700）和 Cyclone DDS 0.10.5，附带三个 iceoryx 2.0.6 包；均按官方 HTTPS Packages 索引 SHA256 核对后解包。旧 APT 索引中的 rmw 包构建已下线，最终实际下载版本及 URL/哈希见 locks/cyclone-debs.json。没有替换 /opt/ros/jazzy 系统包。

DDS 使用回环 lo、禁用 multicast、静态 peer 127.0.0.1、ROS_DOMAIN_ID=76。配置为 configs/cyclonedds-local.xml；运行通过 scripts/ros-cyclone-run.sh 注入项目本地库，不加载 Conda。

## 复现入口

以下按已执行的 r04 命令整理，仅把新 run_id 改为 r05 以避免覆盖现有证据。必须先提交团队代码/配置变更；运行器拒绝脏工作区。已有数据通过 manifest 和 audit 校验后可直接使用，不重复下载。

```bash
cd /home/steve/uw-ao-lab
git status --short
sudo /usr/sbin/sysctl -w net.core.rmem_max=16777216
RUN=w1_v101_a0_r05
bash scripts/ros-cyclone-run.sh /usr/bin/python3 scripts/run_openvins.py   --lab "$PWD" --bag "$PWD/data/public/euroc/V1_01_easy_ros2"   --run-id "$RUN" --rate 0.5
.venv-eval/bin/python scripts/evaluate_openvins.py   --run "$PWD/runs/$RUN"   --audit "$PWD/reports/w1/v101-data-audit/audit.json"   --gt-original "$PWD/data/public/euroc/V1_01_easy_asl/mav0/state_groundtruth_estimate0/data.csv"   --gt-corrected "$PWD/ros_ws/src/open_vins/ov_data/euroc_mav/V1_01_easy.csv"
```

仅当 run.json、estimator/recorder/player.log、输出 bag、evaluation/metrics.json 与轨迹图均存在且相互一致，才评审该次结果。脚本输出 status=scored 仅表示成功计算指标，不代表精度通过；首先检查初始化、覆盖、APE/RPE 全程及所有失败运行。

新机器复建按 README、locks/repositories.json、locks/eval-requirements.txt 和 Jazzy 补丁进行。原始下载脚本依赖锁定的 archive-entries.json；首次下载前将 locks 中的 archive-entries.json、eth-item.json、eth-files.json 复制到 staging。已有原始文件的恢复采用 prepare_euroc.py --resume，会重新核验哈希并保留此前审计目录。下载器拒绝覆盖已经存在的正式 bag/ZIP，重复执行前先判断回执与文件状态。

## 交付物与 W1 后续

| 位置（相对 /home/steve/uw-ao-lab） | 内容 |
| --- | --- |
| scripts/ | 下载、入库、全量审计、原生解码核对、隔离运行、实际秒间隔评分及测试 |
| configs/euroc_v101/ | 三份冻结算法 YAML；文件哈希保存在每次 run.json |
| configs/cyclonedds-local.xml、scripts/ros-cyclone-run.sh | 实际通过的 WSL 本地通信配置与入口 |
| manifests/euroc_v101.yaml | 数据大小、来源、SHA256、转换版本、标定、双真值登记 |
| locks/ | 上游 SHA、依赖、Jazzy 补丁、DDS 包回执、构建指纹、临时 sysctl 前后值及下载元数据 |
| reports/w1/baseline-summary.json | 四次运行的精简指标、退出码、原始证据哈希，纳入 Git |
| reports/w1/v101-baseline.md | 本报告镜像，纳入代码 Git |
| runs/w1_v101_a0_r01 至 r04/ | 完整日志、输出录制；r03/r04 另含 TUM、evo ZIP、完整配对指标、图片；本地保存，不进 Git |
| data/public/euroc/ | 原始、ASL 解压、ROS2 MCAP 与来源元数据；本地保存，不进 Git |

知识库保存报告、轨迹图、更新后的清单/缺口/导航。sources/projects/2026-10-10-w1-v101 中另有精选 JSON 与脚本镜像；受知识库现有白名单影响，这些非 Markdown 原始文件不随 wiki Git 提交，脚本的正式版本由独立代码仓库负责。

W1-02/03/04 已按本页限定验收勾选。剩余 [[plans/acoustic-optic-w1-missing-inputs-2026-10|SDK、D0、水下标定与时间模型]]、独立参考方案、B 的完整几何链、SVIn 后端审计仍需继续。下一步先把半速流程作为离线调试入口；实时化另做输入接收审计与同配置多次运行，关闭问题单独修复，不提前进入声光融合创新方法的效果比较。
