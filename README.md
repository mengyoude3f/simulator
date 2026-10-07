# PD Throughput Simulator

面向 **PD 分离**（prefill / decode 分离）LLM 推理服务的离散事件仿真器。

与拟合输入输出关系的性能模型不同，本仿真器从**机制上**复刻调度器：prefill 线与 decode 线两套并存的状态机、overlap 调度循环固有的一步延迟、chunked prefill 的页对齐切分与跨迭代续跑、KV cache 的逐 chunk 流式传输。物理层补充了框架本身交给硬件、而容量规划无法回避的一环：KV 传输在**发送方出口网卡 / 接收方入口网卡 / 集群 fabric** 三级共享资源上的带宽争夺，由单一分配器统一求解。

状态机的每个相位与转移条件依照 [`SGLANG_SOURCE_FINDINGS.md`](SGLANG_SOURCE_FINDINGS.md)（对 SGLang 源码的核实记录）设计。

## 设计原则

1. **机制与数值解耦** —— 所有耗时与容量经统一的 `CostModel` 接口取得，调度逻辑本身不含任何数值。默认实现是线性的、显式的仿真输入，**不是实测数值**；接入真机压测数据无需改动调度逻辑。
2. **带宽中央分配** —— 一条流的速率是三级资源约束的**联合**结果，必须由全局唯一的分配器求解。
3. **不变量可被违反、违反即失败** —— 资源守恒、token 守恒、字节守恒、P/D 相位顺序均为运行时断言。

## 快速开始

本目录是一个 Python 包，无第三方依赖（仅标准库）。

```bash
# 在本目录下
python run_example.py            # 回放 example_trace.jsonl（真实生产 trace 60 条）
python run_example.py compare    # 受控对照：只换 workload，其余配置相同
python run_example.py demo       # 内置合成负载

# 在父目录下
python -m pd_throughput_simulator_v5.demo
python -m pytest -q pd_throughput_simulator_v5/tests    # 52 项
```

`python run_example.py` 的输出：

```
=== PD Throughput Simulation Report ===
requests: 60  completed: 60
window: [0.0000, 587.8626) s
TTFT  p50/p90/p99: 41.646ms/363.149ms/1590.718ms
TBT   p50/p90/p99: 0.600ms/0.600ms/0.700ms
E2E   p50/p90/p99: 260.694ms/837.375ms/3649.539ms
request throughput: 0.102 req/s
output token throughput: 55.399 tok/s
KV chunk 流总数: 70
KV 传输总量: 17.68 GB
```

## 编程接口

```python
from pd_throughput_simulator_v5 import (
    PDSimulator, Scenario, ClusterConfig, CapacityConfig, WorkloadConfig,
    SimpleAnalyticCostModel, build_report, format_report,
)

scenario = Scenario(
    cluster=ClusterConfig(num_prefill_replicas=1, num_decode_replicas=2),
    capacity=CapacityConfig(chunked_prefill_size=8192, max_running_requests=128),
    workload=WorkloadConfig(num_requests=64, arrival_rate_rps=120,
                            input_tokens=1024, output_tokens=64),
    cost_model=SimpleAnalyticCostModel(congestion_alpha=0.3),
)
states = PDSimulator(scenario).run()
print(format_report(build_report(states, scenario.metrics)))
```

负载有两种来源：生成式（请求形状 + 到达过程，用于受控实验），或 `WorkloadConfig.trace_path` 指向 jsonl 做 trace 回放（逐请求读 input / output / cached token 数与到达时刻，整体平移使最早请求为 t=0）。

## 架构

四层，每层只依赖其下方：

| 层 | 模块 | 职责 |
|---|---|---|
| 装配 | `simulator.py` `scenario.py` `workload.py` `trace_loader.py` `metrics.py` | 请求轮询分配与驱动；配置；负载生成 / trace 回放；TTFT / TBT / E2E 与吞吐统计 |
| 状态机 | `prefill_node.py` `decode_node.py` `kv_transfer.py` `request.py` | 两侧各三个阶段队列与一个运行集；连接两者的传输会话 |
| 物理 | `cost_model.py` `network.py` | 可插拔成本模型；全集群唯一的流量中央分配器 |
| 引擎 | `sim_clock.py` | 单调时钟；按（时刻，插入序）稳定排序的优先队列；带事件总预算的主循环 |

**双线状态机。** 路由层把同一请求并行发给一个 prefill 副本和一个 decode 副本，两条线唯一的耦合是每请求一份的传输会话（五状态、取最大值单调递进、失败态粘滞）。相位推进有三处跨线前置条件：decode 必须先预留 KV 槽位并把目标页索引告知 prefill；首 token 由 prefill 采样后随 KV 送抵 decode；末块携带辅助数据到达后 decode 才开始正式 forward。decode 的首次 forward 是假批——不重算 KV，直接回放 prefill 采样的首 token，首 token 延迟归属给 prefill 与传输。

**overlap 循环。** 每个副本一次只跑一个 forward，每个迭代边界依次：零成本记账推进相位转移 → 选出当前批（此时上一批结果尚未提交）→ 提交上一批结果（产 token / 采样首 token / 发起 KV 发送）→ 启动当前批。

**chunked prefill 与逐 chunk 传输。** 长请求被切成页对齐的块跨迭代续跑，同一时刻至多一个续跑请求。每个 chunk 一完成就发送对应 KV，在 fabric 中生成一条独立的流，传输与计算重叠。prefix cache 命中免掉的是**计算**而不是**传输**，命中的 cached 随第一个块一并发出。

**三级带宽。** 每条流同时占用发送方出口、接收方入口、集群 fabric。分配器以渐进水填充求解速率：每轮取各资源在未定流上的均分份额的全局最小值作为瓶颈，冻结停在该瓶颈的流并扣除其实际占用，剩余容量回到后续轮次。被某一级限死的流只扣掉它用得掉的部分，余量让给共享同一资源的其他流。每轮至少冻结一条流，故算法必然终止。资源容量可设非线性拥塞 `C(n) = C_base / (1 + α(n-1))`，`α = 0` 退化为与并发无关的固定上限。

## 建模范围

**已建模**

| 位置 | 内容 |
|---|---|
| prefill | forward 耗时随 token 数与批大小变化；单副本单 forward 串行；每次 forward 的 token 预算；队首阻塞 |
| decode | forward 耗时随批大小增长；单副本单 forward 串行；并发请求上限；KV 池容量；请求槽位与元数据缓冲 |
| 网络 | 三级共享带宽；渐进水填充分配；可选的并发相关容量非线性拥塞 |
| 握手 | 元数据缓冲可用性；两侧相位推进的前置条件 |

**明确不建模**：non-overlap 与流水线并行；prefill 与 decode 的融合批次；投机解码（decode 每迭代固定 1 token）；请求中止与回撤；decode 到客户端的响应延迟。

**物理近似**（三者均可经成本模型接口覆盖）：渐进水填充的流量分配；容量的非线性拥塞曲线；KV 传输的 replica 级聚合。

**保真度缺口**

| 缺口 | 后果 |
|---|---|
| decode 侧无 prefix 去重（真实 SGLang 可只传后缀） | 高估 KV 传输负载 |
| KV 传输是 replica 级聚合，非 rank↔rank | rank 负载不均时低估瓶颈 |
| prefill 侧不统计 cached 占用的页 | 低估 prefill 侧显存压力 |
| 无显存与 CPU 算力字段 | 无法推出某型号显卡的并发上限 |
| decode 预分配保留整个输出预算 | 高估长输出请求的 decode 显存占用 |

## 结论的可外推性

绝对的毫秒数、GB 数、req/s 数依赖未经校准的成本系数，**不可外推**。仅基于机制、不依赖系数取值的结论可外推，例如：

- 未饱和负载下，长尾来自**形状异构**而非到达异构；用均值形状构造合成负载会同时高估中位、低估尾部。
- chunk 粒度存在 U 形最优，与闭式解 `c* = √(L·c_fix·B/k)` 吻合。
- 副本必须加在瓶颈侧；decode 受限时加 prefill 副本不仅对吞吐无效，还会放大 decode 运行批、拖慢迭代与槽位周转，反而使 TTFT 变差。
- decode 入口容量高于到达率时，KV 传输是纯加性延迟，不经排队放大。
- 三级带宽必须中央求解，逐资源独立均分再取最小值会系统性低估可用容量。

## 后续方向

实机校准成本模型；显存与 KV 页的映射；decode 侧显存带宽吞吐模型；decode 侧 prefix 去重；rank 级 KV 传输。
