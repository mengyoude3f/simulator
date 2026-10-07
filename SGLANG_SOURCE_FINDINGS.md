# SGLang PD 分离场景源码核实记录

核实日期：2026-09-16

源码目录：`../sglang-main`

## 适用边界

本地目录没有 `.git` 元数据，源码包也不能给出可靠的发布版本或 commit。因此：

- 下文的“确定”仅表示能够从这份本地源码确定；
- `framework.version` 和 `framework.commit` 继续保持 `unknown`；
- 在部署版本被固定并与本地源码核对前，不能把这些结论声称为实际部署版本的行为。

所有行号均为 `sglang-main/python/sglang/srt/` 下的定位（个别注明其它路径）。

---

## 一、有哪几种推理模式

推理模式是**两个正交维度**，不应混为一谈：调度循环模式（overlap / non-overlap）与批次内容模式（ForwardMode + chunked prefill）。

### 维度 A：调度循环模式

入口 `run_scheduler_process`（`managers/scheduler.py:4498-4525`）按优先级分派：

```
enable_pdmux        → event_loop_pdmux
pp_size > 1         → event_loop_pp
enable_overlap_mlx  → event_loop_overlap_mlx
enable_overlap      → event_loop_overlap      ← 默认
else                → event_loop_normal
```

| 模式 | 含义 | 关键机制 |
|---|---|---|
| **non-overlap** (`event_loop_normal`, `scheduler.py:1520-1551`) | 每迭代内 `run_batch`(GPU forward) 与 `process_batch_result`(CPU 采样/记账/输出) **串行背靠背**；第 N 步 CPU 处理必须等第 N 步 forward 完成。无排队批次。 | `run_batch`→立即 `process_batch_result`(`:1542-1543`) |
| **overlap** (`event_loop_overlap`, `scheduler.py:1554-1625`)（**默认开**） | 第 N 步 GPU forward 与第 N-1 步 CPU 处理**并行**。 | `result_queue` deque 缓存 `(batch.copy(), result)`(`:1556-1604`)；采样结果**延迟一步**经 `FutureMap`(`overlap_utils.py:232`) 解析；forward 里 `future_map.publish(...)`(`:3374`)、`batch.input_ids=None`(`:3421`)，下一步 `resolve_forward_inputs` 从 `future_map.output_tokens_buf` 取真实 token；独立 `forward_stream`/`copy_stream` + `copy_done` 事件；`delay_sample_func`+`launch_batch_sample_if_needed`(`:3551`) 延迟采样；`_apply_war_barrier`(`:1603`) |
| **PP** (`event_loop_pp`, `scheduler_pp_mixin.py:68-176`) | 流水并行，micro-batch 之间 overlap 输出处理与计算；**与 overlap、spec 互斥**（断言 `server_args.py:7630-7633`）。PD 变体：`event_loop_pp_disagg_prefill/decode`。 | — |

**overlap 何时启用/关闭**：
- `self.enable_overlap = not server_args.disable_overlap_schedule and not use_mlx()`（`scheduler.py:348`）。
- 默认 `disable_overlap_schedule=False` → **默认开**（`server_args.py:848-854`）。
- 强制关闭：MPS 设备（`server_args.py:3510`）、`pp_size>1`、pdmux（`:7684`）、no_buffer。
- 逐批临时关闭：`is_disable_overlap_for_batch`（`scheduler.py:1627-1665`）——`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 下**连续两个 prefill 批**（改善 TTFT）、或 spec 需要 grammar 同步时。关闭时先处理上一批再 launch（`:1588-1589`）。

### 维度 B：批次内容模式（ForwardMode + chunked prefill）

`ForwardMode`（`model_executor/forward_batch_info.py:98-196`）：

- `EXTEND` = prefill（整段或单个 chunk）
- `DECODE` = 解码一个 token
- `MIXED` = **chunk prefill 与 decode 融合进一个 forward**
- `IDLE` = DP-attention 下无序列的 worker
- `PREBUILT` = **PD decode 节点专用**：KV 已传输完毕、待开始解码的“假 extend”
- `TARGET_VERIFY` / `DRAFT_EXTEND_V2` = spec decode；`SPLIT_PREFILL` = PD 多路复用；`DLLM_EXTEND` = 扩散 LLM
- 判定：`is_extend()`(`:127`) 覆盖 EXTEND/MIXED/TARGET_VERIFY/SPLIT_PREFILL/DLLM_EXTEND；`is_prefill()` 是其别名。

**chunked prefill**（`chunked_prefill_size`，`server_args.py:714`；`-1` 关闭，`server_args.py:6196-6200`；默认按显存 2048/4096/8192/16384 自适应，`:3910-3964`）：

- 切分判定：`PrefillAdder.add_one_req`（`schedule_policy.py:1001-1180`）。候选 extend 长度 ≤ `rem_chunk_tokens`（或 chunking 关闭）→ **整段跑**；否则 **切 chunk**（`trunc_len` 页对齐），标记 `new_chunked_req`（`:990/:1174`）。`can_run_list` 为空时首个请求即便超限也强制接纳（`:1105-1113`）。
- 跨步续跑：`scheduler.py:3052-3058` 记 `self.chunked_req`、`inflight_middle_chunks+=1`；下一次 `get_next_batch_to_run`(`:2728-2739`) 把它排除出 running batch 合并并 `stash_chunked_request`，由 `add_chunked_req`(`:2951`) 继续；最后一个 chunk 时 finish 并清空（`:4258`）。
- **MIXED 交织**：仅当 `is_mixed_chunk`（=`chunked_prefill_size is not None and enable_mixed_chunk`，`enable_mixed_chunk` 默认 False）、有非空 running batch、无 logprob 时，`_get_new_batch_prefill_raw`(`scheduler.py:3105-3123`) 调 `new_batch.mix_with_running`(`schedule_batch.py:2533-2562`)：`forward_mode=MIXED`，每个 running 请求给 1-token extend，记 `mix_running_indices`。否则 prefill 与 decode 是分开 forward、交替进行、**prefill 优先**。

**批次优先级** `get_next_batch_to_run`（`scheduler.py:2702-2837`）：
1. 杂务：abort 超时、排除 in-flight `chunked_req`（`:2705-2739`）；
2. 把上一个 prefill 批（`last_batch.is_extend()`）过滤完成/chunk 后 `merge_batch` 进 running batch（`:2754-2779`，extend→decode 毕业）；
3. 尝试组新 prefill 批 `get_new_batch_prefill`（`:2794-2796`）；
4. **有 prefill 就先跑 prefill**（`:2811-2813`）；
5. 否则跑 decode（`update_running_batch`，`:2814-2820`）。

---

## 二、请求事件时间线与充要条件

底层依据 `SchedulerReqTimeStats`（`observability/req_time_stats.py:573`）三条流水线：

```
Unified: wait_queue → forward → completion
Prefill: bootstrap_queue → wait_queue → forward → transfer_queue → completion
Decode : prealloc_queue → transfer_queue → wait_queue → forward → completion
```

跨节点门控核心是 **KVPoll 状态机**（`disaggregation/base/conn.py:84-90`），每个 `bootstrap_room` 一份，经 `max()` 单调递进（`Failed` 强制，`common/conn.py:250-265`）：

```
Failed=0  Bootstrapping=1  WaitingForInput=2  Transferring=3  Success=4
```

三处 scheduler 轮询循环均对一组 sender/receiver 做 `poll_and_all_reduce`（跨 attn TP/CP rank 共识），故状态迁移是全 rank 同步的。

---

### PREFILL 节点（KV 发送方 KVSender）

> PD 里请求由 router **分别**发到 prefill 与 decode 两个节点；prefill 负责算 KV 并发送。

| # | 事件 | 触发点 | 充要条件（necessary + sufficient） |
|---|---|---|---|
| 1 | **request_arrival** | `recv_requests`→`process_input_requests`(`scheduler.py:1678`)→`handle_generate_request`(`:2087`)→`_add_request_to_queue`(`:2406`)；`set_scheduler_recv_time`(`schedule_batch.py:1040`) | 请求在 PREFILL 模式下被 scheduler 接收 |
| 2 | **prefill_bootstrap 入队** (`PREFILL_PREPARE`) | `disagg_prefill_bootstrap_queue.add`(`scheduler.py:2417`)；`set_prefill_bootstrap_queue_entry_time`(`:2420`, setter `req_time_stats.py:964`) | 成功 `create_sender`（状态→**Bootstrapping**，`common/conn.py:1004`），未超 KV 容量（`_check_if_req_exceed_kv_capacity`）；`pending_bootstrap=True`，`max_new_tokens` 被强制为 1 |
| 3 | **bootstrap 完成（handshake）** | manager 监听线程 `mooncake/conn.py:1588-1598` | **decode 已把全部 `required_dst_info_num` 份目标 KV 元数据（dst 指针/aux index/decode_prefix_len）发到该 room** → Bootstrapping→**WaitingForInput**；prefill 侧动作时记 `set_bootstrap_done_time` |
| 4 | **bootstrap_queue → waiting_queue** (`PREFILL_BOOTSTRAP`) | `pop_bootstrapped`(`prefill.py:388-397`)；`set_wait_queue_entry_time`(setter `req_time_stats.py:718-737`) | **poll==WaitingForInput 且有空闲 metadata buffer** → `finalize_bootstrap`(`prefill.py:289-306`) 成功（`sender.init(num_pages, buf_idx)`，`pending_bootstrap=False`）。**此处不需 KV cache 分配**。〔乐观路径：`optimistic_prefill_attempts>0` 且非 retracted 且 buffer 可用时，可在 Bootstrapping 就提前入队（`:376-387`），handshake 推迟到 `resolve_waiting_queue_bootstrap`(`:438`)/forward 后 `handle_pending_bootstrap`(`:952`) 校验〕 |
| 5 | **forward_entry** (`PREFILL_WAITING`) | `PrefillAdder`→`set_time_batch(..., "set_forward_entry_time")`(`scheduler.py:3060`, setter `req_time_stats.py:739-756`) | PrefillAdder **成功预留 KV/token-pool 空间**并把请求排进 extend 批（**真正的 KV cache 分配在此**） |
| 6 | **prefill forward 完成** (`PREFILL_FORWARD`) | `process_batch_result_disagg_prefill`→`set_prefill_finished_time`(`prefill.py:650`, setter `req_time_stats.py:785`) | 该请求**最后一个 chunk 的 forward 完成**（`inflight_middle_chunks<=0`，结果已 `copy_done.synchronize`） |
| 7 | **transfer_queue 入队 + 发起 KV send** | `disagg_prefill_inflight_queue.append`(`prefill.py:660`)；`send_kv_chunk`(`:694`, 内部 `sender.send`)；`set_prefill_transfer_queue_entry_time`(`:696`) | prefill 已完成并入 inflight 队列；实际 `send()` 还需 `not pending_bootstrap`（否则挂起等 handshake） |
| 8 | **KV transfer 完成** (`PREFILL_TRANSFER_KV_CACHE`) | `process_disagg_prefill_inflight_queue`→`set_prefill_kv_transfer_finish_time`(`prefill.py:819-826`, setter `req_time_stats.py:976`) | **poll==Success**：最后一个 chunk + aux 已送达**全部** `required_dst_info_num` 目标 rank 且全部子传输成功（`mooncake/conn.py:1455-1457`）；随后 `release_kv_cache` 解锁 tree |
| 9 | **completion** | `set_completion_time`(`prefill.py:836`, setter `req_time_stats.py:878`) | 请求进入 `done_reqs`（终态 Success/Failed）；`stream_output` 回客户端、释放 metadata buffer |

### DECODE 节点（KV 接收方 KVReceiver）

> decode 节点**先于 KV 到达**就收到请求：先预留 KV 槽 + handshake，把目标地址 `send_metadata` 告诉 prefill，再等 KV 落地。三个 staging 队列每轮由 `process_decode_queue` 轮询。

| # | 事件 | 触发点 | 充要条件 |
|---|---|---|---|
| 1 | **request_arrival** | `_add_request_to_queue` DECODE 分支(`scheduler.py:2421-2426`) | 请求在 DECODE 模式下被 scheduler 接收 |
| 2 | **prealloc_queue 入队** (`DECODE_PREPARE`) | `disagg_decode_prealloc_queue.add`(`decode.py:496`, `scheduler.py:2424`)；`set_decode_prealloc_queue_entry_time`(setter `req_time_stats.py:986`) | 创建 `CommonKVReceiver`（状态→**Bootstrapping**，`common/conn.py:1180`）并入 `self.queue`（非 retracted） |
| 3 | **handshake 完成** | `_resolve_pending_reqs`(`decode.py:814`) 里 `receiver.init()`；`_update_handshake_waiters`(`:721-768`) | `init()` 完成 bootstrap → 状态→**WaitingForInput**(`common/conn.py:1217`)，poll 返回后置 `waiting_for_input=True`、记 `set_bootstrap_done_time`(`decode.py:744-746`) |
| 4 | **prealloc_queue → transfer_queue** (`DECODE_BOOTSTRAP`) | `pop_preallocated`(`decode.py:943-1202`)；`set_decode_transfer_queue_entry_time`(`:1196`, setter `req_time_stats.py:994`) | **全部满足**：`waiting_for_input==True`（`:950`）**且** `req_to_token_pool` 有空槽（`:953`）**且** metadata idx 可用（`:956`）**且** KV 页预算够（`:1005-1024`）→ `_pre_alloc`(`:1397`) 分配 req 槽+KV 槽（必要时逐出 radix cache）+ `send_metadata`(`:1175`) 把目标页索引发给 prefill |
| 5 | **transfer 阶段** | `DecodeTransferQueue.pop_transferred`(`decode.py:1858`) 轮询 | receiver 随 prefill 推 KV 进 WaitingForInput→Transferring→Success；未到 Success 留队 |
| 6 | **transfer_queue → waiting_queue** (`DECODE_TRANSFERRED`) | `_commit_transfer_to_req`(`decode.py:1670-1822`)→`set_wait_queue_entry_time`(`:1821`, setter `req_time_stats.py:718-726` DECODE 分支) | **poll==KVPoll.Success**（`:1923`，KV 全写入 + aux buffer 就绪）〔decode HiCache 时另需 restore 非 PENDING，`:1924-1928`〕。**此刻把 prefill 采样出的首 token append 进 `output_ids`**（`:1757-1758`） |
| 7 | **forward_entry** (`DECODE_WAITING`) | `get_new_prebuilt_batch`(`decode.py:2121`)；`set_time_batch(..., "set_forward_entry_time")`(`:2171`, setter `req_time_stats.py:739-750` DECODE 分支) | `waiting_queue` 非空 **且**有空闲批次槽（`num_not_used_batch = min(pool.size, max_running_requests) − curr_batch_size > 0`，`:2138-2149`） |
| 8 | **prebuilt / fake output** (`DECODE_FAKE_OUTPUT`) | `prepare_for_prebuilt`(`decode_schedule_batch_mixin.py:31`, 置 `forward_mode=PREBUILT`)+`process_prebuilt`；`set_decode_prebuilt_finish_time`(`batch_result_processor.py:90`, setter `req_time_stats.py:1019`) | 用 `PREBUILT` 假 extend **只填批次元数据、`out_cache_loc` 指向已传来的 KV 槽，不跑 prefill 计算**（`_run_batch_prebuilt` 返回空结果）；把 prefill 首 token 作为第一个 decode bonus token 回放。〔1-token 请求走 `DECODE_QUICK_FINISH`/`set_quick_finish_time` 快速结束〕 |
| 9 | **decode loop → completion** | prebuilt 批合并进 running batch（`decode.py:2100-2107`）→`update_running_batch`→常规自回归 decode（`event_loop_*_disagg_decode`） | 每步正常 forward；finish 时 `release_kv_cache` |

---

## 三、两侧握手对齐（关键跨节点因果）

事件是**跨节点交错**的，成对来看：

- **decode #4（`send_metadata`）** 是 **prefill #3（bootstrap 完成→WaitingForInput）** 的**充要触发**——prefill 必须收齐 decode 的目标地址才能进 waiting_queue。故正常时序：decode 先 prealloc 并 `send_metadata` → prefill 才 bootstrap done。
- **prefill #7-8（`send` + Success）** 驱动 **decode #5-6（Transferring→Success）**。
- **首 token 由 prefill 采样**（prefill #6 后 `output_ids.append`），随 KV 一起经 metadata buffer 传给 decode，在 **decode #6** 落地——因此 decode 的“第一次 forward”是 `PREBUILT` 假批次，不重算 prefill。

---

## 四、事件名 → 源码 setter 速查表

| 概念事件 | RequestStage | setter（`req_time_stats.py`） |
|---|---|---|
| request_arrival | — | `set_scheduler_recv_time` (:639) |
| prefill_prealloc(=bootstrap)_start | PREFILL_PREPARE | `set_prefill_bootstrap_queue_entry_time` (:964) |
| prefill_queue_start (bootstrap→wait) | PREFILL_BOOTSTRAP | `set_wait_queue_entry_time` (:718) |
| prefill_forward_start | PREFILL_WAITING→FORWARD | `set_forward_entry_time` (:739) |
| prefill_forward_end | PREFILL_FORWARD | `set_prefill_finished_time` (:785) |
| kv_transfer_start | — | `set_prefill_transfer_queue_entry_time` (:972) |
| kv_transfer_end | PREFILL_TRANSFER_KV_CACHE | `set_prefill_kv_transfer_finish_time` (:976) |
| completion | — | `set_completion_time` (:878) |
| decode_prealloc_start | DECODE_PREPARE | `set_decode_prealloc_queue_entry_time` (:986) |
| decode_prealloc_end (→transfer) | DECODE_BOOTSTRAP | `set_decode_transfer_queue_entry_time` (:994) |
| decode_transfer_end (→wait) | DECODE_TRANSFERRED | `set_wait_queue_entry_time` (:718, DECODE) |
| decode_forward_start | DECODE_WAITING | `set_forward_entry_time` (:739, DECODE) |
| decode_prebuilt_end | DECODE_FAKE_OUTPUT | `set_decode_prebuilt_finish_time` (:1019) |

> 追踪开关：`--enable-trace` + `--otlp-traces-endpoint`；`SGLANG_TRACE_LEVEL` 0~3；运行时 `curl .../set_trace_level?level=N`（见 `docs_new/docs/references/production_request_trace.mdx`）。
