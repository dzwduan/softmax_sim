#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMT + Tensor Core + Softmax Engine 模拟器（tile 粒度 + Online Softmax）

新增特性：
- Online Softmax 状态 (m_i, l_i, O_i) 的抽象建模：
  - 对每个 (warp, row) 维护一份在线 softmax 状态，按 tile 顺序更新。
  - 虽然不做数值计算，但明确顺序依赖，体现在线 softmax 的流水方式。
- 更真实的 SIMT warp 调度：
  - 简单的 round-robin warp scheduler。
  - 同一 cycle 允许对 Tensor Core / Softmax 发射多条指令（可配置）。
- 美化输出：
  - 对结果做表格化打印。
  - 时间线对齐，并标注执行单元名。
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple


# ==========================
# 枚举 & 基本数据结构
# ==========================

class OpType(Enum):
    """指令类型：按 tile 粒度抽象"""
    TC_QK = auto()          # Tensor Core 做 QK^T
    SOFTMAX_ENGINE = auto() # Softmax Engine tile softmax（在线）
    SOFTMAX_SFU = auto()    # 用 SFU/ALU 实现的 softmax（baseline）
    TC_PV = auto()          # Tensor Core 做 P·V
    NOP = auto()            # 空操作（占位）


@dataclass
class ExecUnitConfig:
    """
    执行单元配置：
    - latency: 处理一个 tile 的 pipeline 深度
    - issue_interval: 连续发射两条同类指令之间至少间隔多少 cycle（II）
    - max_active: pipeline 内最多能容纳多少个正在执行的 tile
    - max_issue_per_cycle: 每个 cycle 最多可以发射多少条到该单元
    """
    name: str
    latency: int
    issue_interval: int
    max_active: int
    max_issue_per_cycle: int = 1


@dataclass
class Instruction:
    """
    Warp 指令：
    - 每条指令代表 “对某个 tile 做一次操作”。
    - 依赖通过 dependence_ids 表示：只有所有依赖完成后，该指令才可以发射。
    - row_id / tile_k 用于在线 softmax 状态跟踪：
        row_id: 这一 tile 属于哪一条 attention row
        tile_k: 这一 row 上的第几个 K-chunk（online softmax 顺序）
    """
    id: int
    warp_id: int
    row_id: int
    tile_k: int
    op_type: OpType
    dependence_ids: List[int] = field(default_factory=list)

    # 执行状态（由模拟器填充）
    issued_cycle: Optional[int] = None
    start_cycle: Optional[int] = None
    complete_cycle: Optional[int] = None


@dataclass
class InFlightOp:
    """正在执行中的指令实例"""
    inst_id: int
    complete_cycle: int


@dataclass
class OnlineRowState:
    """
    Online Softmax 状态 (m_i, l_i, O_i) 的抽象：
    - 实际数值不重要，这里只关心 “这个 row 已经处理了多少个 tile”，
      以及顺序是否被尊重。
    """
    m: float = float("-inf")
    l: float = 0.0
    tiles_done: int = 0


# ==========================
# 执行单元（Tensor Core / Softmax）
# ==========================

class ExecUnit:
    """
    通用执行单元：
    - in_flight: 当前 pipeline 中的指令
    - last_issue_cycle: 最近一次发射的 cycle
    - issued_this_cycle: 当前 cycle 已经发射了多少条（受 max_issue_per_cycle 限制）
    """
    def __init__(self, config: ExecUnitConfig):
        self.config = config
        self.in_flight: List[InFlightOp] = []
        self.last_issue_cycle: int = -1
        self.issued_this_cycle: int = 0

    def reset_cycle(self):
        """每个 cycle 开始时重置本 cycle 的发射计数"""
        self.issued_this_cycle = 0

    def can_issue(self, now: int) -> bool:
        """当前 cycle 是否可以再发射一条指令"""
        # issue 间隔约束
        if self.last_issue_cycle >= 0:
            if now - self.last_issue_cycle < self.config.issue_interval:
                return False
        # pipeline 容量约束
        if len(self.in_flight) >= self.config.max_active:
            return False
        # 本 cycle 发射条数限制
        if self.issued_this_cycle >= self.config.max_issue_per_cycle:
            return False
        return True

    def issue(self, inst_id: int, now: int) -> InFlightOp:
        """发射一条指令进入本执行单元"""
        complete_cycle = now + self.config.latency
        op = InFlightOp(inst_id=inst_id, complete_cycle=complete_cycle)
        self.in_flight.append(op)
        self.last_issue_cycle = now
        self.issued_this_cycle += 1
        return op

    def update(self, now: int) -> List[int]:
        """推进时间，返回在本 cycle 完成的 inst_id 列表"""
        completed: List[int] = []
        remain: List[InFlightOp] = []
        for op in self.in_flight:
            if now >= op.complete_cycle:
                completed.append(op.inst_id)
            else:
                remain.append(op)
        self.in_flight = remain
        return completed


# ==========================
# Warp & SM & 处理器抽象
# ==========================

@dataclass
class Warp:
    """
    简化的 Warp 模型：
    - program: 指令列表（按构造顺序执行）
    - pc: 当前待执行指令的索引
    """
    id: int
    program: List[Instruction]
    pc: int = 0

    def is_done(self) -> bool:
        return self.pc >= len(self.program)

    def next_inst(self) -> Optional[Instruction]:
        if self.is_done():
            return None
        return self.program[self.pc]

    def advance(self):
        """当前指令完成后，pc 前进一位"""
        self.pc += 1


@dataclass
class SMConfig:
    """SM 配置"""
    num_warps: int
    rows_per_warp: int
    tiles_per_row: int

    # Tensor Core 配置（QK^T + PV 共用）
    tc_latency: int = 4
    tc_issue_interval: int = 1
    tc_max_active: int = 8
    tc_max_issue_per_cycle: int = 1

    # Softmax Engine 配置
    softmax_engine_latency: int = 4
    softmax_engine_issue_interval: int = 1
    softmax_engine_max_active: int = 8
    softmax_engine_max_issue_per_cycle: int = 1

    # SFU softmax baseline 配置
    sfu_softmax_latency: int = 20
    sfu_softmax_issue_interval: int = 4
    sfu_softmax_max_active: int = 4
    sfu_softmax_max_issue_per_cycle: int = 1


class SM:
    """
    单个 SM：
    - 拥有一个 Tensor Core 执行单元
    - 拥有一个 Softmax Engine（可选）或 SFU softmax 模型
    - 有若干个 Warp
    - 为每个 (warp, row) 维护 Online Softmax 状态
    """
    def __init__(self, config: SMConfig, use_softmax_engine: bool):
        self.config = config
        self.use_softmax_engine = use_softmax_engine

        # Tensor Core
        tc_cfg = ExecUnitConfig(
            name="TensorCore",
            latency=config.tc_latency,
            issue_interval=config.tc_issue_interval,
            max_active=config.tc_max_active,
            max_issue_per_cycle=config.tc_max_issue_per_cycle,
        )
        self.tc_unit = ExecUnit(tc_cfg)

        # Softmax 单元
        if use_softmax_engine:
            smx_cfg = ExecUnitConfig(
                name="SoftmaxEngine",
                latency=config.softmax_engine_latency,
                issue_interval=config.softmax_engine_issue_interval,
                max_active=config.softmax_engine_max_active,
                max_issue_per_cycle=config.softmax_engine_max_issue_per_cycle,
            )
            self.softmax_unit = ExecUnit(smx_cfg)
            self.sfu_unit = None
        else:
            sfu_cfg = ExecUnitConfig(
                name="SoftmaxSFU",
                latency=config.sfu_softmax_latency,
                issue_interval=config.sfu_softmax_issue_interval,
                max_active=config.sfu_softmax_max_active,
                max_issue_per_cycle=config.sfu_softmax_max_issue_per_cycle,
            )
            self.softmax_unit = None
            self.sfu_unit = ExecUnit(sfu_cfg)

        # 构造 warps 与程序
        self.warps: List[Warp] = self._build_warps()

        # 指令索引表
        self.instructions: Dict[int, Instruction] = {}
        for w in self.warps:
            for inst in w.program:
                self.instructions[inst.id] = inst

        # Online softmax 状态：(warp_id, row_id) -> OnlineRowState
        self.online_state: Dict[Tuple[int, int], OnlineRowState] = {}
        for w in range(self.config.num_warps):
            for r in range(self.config.rows_per_warp):
                self.online_state[(w, r)] = OnlineRowState()

        # 时间 & 完成集
        self.now: int = 0
        self.completed_inst_ids: set[int] = set()

        # 每拍的发射时间线
        self.tc_timeline: List[str] = []
        self.softmax_timeline: List[str] = []

        # round-robin warp scheduler
        self.next_warp_to_schedule: int = 0

    # ---------- 程序生成：含 Online Softmax 顺序依赖 ----------

    def _build_warps(self) -> List[Warp]:
        """
        为每个 warp 生成：
          对 row = 0..rows_per_warp-1：
             对 tile_k = 0..tiles_per_row-1：
                QK^T[row, k] -> Softmax[row, k] -> PV[row, k]
          其中：
            Softmax[row, k>0] 依赖 Softmax[row, k-1]
            保证 Online Softmax (m_i,l_i,O_i) 顺序更新
        """
        warps: List[Warp] = []
        for w in range(self.config.num_warps):
            prog: List[Instruction] = []
            inst_id_base = w * 100000  # 每个 warp 一个大块 id 空间

            for r in range(self.config.rows_per_warp):
                for k in range(self.config.tiles_per_row):
                    base = inst_id_base + r * 1000 + k * 10

                    # 1) QK^T(row, k)
                    qk_id = base + 0
                    qk_inst = Instruction(
                        id=qk_id,
                        warp_id=w,
                        row_id=r,
                        tile_k=k,
                        op_type=OpType.TC_QK,
                        dependence_ids=[],
                    )
                    prog.append(qk_inst)

                    # 2) Softmax(row, k)
                    smx_id = base + 1
                    smx_deps = [qk_id]
                    if k > 0:
                        prev_smx_id = inst_id_base + r * 1000 + (k - 1) * 10 + 1
                        smx_deps.append(prev_smx_id)

                    smx_inst = Instruction(
                        id=smx_id,
                        warp_id=w,
                        row_id=r,
                        tile_k=k,
                        op_type=OpType.SOFTMAX_ENGINE
                        if self.use_softmax_engine
                        else OpType.SOFTMAX_SFU,
                        dependence_ids=smx_deps,
                    )
                    prog.append(smx_inst)

                    # 3) PV(row, k)
                    pv_id = base + 2
                    pv_inst = Instruction(
                        id=pv_id,
                        warp_id=w,
                        row_id=r,
                        tile_k=k,
                        op_type=OpType.TC_PV,
                        dependence_ids=[smx_id],
                    )
                    prog.append(pv_inst)

            warps.append(Warp(id=w, program=prog))

        return warps

    # ---------- 主循环 ----------

    def all_done(self) -> bool:
        return all(w.is_done() for w in self.warps)

    def step(self):
        """
        模拟一个 cycle：
          1. 重置执行单元本拍发射计数
          2. 更新执行单元，收集完成的 inst_id
          3. 对完成的 softmax 指令，更新 Online Softmax 状态
          4. Warp 调度（round-robin）
        """
        # 0) 重置 issue 计数
        self.tc_unit.reset_cycle()
        if self.use_softmax_engine:
            self.softmax_unit.reset_cycle()
        else:
            self.sfu_unit.reset_cycle()

        # 1) 更新执行单元，记录谁完成了
        completed_tc = self.tc_unit.update(self.now)
        if self.use_softmax_engine:
            completed_softmax = self.softmax_unit.update(self.now)
        else:
            completed_softmax = self.sfu_unit.update(self.now)

        completed_all = completed_tc + completed_softmax
        for inst_id in completed_all:
            if inst_id in self.completed_inst_ids:
                continue
            self.completed_inst_ids.add(inst_id)
            inst = self.instructions[inst_id]
            inst.complete_cycle = self.now

            # 2) softmax 完成时，更新 Online Softmax 状态
            if inst.op_type in (OpType.SOFTMAX_ENGINE, OpType.SOFTMAX_SFU):
                self._update_online_state(inst)

        # 3) warp 调度：round-robin
        tc_issued_this_cycle = False
        smx_issued_this_cycle = False

        num_warps = len(self.warps)
        for _ in range(num_warps):
            w_id = self.next_warp_to_schedule
            warp = self.warps[w_id]
            self.next_warp_to_schedule = (self.next_warp_to_schedule + 1) % num_warps

            inst = warp.next_inst()
            if inst is None:
                continue

            # 依赖检查
            if not all(dep_id in self.completed_inst_ids for dep_id in inst.dependence_ids):
                continue

            # TensorCore 指令
            if inst.op_type in (OpType.TC_QK, OpType.TC_PV):
                if not self.tc_unit.can_issue(self.now):
                    continue
                inst.issued_cycle = self.now
                inst.start_cycle = self.now
                self.tc_unit.issue(inst.id, self.now)
                tc_issued_this_cycle = True
                warp.advance()

            # Softmax 指令
            elif inst.op_type in (OpType.SOFTMAX_ENGINE, OpType.SOFTMAX_SFU):
                unit = self.softmax_unit if self.use_softmax_engine else self.sfu_unit
                if not unit.can_issue(self.now):
                    continue
                inst.issued_cycle = self.now
                inst.start_cycle = self.now
                unit.issue(inst.id, self.now)
                smx_issued_this_cycle = True
                warp.advance()

            else:
                warp.advance()

        # 4) 记录时间线
        self.tc_timeline.append("-" if tc_issued_this_cycle else ".")
        self.softmax_timeline.append("-" if smx_issued_this_cycle else ".")

        # 时间推进
        self.now += 1

    def run(self, max_cycles: int = 100000) -> int:
        """一直跑到所有 warp 完成或达到 max_cycles"""
        while not self.all_done() and self.now < max_cycles:
            self.step()
        return self.now

    # ---------- Online Softmax 状态更新 ----------

    def _update_online_state(self, inst: Instruction):
        """
        softmax(row, k) 完成时，更新对应 (warp, row) 的在线状态。
        这里只做顺序检查 + 计数更新，不做真实数值计算。
        """
        key = (inst.warp_id, inst.row_id)
        state = self.online_state[key]
        expected_k = state.tiles_done
        if inst.tile_k != expected_k:
            print(f"[WARN] Online state 顺序异常: warp{inst.warp_id}, "
                  f"row{inst.row_id}, 完成 tile_k={inst.tile_k}, "
                  f"但期望 {expected_k}")
            state.tiles_done = inst.tile_k + 1
        else:
            state.tiles_done += 1

        # 实际数值这里不算，只是标记成“已更新过”
        state.m = 0.0
        state.l = 1.0

    # ---------- 结果与可视化 ----------

    def summarize(self):
        """打印总体信息和简要统计"""
        mode_str = "Softmax Engine" if self.use_softmax_engine else "SFU Baseline"
        print("=" * 60)
        print(f"SM 模拟结果 (模式: {mode_str})")
        print("-" * 60)
        print(f"  Warp 数量          : {self.config.num_warps}")
        print(f"  每个 warp 的 rows  : {self.config.rows_per_warp}")
        print(f"  每条 row 的 tiles  : {self.config.tiles_per_row}")
        print(f"  总周期数           : {self.now}")
        print()

        # 每个 warp 完成时间
        print("  每个 warp 的完成时间 (最后一条指令完成 cycle):")
        print("  " + "-" * 52)
        print("  {:>6} | {:>10}".format("Warp", "Complete"))
        print("  " + "-" * 52)
        for w in self.warps:
            last_complete = max(
                (inst.complete_cycle for inst in w.program if inst.complete_cycle is not None),
                default=-1,
            )
            print("  {:>6} | {:>10}".format(w.id, last_complete))
        print("  " + "-" * 52)
        print()

        # 在线 softmax 状态检查
        print("  Online Softmax 状态 (tiles_done)：")
        print("  " + "-" * 52)
        print("  {:>6} | {:>6} | {:>10}".format("Warp", "Row", "TilesDone"))
        print("  " + "-" * 52)
        for (w, r), state in sorted(self.online_state.items()):
            print("  {:>6} | {:>6} | {:>10}".format(w, r, state.tiles_done))
        print("  " + "-" * 52)
        print()

        # 执行单元利用率（按有无发射近似）
        tc_busy = sum(1 for c in self.tc_timeline if c == "-")
        smx_busy = sum(1 for c in self.softmax_timeline if c == "-")
        total = len(self.tc_timeline)
        print("  执行单元利用率（按有无发射的 cycle 近似衡量）：")
        print("  " + "-" * 52)
        print("  {:<16}: {:>4}/{:<4} cycles (≈ {:.3f})".format(
            "TensorCore", tc_busy, total, tc_busy / total if total else 0.0))
        print("  {:<16}: {:>4}/{:<4} cycles (≈ {:.3f})".format(
            "SoftmaxUnit", smx_busy, total, smx_busy / total if total else 0.0))
        print("=" * 60)
        print()

    def print_timeline(self, max_cycles: int = 200):
        """
        打印前 max_cycles 个 cycle 的简化时间线：
        - '-' 表示该 cycle 有指令发射到该单元
        - '.' 表示空闲
        """
        cut = min(max_cycles, len(self.tc_timeline))
        if cut == 0:
            print("无时间线数据（尚未运行模拟）。")
            return

        indices = "".join(str(i % 10) for i in range(cut))
        print("时间线 (前 {} cycles)：".format(cut))
        print("-" * (cut + 12))
        print("cycle     | " + indices)
        print("TensorCore| " + "".join(self.tc_timeline[:cut]))
        print("Softmax   | " + "".join(self.softmax_timeline[:cut]))
        print("-" * (cut + 12))
        print()


# ==========================
# 命令行入口
# ==========================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SIMT + Tensor Core + Softmax Engine 模拟器（Online Softmax）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-warps", type=int, default=4, help="每个 SM 上的 warp 数量")
    parser.add_argument("--rows-per-warp", type=int, default=2, help="每个 warp 的 attention rows 数量")
    parser.add_argument("--tiles-per-row", type=int, default=4, help="每条 row 需要处理的 K-tiles 数量")
    parser.add_argument("--use-softmax-engine", action="store_true", help="启用 Softmax Engine，而不是 SFU baseline")

    # Tensor Core 参数
    parser.add_argument("--tc-latency", type=int, default=4, help="Tensor Core 处理一个 tile 的 latency（cycle）")
    parser.add_argument("--tc-ii", type=int, default=1, help="Tensor Core issue interval（II）")
    parser.add_argument("--tc-max-active", type=int, default=8, help="Tensor Core pipeline 中最多 active tiles 数量")
    parser.add_argument("--tc-max-issue-per-cycle", type=int, default=1, help="Tensor Core 每个 cycle 最多发射多少条指令")

    # Softmax Engine 参数
    parser.add_argument("--smx-latency", type=int, default=4, help="Softmax Engine 的 pipeline 深度")
    parser.add_argument("--smx-ii", type=int, default=1, help="Softmax Engine issue interval")
    parser.add_argument("--smx-max-active", type=int, default=8, help="Softmax Engine pipeline 中 active tiles 上限")
    parser.add_argument("--smx-max-issue-per-cycle", type=int, default=1, help="Softmax Engine 每个 cycle 最多发射多少条指令")

    # SFU softmax baseline 参数
    parser.add_argument("--sfu-latency", type=int, default=20, help="SFU softmax baseline 的 latency（一个 tile）")
    parser.add_argument("--sfu-ii", type=int, default=4, help="SFU softmax baseline issue interval（>1 表示吞吐 < 1 tile/cycle）")
    parser.add_argument("--sfu-max-active", type=int, default=4, help="SFU softmax baseline pipeline 容量")
    parser.add_argument("--sfu-max-issue-per-cycle", type=int, default=1, help="SFU softmax baseline 每个 cycle 最多发射多少条指令")

    args = parser.parse_args()

    sm_cfg = SMConfig(
        num_warps=args.num_warps,
        rows_per_warp=args.rows_per_warp,
        tiles_per_row=args.tiles_per_row,
        tc_latency=args.tc_latency,
        tc_issue_interval=args.tc_ii,
        tc_max_active=args.tc_max_active,
        tc_max_issue_per_cycle=args.tc_max_issue_per_cycle,
        softmax_engine_latency=args.smx_latency,
        softmax_engine_issue_interval=args.smx_ii,
        softmax_engine_max_active=args.smx_max_active,
        softmax_engine_max_issue_per_cycle=args.smx_max_issue_per_cycle,
        sfu_softmax_latency=args.sfu_latency,
        sfu_softmax_issue_interval=args.sfu_ii,
        sfu_softmax_max_active=args.sfu_max_active,
        sfu_softmax_max_issue_per_cycle=args.sfu_max_issue_per_cycle,
    )

    sm = SM(sm_cfg, use_softmax_engine=args.use_softmax_engine)
    sm.run()
    sm.summarize()
    sm.print_timeline()


if __name__ == "__main__":
    main()
