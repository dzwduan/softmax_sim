#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMT + Tensor Core + Softmax Engine 模拟器（tile 粒度）

设计目标：
- 抽象一个 SM，上面有：
  - Tensor Core：负责 QK^T / PV 的 MMA 计算
  - Softmax Engine：专用 softmax 引擎（可选）
  - SFU Softmax：通用 ALU+SFU 实现 softmax（baseline）
- 把 QK^T → Softmax → PV 三段流程，用 tile 为粒度建模。
- 支持多 warp，每个 warp 处理若干 tiles。
- 比较两种模式下的总时许和执行单元利用率：

  1) baseline：   QK^T (TC) → softmax (SFU)   → PV (TC)
  2) engine 版：  QK^T (TC) → softmax (Engine) → PV (TC)

注意：
- 这是一个高层次性能模型，不关心具体每个元素的计算，也不关心寄存器分配。
- 提供 ASCII 时间线，方便肉眼观察 Tensor Core / Softmax 之间的流水关系。
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
    SOFTMAX_ENGINE = auto() # Softmax Engine tile softmax
    SOFTMAX_SFU = auto()    # 用 SFU/ALU 实现的 softmax（baseline）
    TC_PV = auto()          # Tensor Core 做 P·V
    NOP = auto()            # 空操作（占位用，可扩展）


@dataclass
class ExecUnitConfig:
    """
    执行单元配置：
    - latency: 处理一个 tile 的 pipeline 深度
    - issue_interval: 连续发射两条同类指令之间至少间隔多少 cycle（II）
    - max_active: pipeline 内最多能容纳多少个正在执行的 tile
    """
    name: str
    latency: int
    issue_interval: int
    max_active: int


@dataclass
class Instruction:
    """
    Warp 指令：
    - 每条指令代表 “对某个 tile 做一次操作”。
    - 依赖通过 dependence_ids 表示：只有所有依赖的指令完成后，该指令才可以发射。
    """
    id: int
    warp_id: int
    tile_id: int
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


# ==========================
# 执行单元（Tensor Core / Softmax）
# ==========================

class ExecUnit:
    """
    通用执行单元基类，内部维护：
    - in_flight: 当前 pipeline 中的指令（tile）
    - last_issue_cycle: 上一次发射的 cycle
    """
    def __init__(self, config: ExecUnitConfig):
        self.config = config
        self.in_flight: List[InFlightOp] = []
        self.last_issue_cycle: int = -1

    def can_issue(self, now: int) -> bool:
        """当前 cycle 是否可以再发射一条指令"""
        # issue 间隔约束
        if self.last_issue_cycle >= 0:
            if now - self.last_issue_cycle < self.config.issue_interval:
                return False
        # pipeline 容量约束
        if len(self.in_flight) >= self.config.max_active:
            return False
        return True

    def issue(self, inst_id: int, now: int) -> InFlightOp:
        """发射一条指令进入本执行单元"""
        complete_cycle = now + self.config.latency
        op = InFlightOp(inst_id=inst_id, complete_cycle=complete_cycle)
        self.in_flight.append(op)
        self.last_issue_cycle = now
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
    - program: 指令列表（按 id 顺序填充）
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
    tiles_per_warp: int

    # Tensor Core 配置（QK^T + PV 共用）
    tc_latency: int = 4
    tc_issue_interval: int = 1
    tc_max_active: int = 8

    # Softmax Engine 配置
    softmax_engine_latency: int = 4
    softmax_engine_issue_interval: int = 1
    softmax_engine_max_active: int = 8

    # SFU softmax baseline 配置
    sfu_softmax_latency: int = 20
    sfu_softmax_issue_interval: int = 4
    sfu_softmax_max_active: int = 4


class SM:
    """
    单个 SM：
    - 拥有一个 Tensor Core 执行单元
    - 拥有一个 Softmax Engine（可选）或 SFU softmax 模型
    - 有若干个 Warp
    """
    def __init__(self, config: SMConfig, use_softmax_engine: bool):
        self.config = config
        self.use_softmax_engine = use_softmax_engine

        # 创建执行单元
        tc_cfg = ExecUnitConfig(
            name="TensorCore",
            latency=config.tc_latency,
            issue_interval=config.tc_issue_interval,
            max_active=config.tc_max_active,
        )
        self.tc_unit = ExecUnit(tc_cfg)

        if use_softmax_engine:
            smx_cfg = ExecUnitConfig(
                name="SoftmaxEngine",
                latency=config.softmax_engine_latency,
                issue_interval=config.softmax_engine_issue_interval,
                max_active=config.softmax_engine_max_active,
            )
            self.softmax_unit = ExecUnit(smx_cfg)
            self.sfu_unit = None
        else:
            sfu_cfg = ExecUnitConfig(
                name="SoftmaxSFU",
                latency=config.sfu_softmax_latency,
                issue_interval=config.sfu_softmax_issue_interval,
                max_active=config.sfu_softmax_max_active,
            )
            self.softmax_unit = None
            self.sfu_unit = ExecUnit(sfu_cfg)

        # 构造 warps 与程序
        self.warps: List[Warp] = self._build_warps()

        # 记录每个指令对象（方便通过 id 查状态）
        self.instructions: Dict[int, Instruction] = {}
        for w in self.warps:
            for inst in w.program:
                self.instructions[inst.id] = inst

        # 执行完成标记
        self.now: int = 0
        self.completed_inst_ids: set[int] = set()

        # 用于画 timeline：每个 cycle 记录对应执行单元干了什么
        self.tc_timeline: List[str] = []
        self.softmax_timeline: List[str] = []

    # ---------- 程序生成 ----------

    def _build_warps(self) -> List[Warp]:
        """
        为每个 warp 生成：
            对 tiles 0..tiles_per_warp-1：
                TC_QK -> SOFTMAX_* -> TC_PV
        指令 id 的编码方式只是为了区分，不影响语义。
        """
        warps: List[Warp] = []
        for w in range(self.config.num_warps):
            prog: List[Instruction] = []
            inst_id_base = w * 10_000

            for t in range(self.config.tiles_per_warp):
                base = inst_id_base + t * 10

                # 1) QK^T
                qk_id = base + 0
                qk_inst = Instruction(
                    id=qk_id,
                    warp_id=w,
                    tile_id=t,
                    op_type=OpType.TC_QK,
                    dependence_ids=[],  # 对同一个 warp，假设各 tile 之间独立
                )
                prog.append(qk_inst)

                # 2) Softmax
                smx_id = base + 1
                smx_inst = Instruction(
                    id=smx_id,
                    warp_id=w,
                    tile_id=t,
                    op_type=OpType.SOFTMAX_ENGINE
                    if self.use_softmax_engine
                    else OpType.SOFTMAX_SFU,
                    dependence_ids=[qk_id],
                )
                prog.append(smx_inst)

                # 3) PV
                pv_id = base + 2
                pv_inst = Instruction(
                    id=pv_id,
                    warp_id=w,
                    tile_id=t,
                    op_type=OpType.TC_PV,
                    dependence_ids=[smx_id],
                )
                prog.append(pv_inst)

            warps.append(Warp(id=w, program=prog))

        return warps

    # ---------- 模拟主循环 ----------

    def all_done(self) -> bool:
        return all(w.is_done() for w in self.warps)

    def step(self):
        """
        模拟一个 cycle：
          1. 更新执行单元，收集完成的 inst_id
          2. 轮询 warp，尝试发射下一条可执行的指令
        """
        # 1) 更新执行单元，记录谁完成了
        completed_tc = self.tc_unit.update(self.now)
        completed_softmax: List[int] = []
        if self.use_softmax_engine:
            completed_softmax = self.softmax_unit.update(self.now)
        else:
            completed_softmax = self.sfu_unit.update(self.now)

        for inst_id in completed_tc + completed_softmax:
            self.completed_inst_ids.add(inst_id)
            inst = self.instructions[inst_id]
            inst.complete_cycle = self.now

        # 2) 尝试发射新指令（简单轮询 scheduler）
        tc_issued_this_cycle = False
        smx_issued_this_cycle = False

        for warp in self.warps:
            inst = warp.next_inst()
            if inst is None:
                continue

            # 依赖检查：所有 dependence_ids 都已完成
            if not all(dep_id in self.completed_inst_ids for dep_id in inst.dependence_ids):
                continue

            # 根据 op_type 选择执行单元
            if inst.op_type in (OpType.TC_QK, OpType.TC_PV):
                if tc_issued_this_cycle:
                    continue  # 本 cycle 已经给 TensorCore 发过一条了，简单做个单发射
                if self.tc_unit.can_issue(self.now):
                    inst.issued_cycle = self.now
                    inst.start_cycle = self.now
                    self.tc_unit.issue(inst.id, self.now)
                    tc_issued_this_cycle = True
                    # 指令已经发射，可以前进 PC
                    warp.advance()
            elif inst.op_type in (OpType.SOFTMAX_ENGINE, OpType.SOFTMAX_SFU):
                if smx_issued_this_cycle:
                    continue
                unit = self.softmax_unit if self.use_softmax_engine else self.sfu_unit
                if unit.can_issue(self.now):
                    inst.issued_cycle = self.now
                    inst.start_cycle = self.now
                    unit.issue(inst.id, self.now)
                    smx_issued_this_cycle = True
                    warp.advance()
            else:
                # NOP 或未实现类型
                warp.advance()

        # 3) 记录 timeline（只记录这一 cycle 是否有指令发射）
        tc_symbol = "-" if tc_issued_this_cycle else "."
        self.tc_timeline.append(tc_symbol)

        smx_symbol = "-" if smx_issued_this_cycle else "."
        self.softmax_timeline.append(smx_symbol)

        # 时间推进
        self.now += 1

    def run(self, max_cycles: int = 100000) -> int:
        """一直跑到所有 warp 完成或达到 max_cycles"""
        while not self.all_done() and self.now < max_cycles:
            self.step()
        return self.now

    # ---------- 结果与可视化 ----------

    def summarize(self):
        """打印总体信息和简单统计"""
        print("==== 模拟结果 ====")
        print(f"软max模式: {'Softmax Engine' if self.use_softmax_engine else 'SFU Baseline'}")
        print(f"总 cycle 数: {self.now}")

        # 统计 TC 和 softmax 单元的占用
        tc_busy = sum(1 for c in self.tc_timeline if c == "-")
        smx_busy = sum(1 for c in self.softmax_timeline if c == "-")
        total = len(self.tc_timeline)
        print()
        print("执行单元利用率（按发射次数近似）：")
        print(f"  TensorCore: {tc_busy}/{total} cycles, 利用率 ≈ {tc_busy/total:.3f}")
        print(f"  Softmax  : {smx_busy}/{total} cycles, 利用率 ≈ {smx_busy/total:.3f}")
        print()

    def print_timeline(self, max_cycles: int = 200):
        """
        打印前 max_cycles 个 cycle 的简化时间线：
        - '-' 表示该 cycle 有指令发射到该单元
        - '.' 表示空闲
        """
        cut = min(max_cycles, len(self.tc_timeline))
        indices = "".join(str(i % 10) for i in range(cut))
        print("cycle:     ", indices)
        print("TC   :     ", "".join(self.tc_timeline[:cut]))
        print("Softmax:   ", "".join(self.softmax_timeline[:cut]))
        print()


# ==========================
# 命令行入口
# ==========================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SIMT + Tensor Core + Softmax Engine 模拟器（tile 级）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-warps", type=int, default=4, help="每个 SM 上的 warp 数量")
    parser.add_argument("--tiles-per-warp", type=int, default=8, help="每个 warp 需要处理的 tile 数量")
    parser.add_argument("--use-softmax-engine", action="store_true", help="启用 Softmax Engine，而不是 SFU baseline")

    # 可以调 tensor core 和 softmax 的时序参数
    parser.add_argument("--tc-latency", type=int, default=4, help="Tensor Core 处理一个 tile 的 latency（cycle）")
    parser.add_argument("--tc-ii", type=int, default=1, help="Tensor Core 连续发射的 issue interval（II）")
    parser.add_argument("--tc-max-active", type=int, default=8, help="Tensor Core pipeline 中最多 active tiles 数量")

    parser.add_argument("--smx-latency", type=int, default=4, help="Softmax Engine 的 pipeline 深度")
    parser.add_argument("--smx-ii", type=int, default=1, help="Softmax Engine 的 issue interval")
    parser.add_argument("--smx-max-active", type=int, default=8, help="Softmax Engine pipeline 中 active tiles 上限")

    parser.add_argument("--sfu-latency", type=int, default=20, help="SFU softmax baseline 的 latency（一个 tile）")
    parser.add_argument("--sfu-ii", type=int, default=4, help="SFU softmax baseline 的 issue interval（>1 表示吞吐 < 1 tile/cycle）")
    parser.add_argument("--sfu-max-active", type=int, default=4, help="SFU softmax baseline pipeline 容量")

    args = parser.parse_args()

    sm_cfg = SMConfig(
        num_warps=args.num_warps,
        tiles_per_warp=args.tiles_per_warp,
        tc_latency=args.tc_latency,
        tc_issue_interval=args.tc_ii,
        tc_max_active=args.tc_max_active,
        softmax_engine_latency=args.smx_latency,
        softmax_engine_issue_interval=args.smx_ii,
        softmax_engine_max_active=args.smx_max_active,
        sfu_softmax_latency=args.sfu_latency,
        sfu_softmax_issue_interval=args.sfu_ii,
        sfu_softmax_max_active=args.sfu_max_active,
    )

    sm = SM(sm_cfg, use_softmax_engine=args.use_softmax_engine)
    total_cycles = sm.run()
    sm.summarize()
    sm.print_timeline()


if __name__ == "__main__":
    main()
