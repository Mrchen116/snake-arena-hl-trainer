from __future__ import annotations

# ============================================================
# train.py — Heuristic Learning 多轮训练编排器
#
# 数据/代码同仓布局：
#   ~/Repos/snake-arena-hl/                trainer code
#   ├── train.py
#   ├── snake_hl/                          trainer modules
#   │   ├── env.py, eval.py, baselines.py, ...
#   │   └── policy.py                      ← 运行槽（gitignored，由 train.py 在每次启动时
#   │                                         从 experiments/<exp>/policy.py 拷贝填充）
#   ├── experiments/                       ← gitignored 整体
#   │   ├── <exp-A>/                       ← 一个实验 = 一个目录（可独立 git）
#   │   │   ├── policy.py                  canonical 持久态
#   │   │   ├── heuristic_notes.md
#   │   │   ├── runs/, replays/, reports/
#   │   │   ├── trials.jsonl, training_curve.{json,html}
#   │   │   └── .claude/settings.local.json   train.py 启动时重新生成，deny 兄弟实验
#   │   └── <exp-B>/ ...
#   └── tests/
#
# 工作流：
#   1. train.py 启动时：单文件 copy_in，把 experiments/<exp>/policy.py → snake_hl/policy.py
#   2. 写 experiments/<exp>/.claude/settings.local.json（动态枚举当前 siblings）
#   3. 启动 optimizer subprocess，cwd=experiments/<exp>/
#      - optimizer 读到 cwd 下的 .claude/settings.local.json，被 deny 访问兄弟实验
#      - optimizer 编辑 trainer 的 snake_hl/policy.py（绝对路径）；其他 data 文件用 cwd 相对
#   4. 每轮结束：单文件 copy_out，把 snake_hl/policy.py → experiments/<exp>/policy.py
#
# 安全设计：
#   - settings.local.json 落在 exp 目录而不是 trainer 根，避免影响开发者自己的 CC 会话
#   - --permission-mode dontAsk + 显式 allow 列表，禁用 bypass mode
#   - hash-based 边界检查（仍保留作为 belt-and-braces）
#
# 【DL 类比】
#   模型参数 (weights)    snake_hl/policy.py（运行槽，每轮从 exp 拷入、拷出）
#   损失函数 (loss)       -avg_score
#   优化器 (optimizer)    Claude Code CLI
#   一个 epoch            一轮 run_round()
#   模型检查点            每轮 copy_out 持久化到 experiments/<exp>/policy.py
# ============================================================

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values

# snake_hl.eval / failure_report / policy 都需要 snake_hl/policy.py 存在才能 import。
# train.py 启动早期要先做 copy_in 才能让这些模块可用，所以延迟 import。
eval_module = None  # type: ignore
failure_report_module = None  # type: ignore
policy_module = None  # type: ignore


def _load_snake_hl_modules() -> None:
    """在 copy_in() 之后调用，把 snake_hl.* 三个模块绑到全局。"""
    global eval_module, failure_report_module, policy_module
    from snake_hl import eval as _em
    from snake_hl import failure_report as _frm
    from snake_hl import policy as _pm
    eval_module = _em
    failure_report_module = _frm
    policy_module = _pm


ROOT = Path(__file__).resolve().parent

# 实验数据目录根。固定在 trainer 内部，不允许通过环境变量覆盖（参考 settings.local.json
# 提供路径泄露的考量：把这个常量留在源码里是 OK 的，因为不暴露具体实验名）。
DATA_HOME_BASE = ROOT / "experiments"

# Trainer venv 的 python 绝对路径，optimizer prompt + permissions allow 列表都要用到
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

# 旧 monolithic 仓的归档目录。出现时连同兄弟实验一起 deny。
ARCHIVE_DIR = Path.home() / "Repos" / "snake-arena-hl-archive"

# 单文件 cp：optimizer 编辑 snake_hl/policy.py 这个运行槽，每轮结束 copy_out 回 exp。
# 其他 data 文件（heuristic_notes、runs、replays、reports 等）都直接在
# experiments/<exp>/ 下读写，不再有 trainer-side 运行槽。
POLICY_SLOT = ROOT / "snake_hl" / "policy.py"

# Hash-based 边界检查白名单。只剩 policy.py，因为其他 data 文件都在 experiments/ 下
# （被 IGNORED_SNAPSHOT_PARTS 排除），不会进入 hash 比对。
ALLOWED_OPTIMIZER_EDITS = {
    "snake_hl/policy.py",
}

# Hash 快照时跳过这些目录或前缀
IGNORED_SNAPSHOT_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "snake_arena_hl.egg-info",
    "experiments",  # data 目录整个跳过，optimizer 在里面的活动不受 hash 检查
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def repo_path(path: Path) -> str:
    """trainer 内部相对路径（用于日志显示）。"""
    path = path if path.is_absolute() else ROOT / path
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def should_snapshot(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    parts = set(relative.parts)
    if parts & IGNORED_SNAPSHOT_PARTS:
        return False
    return path.is_file()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_repo() -> dict[str, str]:
    """对 trainer dir（排除 experiments/、缓存等）生成 {相对路径: 哈希}。"""
    return {
        repo_path(path): file_digest(path)
        for path in ROOT.rglob("*")
        if should_snapshot(path)
    }


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    paths = set(before) | set(after)
    return sorted(path for path in paths if before.get(path) != after.get(path))


def run_command(args: list[str], *, output: Path | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """执行 shell 命令；指定 output 则写日志；失败抛异常。"""
    result = subprocess.run(args, cwd=cwd or ROOT, text=True, capture_output=True)
    if output:
        output.write_text(
            " ".join(args)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr,
            encoding="utf-8",
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}\n{result.stderr}")
    return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reload_policy_modules() -> None:
    """重 import policy 模块，让 optimizer 的改动在本进程内生效。"""
    importlib.reload(policy_module)
    importlib.reload(eval_module)
    importlib.reload(failure_report_module)


def load_dotenv(path: Path = ROOT / ".env") -> dict[str, str]:
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def claude_env() -> dict[str, str]:
    """构造 Claude CLI 子进程的环境变量。

    .env 提供默认值，shell 当前 environment 覆盖。
    不在 trainer 源码里硬编码任何 provider/base-URL/model—— 这些都从 .env 读。
    支持的 vars 见 .env.example（ANTHROPIC_API_KEY 必填，BASE_URL / model overrides 等
    取决于使用 Kimi / Mimo / 官方 API 哪一家）。
    """
    return {**load_dotenv(), **os.environ}


MIN_IMPROVEMENT = 2.0


# ============================================================
# 实验目录 ↔ trainer 运行槽 同步（仅 policy.py）
# ============================================================


def data_home_for(exp_name: str) -> Path:
    return DATA_HOME_BASE / exp_name


def validate_data_home(data_home: Path) -> None:
    if not data_home.is_dir():
        raise RuntimeError(
            f"Experiment directory not found: {data_home}\n"
            f"To create a fresh experiment, fork from an existing one:\n"
            f"  python train.py --exp {data_home.name} --new-from <existing-exp>"
        )
    policy_file = data_home / "policy.py"
    if not policy_file.is_file():
        raise RuntimeError(f"Required file missing in data dir: {policy_file}")


def copy_in_policy(data_home: Path) -> None:
    """启动时把 experiments/<exp>/policy.py 拷到 snake_hl/policy.py 运行槽。"""
    src = data_home / "policy.py"
    POLICY_SLOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, POLICY_SLOT)


def copy_out_policy(data_home: Path) -> None:
    """每轮结束把运行槽里的 policy.py 写回 experiments/<exp>/policy.py。"""
    if not POLICY_SLOT.exists():
        return
    shutil.copy2(POLICY_SLOT, data_home / "policy.py")


# ============================================================
# Optimizer 权限配置
# ============================================================


def write_optimizer_permissions(data_home: Path) -> None:
    """
    生成 experiments/<exp>/.claude/settings.local.json，让 optimizer subprocess
    （cwd=data_home）启动时读到 deny 规则，无法 Read/Glob/Grep 兄弟实验和归档。

    设计要点：
    - allow 列表覆盖 optimizer 日常需要的命令（python 跑 eval、git 只读、cd / ls / cat
      等基础工具），用 dontAsk 模式自动批准；不在列表里的工具会自动拒绝
    - deny 列表枚举当前所有 sibling 实验目录 + 旧 monolithic 仓归档目录
    - 用绝对路径（//... 双斜杠开头）保证匹配正确
    - 禁用 disableBypassPermissionsMode，杜绝 optimizer 自行切到 bypass

    限制（已知 + 接受）：
    - Bash(python *) 太宽，optimizer 可通过 python -c "open('禁止路径')" 绕过
      Read 的 deny；这只能 OS 级 sandbox 才能彻底防
    - allow 里 python 的绝对路径包含开发者 homedir，所以 settings.local.json
      跨机器不可移植；但每次 train.py 启动都重新生成，所以不是问题
    """
    base = data_home.parent  # DATA_HOME_BASE
    siblings = sorted(
        p for p in base.iterdir()
        if p.is_dir() and p != data_home and not p.name.startswith(".")
    )

    deny: list[str] = []
    # 兄弟实验：全套 CC 文件工具都 deny。
    # CC 绝对路径模式用 // 开头（双斜杠），而 Path 转字符串后已经以 / 开头，
    # 所以 f-string 写 `Read(/{path}/**)` —— 单斜杠 + 已含 / 的绝对路径 = 双斜杠。
    for sib in siblings:
        deny.append(f"Read(/{sib}/**)")
        deny.append(f"Glob(/{sib}/**)")
        deny.append(f"Grep(/{sib}/**)")
    # 归档目录
    if ARCHIVE_DIR.exists():
        deny.append(f"Read(/{ARCHIVE_DIR}/**)")
        deny.append(f"Glob(/{ARCHIVE_DIR}/**)")
        deny.append(f"Grep(/{ARCHIVE_DIR}/**)")
        deny.append("Bash(*snake-arena-hl-archive*)")
    # 禁止 optimizer 编辑 trainer 代码（双保险：hash 检查还在）
    forbidden_edits = [
        ROOT / "snake_hl" / "env.py",
        ROOT / "snake_hl" / "eval.py",
        ROOT / "snake_hl" / "baselines.py",
        ROOT / "snake_hl" / "failure_report.py",
        ROOT / "snake_hl" / "html_replay.py",
        ROOT / "snake_hl" / "replay.py",
        ROOT / "snake_hl" / "baseline_snapshot.py",
        ROOT / "snake_hl" / "__init__.py",
        ROOT / "train.py",
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "EVAL_PROTOCOL.md",
        ROOT / "pyproject.toml",
    ]
    for path in forbidden_edits:
        deny.append(f"Edit({path})")
        deny.append(f"Write({path})")

    allow = [
        # Python：用绝对路径锁定到 trainer venv
        f"Bash({VENV_PYTHON} *)",
        "Bash(python *)",
        "Bash(python3 *)",
        # 只读 git
        "Bash(git status)",
        "Bash(git diff *)",
        "Bash(git log *)",
        "Bash(git show *)",
        "Bash(git blame *)",
        # 浏览 / 小操作（不含 rm）
        "Bash(ls *)",
        "Bash(cat *)",
        "Bash(head *)",
        "Bash(tail *)",
        "Bash(grep *)",
        "Bash(find *)",
        "Bash(wc *)",
        "Bash(mkdir *)",
        "Bash(touch *)",
        "Bash(cp *)",
        "Bash(mv *)",
        "Bash(cd *)",
        "Bash(echo *)",
        # CC 内置工具
        "Edit",
        "Write",
        "Read",
        "Glob",
        "Grep",
    ]

    settings = {
        "permissions": {
            "defaultMode": "dontAsk",
            "allow": allow,
            "deny": deny,
            "disableBypassPermissionsMode": "disable",
        }
    }

    claude_dir = data_home / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.local.json").write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )


# ============================================================
# Optimizer prompt + 实验日志模板 + 每轮 snapshot
# ============================================================


def build_optimizer_prompt(
    *,
    data_home: Path,
    round_dir: Path,
    round_index: int,
    before_train: dict,
    failure_report: Path,
) -> str:
    """
    optimizer 的 cwd 是 data_home (experiments/<exp>/)，所以：
    - trainer-side 引用全部用绝对路径（README.md / snake_hl/policy.py 等）
    - data-side 引用用 cwd 相对路径（heuristic_notes.md / runs/<ts>/<round>/journal.md 等）
    - 验证命令用 trainer venv 的绝对路径
    """
    target_score = round(before_train["avg_score"] + MIN_IMPROVEMENT, 3)
    # exp 内相对路径（用于 prompt 里 cwd-relative 引用）
    round_dir_in_exp = round_dir.relative_to(data_home).as_posix()
    # trainer-side 绝对路径
    trainer_readme = ROOT / "README.md"
    trainer_agents = ROOT / "AGENTS.md"
    trainer_eval_protocol = ROOT / "EVAL_PROTOCOL.md"
    trainer_policy = POLICY_SLOT
    trainer_tests = ROOT / "tests"

    return f"""你是 Snake Arena HL 这个 Heuristic Learning 闭环中的一轮优化器。

# 你当前的工作目录（cwd）

{data_home}

这是本次实验（{data_home.name}）的数据目录。prompt 里的"相对路径"全部是从这里算起。
trainer 代码在 {ROOT}/ 下，相关引用都给绝对路径。

# 目标

- 改进显式 Snake 启发式系统，而不是训练神经网络。
- 本轮开始前，当前策略在 train split 上的 avg_score 是：{before_train["avg_score"]}。
- 本轮目标：train avg_score 必须达到 {target_score}（至少提升 {MIN_IMPROVEMENT} 分）。

# 必读项目文件

trainer 侧（绝对路径）：
- {trainer_readme}
- {trainer_eval_protocol}
- {trainer_agents}
- {trainer_policy}   ← 这是 trainer 的运行槽，**你要编辑的就是这个文件**

本实验侧（cwd 相对）：
- heuristic_notes.md
- reports/train-failures.md

# 允许修改的文件

- **{trainer_policy}**（trainer 运行槽，绝对路径——你的所有 policy 改动都在这里）
- heuristic_notes.md（cwd 相对，本实验的跨轮笔记）
- {round_dir_in_exp}/journal.md（cwd 相对，本轮实验日志，下面有协议）
- {round_dir_in_exp}/scripts/（cwd 相对，你保留的诊断脚本写这里）

# 禁止事项

- 不要编辑 trainer 代码（{ROOT}/snake_hl/env.py、eval.py、baselines.py、train.py 等）。
  这些已经在 .claude/settings.local.json 里 deny 掉，CC 工具层会直接拒绝。
- 不要硬编码特定 seed、replay 路径或当前 worst cases。
- 本轮不要针对 eval 做优化（eval 是 held-out）。
- 不要尝试读取你 cwd 之外的其他实验数据目录（已通过 deny 规则隔离）。

# 关于 cwd 下的 policy.py

注意：你的 cwd 下有一个 policy.py 文件（本实验 round 起点的版本），**不要编辑它**。
那是 train.py 每轮结束 copy_out 时回写的目标文件，round 期间它是"陈旧的"，
真正的运行槽是 {trainer_policy}。Edit 那里，不要 Edit 你 cwd 下的 policy.py。

# 实验日志协议（必读，硬性要求）

文件路径：{round_dir_in_exp}/journal.md（train.py 已经为你预创建好了模板）

这是你的长期记忆。conversation context 会被压缩、丢失细节；这个文件不会。
压缩之后你能依赖的只有它。

硬性规则：
1. 开始任何新实验前，先 Read 整个 journal.md。在 "Attempted modifications"
   表里搜索你打算试的改动签名——如果已经存在，禁止重试，跳到下一个方向。
2. 每个实验完成后立即追加一行（不要等结束统一写）。改动签名必须具体到
   "_move_threshold normal branch: snake_len → snake_len - 1" 这种粒度，
   而不是 "adjust threshold"，否则下一次压缩后无法去重。
3. 如果你看到这条 prompt 但 journal.md 已经有非模板内容，说明你是从压缩
   恢复过来的。先把整个 journal 读完，再继续——不要重试已记录失败的方向。

# 累积进步

目标 +{MIN_IMPROVEMENT} 可以是单次大改，也可以是多次小改的累积——由你判断
哪种更适合当前情况。但请注意 session 结束时 {trainer_policy} 应该反映本轮
找到的"最好版本"，避免做完 N 次有效改动后因为最后一次失败连带丢掉前面的
累积。何时落盘、何时合并由你决定，journal.md 里把每次改动的分数记清楚就好。

# 本轮任务

1. 结合 failure report 和当前 policy，诊断 train failures，识别 1–3 个独立的
   失败模式（不要只盯一个 seed）。

2. 做一个连贯的策略改进。改动类型不限——可以是参数调优、tiebreaker 重排，
   也可以是算法/结构层面的改动：新增针对失败模式的判定分支、扩展规划深度、
   引入新的状态特征（连通度、孔洞、密度等）、重构控制流。
   当 baseline 已经经过多轮优化（看 heuristic_notes.md 的历史）时，
   纯参数微调很容易撞天花板，应该认真考虑结构性改动。

3. 保持 policy 可读：新增逻辑变复杂时，拆成有名字的小 helper。
   注意"可读" ≠ "保守"——为支持算法改动而新增 helper 是欢迎的。

4. 更新 heuristic_notes.md（cwd 相对），记录你改了什么、为什么应该泛化、
   有什么风险。这是给未来轮次看的高层笔记，与 journal 的逐条实验记录互补。

5. 必须运行测试和 train 评估（见下方验收命令）。

6. 如果 train avg_score 没有达到 {target_score}，不要结束任务；继续诊断、
   修改、测试、评估，直到达标。

7. 本轮不要运行 eval；eval 是外层在 train 确认提升后做的 held-out check。

# 实验效率（避免常见的反模式）

- **批量参数扫描**：每次 eval 约耗时 130 秒。如果你要在同一处试多个候选值
  （如阈值取 2/3/4/5），不要做 4 次 Edit + eval + revert——这是最慢的方式。
  正确做法是写一个 python3 -c 内联脚本，在同一个 Python 进程里 monkey-patch
  目标函数为不同值，连续调用 evaluate() 多次，打印 (value, score) 列表。
  多个候选只占 1 次进程启动开销，节省 70% 以上 wall time。
  找到 winner 后再用 Edit 一次性写入 {trainer_policy}。

- **禁止 Edit→eval→revert 颠簸**：journal 里如果出现 3 次以上 "Edit A → eval →
  Edit A 回原值"，必须停下来用上面的批量扫描重做。

- **中途刷新 failure report**：累积涨分超过 1 时，重跑
    {VENV_PYTHON} -m snake_hl.failure_report --policy current --split train --limit 5
  看最新瓶颈——之前的失败 seed 可能已经解决，继续优化它就是浪费。

- **历史轮次对照**：你 cwd 下的 runs/ 目录有以前各轮的产物（policy.py 快照、
  heuristic_notes.md 快照、replays）。需要做 round-over-round 对比时直接读这些
  目录，不要靠 git。

# 临时脚本

- 一次性诊断用 python3 -c 内联，跑完即弃。
- 如果确实需要保留脚本，写到 {round_dir_in_exp}/scripts/ 下（cwd 相对），
  不要写到 trainer 根目录或别处。

# 验收命令（cwd 是 {data_home}）

```bash
# pytest 在 trainer 那边，传 tests 目录的绝对路径让 pytest 找到测试
{VENV_PYTHON} -m pytest {trainer_tests}

# eval 默认使用当前运行槽（{trainer_policy}）
{VENV_PYTHON} -m snake_hl.eval --policy current --split train

# 刷新 failure report（写到你 cwd 下的 reports/train-failures.md）
{VENV_PYTHON} -m snake_hl.failure_report --policy current --split train --limit 5
```

# 完成条件

- pytest 通过。
- train avg_score 必须 >= {target_score}（即比本轮开始前至少提升 {MIN_IMPROVEMENT} 分）。
- 如果没有达到这个条件，你必须继续迭代，而不是总结失败后退出。
- {round_dir_in_exp}/journal.md 已经完整记录每一个尝试。

# 完成后请用中文简要总结

- 你改了什么 heuristic（标注是参数 / 结构 / 算法类型）。
- 为什么它应该能泛化，而不是只适配某个 seed。
- 你运行了哪些命令。
- 修改后 train avg_score 是多少，比 {before_train["avg_score"]} 提升了多少。
- 你认为下一轮应该观察什么。

本轮产物目录（cwd 相对）：{round_dir_in_exp}
"""


def write_journal_template(journal_path: Path, round_index: int, before_train: dict) -> None:
    """在 round 开始时预创建实验日志模板（optimizer 的长期记忆载体）。"""
    target = round(before_train["avg_score"] + MIN_IMPROVEMENT, 3)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(
        f"""# Round {round_index} Experiment Journal

Goal: train avg_score {before_train["avg_score"]} → >= {target} (+{MIN_IMPROVEMENT} required)

## Rules

- Before any new experiment: Read this entire file.
  Search "Attempted modifications" for your planned change.
  If present, skip — don't retry what already failed.
- After each experiment: immediately append a row to the table below
  (don't wait until round end).
- Change signature must be specific enough to dedupe across conversation
  compactions. Examples:
  - Good: `_move_threshold normal branch: snake_len → snake_len - 1`
  - Good: `_simulate_survival horizon: 21 → 30`
  - Bad:  `adjust threshold` (too vague — will dedupe-fail after compaction)
- Type column: `P` = parametric (numeric / tiebreaker / boolean tweak),
  `S` = structural (new def, new branch, new state feature, control-flow change).

## Attempted modifications

| Time (UTC) | Type | Change signature | Score | Δ vs prior | Decision |
|------------|------|------------------|-------|------------|----------|

## Failed directions summary

(Append one line here when abandoning a direction, summarizing why.)

""",
        encoding="utf-8",
    )


def snapshot_round_artifacts(round_dir: Path, data_home: Path) -> None:
    """
    把当轮结束时的 policy.py、heuristic_notes.md、replays 拷贝到 round_dir。
    类比 DL：每个 epoch 结束时存模型权重 checkpoint + 推理样本。
    """
    # policy.py 来自 trainer 运行槽（当前最新）
    shutil.copy2(POLICY_SLOT, round_dir / "policy.py")
    # heuristic_notes.md 在 exp 目录
    notes_src = data_home / "heuristic_notes.md"
    if notes_src.exists():
        shutil.copy2(notes_src, round_dir / "heuristic_notes.md")
    # replays/current/ 在 exp 目录
    replays_src = data_home / "replays" / "current"
    replays_dst = round_dir / "replays"
    if replays_src.exists():
        if replays_dst.exists():
            shutil.rmtree(replays_dst)
        shutil.copytree(replays_src, replays_dst)


# ============================================================
# Optimizer subprocess
# ============================================================


def require_claude_env() -> None:
    env = claude_env()
    api_key = env.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "replace-with-your-api-key":
        raise RuntimeError(
            "Missing required environment variable: ANTHROPIC_API_KEY. "
            "Put it in .env or export it before running optimizer mode."
        )


def run_claude_optimizer(prompt: str, output: Path, max_budget_usd: float, cwd: Path) -> None:
    """
    cwd 指向 experiments/<exp>/。这样 optimizer 启动时会读到该目录下的
    .claude/settings.local.json，被 deny 访问兄弟实验和归档。

    --permission-mode dontAsk：白名单自动批准，未在 allow 列表里的工具自动拒绝
                               （不弹窗等用户确认）。比 --dangerously-skip-permissions
                               安全得多——后者直接关掉权限层。
    """
    require_claude_env()
    env = claude_env()
    # --model 传 alias "sonnet"（不是具体 model ID），让 Claude CLI 经由 .env 里的
    # ANTHROPIC_DEFAULT_SONNET_MODEL 重映射到 provider 实际模型。
    # 选 sonnet 而不是 opus 的理由：sonnet 是 200K 上下文，opus 是 1M；
    # Kimi 和 Mimo 等第三方 provider 通常只提供 200K 级模型，传 opus 会被拒。
    # 硬编码具体 model ID（如 claude-sonnet-4-6）会让某些 provider 返回 400。
    result = subprocess.run(
        [
            "claude",
            "--print",
            "--permission-mode",
            "dontAsk",
            "--model",
            "sonnet",
            "--max-budget-usd",
            str(max_budget_usd),
            prompt,
        ],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    output.write_text(
        "STDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude optimizer failed ({result.returncode}). See {output}")


def assert_optimizer_boundary(changes: Iterable[str]) -> None:
    illegal = sorted(path for path in changes if path not in ALLOWED_OPTIMIZER_EDITS)
    if illegal:
        formatted = "\n".join(f"- {path}" for path in illegal)
        raise RuntimeError(f"Optimizer changed files outside the allowed boundary:\n{formatted}")


# ============================================================
# Training curve（每实验一份，写在 data_home 下）
# ============================================================


def update_training_curve(
    data_home: Path,
    round_index: int,
    before_train: dict,
    after_train: dict,
    after_eval: dict,
) -> None:
    """追加一轮结果到训练曲线 JSON 并重新生成 HTML。每个实验有自己的曲线。

    同时刷新 experiments/all-curves.html 跨实验聚合视图。
    """
    curve_json = data_home / "training_curve.json"
    curve_html = data_home / "training_curve.html"

    records: list[dict] = []
    if curve_json.exists():
        records = json.loads(curve_json.read_text(encoding="utf-8"))

    global_round = len(records) + 1
    records.append({
        "round": global_round,
        "train_before": round(before_train["avg_score"], 3),
        "train_after": round(after_train["avg_score"], 3),
        "eval_after": round(after_eval["avg_score"], 3),
        "train_delta": round(after_train["avg_score"] - before_train["avg_score"], 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    curve_json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    generate_training_curve_html(records, curve_html, data_home.name)
    update_aggregate_curve()


def generate_training_curve_html(records: list[dict], curve_html: Path, exp_name: str) -> None:
    """用 Chart.js 生成训练曲线 HTML。"""
    labels = [f"R{r['round']}" for r in records]
    train_data = [r["train_after"] for r in records]
    eval_data = [r.get("eval_after") for r in records]

    labels_js = json.dumps(labels)
    train_js = json.dumps(train_data)
    eval_js = json.dumps(eval_data)

    valid_scores = [s for s in train_data + eval_data if s is not None]
    y_min = round(min(valid_scores) * 0.97) if valid_scores else 0
    y_max = round(max(valid_scores) * 1.02) if valid_scores else 800

    def row(r: dict) -> str:
        delta = r.get("train_delta")
        before = r.get("train_before")
        ts = r.get("timestamp", "")
        before_cell = str(before) if before is not None else "—"
        delta_cell = f'<td class="{"pos" if delta >= 0 else "neg"}">{delta:+.3f}</td>' if delta is not None else "<td>—</td>"
        eval_cell = str(r["eval_after"]) if r.get("eval_after") is not None else "—"
        return (
            f'<tr><td>{r["round"]}</td>'
            f'<td class="time" data-ts="{ts}"></td>'
            f'<td>{before_cell}</td>'
            f'<td>{r["train_after"]}</td>'
            f'{delta_cell}'
            f'<td>{eval_cell}</td></tr>'
        )

    timestamps = [r.get("timestamp", "") for r in records]
    timestamps_js = json.dumps(timestamps)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Snake HL Training Curve — {exp_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; padding: 24px; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 8px; }}
  .meta {{ font-size: 0.85rem; color: #aaa; margin-bottom: 20px; }}
  canvas {{ max-width: 900px; }}
  table {{ border-collapse: collapse; margin-top: 24px; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #444; padding: 6px 12px; text-align: right; }}
  th {{ background: #2a2a4e; }}
  tr:hover td {{ background: #2a2a3e; }}
  .pos {{ color: #6ef59a; }} .neg {{ color: #f56e6e; }}
  td.time {{ color: #aaa; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>Snake Arena HL — Training Curve ({exp_name})</h1>
<p class="meta">Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp; {len(records)} rounds</p>
<canvas id="chart"></canvas>
<script>
const labels = {labels_js};
const trainData = {train_js};
const evalData = {eval_js};
const timestamps = {timestamps_js};

function fmtTime(ts) {{
  if (!ts) return '';
  return new Date(ts).toLocaleString('zh-CN', {{
    month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false
  }});
}}

new Chart(document.getElementById('chart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'Train avg_score',
        data: trainData,
        borderColor: '#4fc3f7',
        backgroundColor: 'rgba(79,195,247,0.1)',
        pointRadius: 5,
        tension: 0.3,
      }},
      {{
        label: 'Eval avg_score',
        data: evalData,
        borderColor: '#a5d6a7',
        backgroundColor: 'rgba(165,214,167,0.1)',
        pointRadius: 5,
        tension: 0.3,
        spanGaps: false,
      }},
    ],
  }},
  options: {{
    scales: {{
      y: {{ min: {y_min}, max: {y_max}, grid: {{ color: '#333' }}, ticks: {{ color: '#ccc' }} }},
      x: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#ccc', maxRotation: 0 }} }},
    }},
    plugins: {{
      legend: {{ labels: {{ color: '#eee' }} }},
      tooltip: {{
        mode: 'index',
        intersect: false,
        callbacks: {{
          afterTitle: (items) => {{
            const ts = timestamps[items[0].dataIndex];
            return ts ? fmtTime(ts) : '';
          }},
        }},
      }},
    }},
  }},
}});

document.querySelectorAll('td.time').forEach(el => {{
  el.textContent = fmtTime(el.dataset.ts);
}});
</script>
<table>
<tr><th>Round</th><th>Time</th><th>Train before</th><th>Train after</th><th>Delta</th><th>Eval after</th></tr>
{"".join(row(r) for r in records)}
</table>
</body>
</html>
"""
    curve_html.write_text(html, encoding="utf-8")


def update_aggregate_curve() -> None:
    """
    扫描 experiments/*/training_curve.json，把所有实验的曲线画在一个 HTML 里，
    便于跨实验对比（类比 wandb 的 project view）。

    输出到 experiments/all-curves.html。每个实验一种颜色，train 实线 + eval 虚线。
    X 轴是 round 号；不同实验长度不同时，短的会留空白（spanGaps=false）。
    没有任何实验数据时跳过（不会写空文件）。
    """
    if not DATA_HOME_BASE.exists():
        return

    experiments_data: list[dict] = []
    for exp_dir in sorted(DATA_HOME_BASE.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        curve_json = exp_dir / "training_curve.json"
        if not curve_json.exists():
            continue
        try:
            records = json.loads(curve_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if records:
            experiments_data.append({"name": exp_dir.name, "records": records})

    if not experiments_data:
        return

    max_round = max(max(r["round"] for r in exp["records"]) for exp in experiments_data)
    labels = [f"R{i + 1}" for i in range(max_round)]

    # 暗色背景上效果较好的 8 色调色板（与 Tailwind 接近的中等亮度色）
    palette = [
        "#4fc3f7", "#a5d6a7", "#ffb74d", "#f06292",
        "#ba68c8", "#7986cb", "#e57373", "#90a4ae",
    ]

    datasets: list[dict] = []
    all_scores: list[float] = []
    summary_rows: list[str] = []
    for i, exp in enumerate(experiments_data):
        color = palette[i % len(palette)]
        records = exp["records"]
        train_data: list[float | None] = [None] * max_round
        eval_data: list[float | None] = [None] * max_round
        for r in records:
            idx = r["round"] - 1
            if 0 <= idx < max_round:
                train_data[idx] = r.get("train_after")
                eval_data[idx] = r.get("eval_after")
                for v in (r.get("train_after"), r.get("eval_after")):
                    if v is not None:
                        all_scores.append(v)
        datasets.append({
            "label": f"{exp['name']} train",
            "data": train_data,
            "borderColor": color,
            "backgroundColor": "transparent",
            "pointRadius": 3,
            "tension": 0.3,
            "spanGaps": False,
        })
        datasets.append({
            "label": f"{exp['name']} eval",
            "data": eval_data,
            "borderColor": color,
            "backgroundColor": "transparent",
            "borderDash": [5, 5],
            "pointRadius": 3,
            "tension": 0.3,
            "spanGaps": False,
        })

        # 汇总表行：最后一轮的状态
        last = records[-1]
        summary_rows.append(
            f'<tr>'
            f'<td><span class="swatch" style="background:{color}"></span>{exp["name"]}</td>'
            f'<td>{len(records)}</td>'
            f'<td>{last.get("train_after", "—")}</td>'
            f'<td>{last.get("eval_after", "—")}</td>'
            f'<td class="time" data-ts="{last.get("timestamp", "")}"></td>'
            f'</tr>'
        )

    y_min = round(min(all_scores) * 0.97) if all_scores else 0
    y_max = round(max(all_scores) * 1.02) if all_scores else 800

    labels_js = json.dumps(labels)
    datasets_js = json.dumps(datasets)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Snake HL — All Experiments</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; padding: 24px; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 8px; }}
  .meta {{ font-size: 0.85rem; color: #aaa; margin-bottom: 20px; }}
  canvas {{ max-width: 1100px; }}
  table {{ border-collapse: collapse; margin-top: 24px; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #444; padding: 6px 12px; text-align: right; }}
  th {{ background: #2a2a4e; }}
  td:first-child {{ text-align: left; }}
  tr:hover td {{ background: #2a2a3e; }}
  td.time {{ color: #aaa; font-size: 0.8rem; }}
  .swatch {{
    display: inline-block; width: 10px; height: 10px;
    margin-right: 6px; border-radius: 2px; vertical-align: middle;
  }}
  .legend-note {{ color: #888; font-size: 0.8rem; margin-top: 8px; }}
</style>
</head>
<body>
<h1>Snake Arena HL — All Experiments</h1>
<p class="meta">Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp; {len(experiments_data)} experiments</p>
<canvas id="chart"></canvas>
<p class="legend-note">实线 = train avg_score，虚线 = eval avg_score。同色 = 同一实验。</p>
<script>
const labels = {labels_js};
const datasets = {datasets_js};

function fmtTime(ts) {{
  if (!ts) return '';
  return new Date(ts).toLocaleString('zh-CN', {{
    month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false
  }});
}}

new Chart(document.getElementById('chart'), {{
  type: 'line',
  data: {{ labels, datasets }},
  options: {{
    scales: {{
      y: {{ min: {y_min}, max: {y_max}, grid: {{ color: '#333' }}, ticks: {{ color: '#ccc' }} }},
      x: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#ccc', maxRotation: 0 }} }},
    }},
    plugins: {{
      legend: {{ labels: {{ color: '#eee', boxWidth: 14 }} }},
      tooltip: {{ mode: 'index', intersect: false }},
    }},
  }},
}});

document.querySelectorAll('td.time').forEach(el => {{
  el.textContent = fmtTime(el.dataset.ts);
}});
</script>
<table>
<tr><th>Experiment</th><th>Rounds</th><th>Latest train</th><th>Latest eval</th><th>Last updated</th></tr>
{"".join(summary_rows)}
</table>
</body>
</html>
"""
    (DATA_HOME_BASE / "all-curves.html").write_text(html, encoding="utf-8")


# ============================================================
# 单轮编排
# ============================================================


def run_round(
    *,
    run_dir: Path,
    round_index: int,
    optimizer: str,
    dry_run: bool,
    max_budget_usd: float,
    data_home: Path,
    cached_before_train: dict | None = None,
) -> dict:
    """
    单轮 HL 优化流程：

      ① 前向推理     train set 评分（= loss_before）
      ② 误差分析     生成 failure report = hard examples
      ③ 梯度信号     写 journal 模板 + 构建 prompt + 生成 .claude/settings.local.json
      ④ 参数更新     Claude 反复迭代直到 train 分超过基准
      ⑤ 参数约束检查 hash 对比 trainer 代码
      ⑥ 加载新参数   reload policy 模块
      ⑦ 回归测试     pytest
      ⑧ 训练集评分   after_train
      ⑨ 验证集评分   eval split
      ⑩ 存档         快照 + 训练曲线 + copy_out
    """
    round_dir = run_dir / f"round-{round_index:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    failure_report = data_home / "reports" / "train-failures.md"

    # ── ① 前向推理 ───────────────────────────
    if cached_before_train is not None:
        before_train = cached_before_train
    else:
        before_train, _ = eval_module.evaluate("current", "train")
        before_train["run_type"] = "train_before_optimizer"
        before_train["round"] = round_index
    write_json(round_dir / "train-before.json", before_train)

    # ── ② 误差分析 ───────────────────────────
    failure_report_module.write_failure_report("current", "train", 5, failure_report)

    # ── ③ 梯度信号 ───────────────────────────
    write_journal_template(round_dir / "journal.md", round_index, before_train)
    write_optimizer_permissions(data_home)
    prompt = build_optimizer_prompt(
        data_home=data_home,
        round_dir=round_dir,
        round_index=round_index,
        before_train=before_train,
        failure_report=failure_report,
    )
    prompt_path = round_dir / "optimizer-prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    if dry_run or optimizer == "none":
        return {
            "round": round_index,
            "status": "prepared",
            "prompt": str(prompt_path),
            "before_train": before_train,
        }

    # ── ④ 参数更新（cwd=data_home，optimizer 读 exp dir 的 settings 被 deny）──
    before_hashes = snapshot_repo()
    if optimizer == "claude":
        run_claude_optimizer(prompt, round_dir / "claude-output.txt", max_budget_usd, cwd=data_home)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # ── ⑤ 参数约束 ────────────────────────────
    after_hashes = snapshot_repo()
    optimizer_changes = changed_files(before_hashes, after_hashes)
    assert_optimizer_boundary(optimizer_changes)

    # ── ⑥ 加载新参数 ──────────────────────────
    reload_policy_modules()

    # ── ⑦ 回归测试 ────────────────────────────
    run_command([sys.executable, "-m", "pytest"], output=round_dir / "pytest.txt")

    # ── ⑧ 训练集评分 ──────────────────────────
    after_train, _ = eval_module.evaluate("current", "train")
    after_train["run_type"] = "train_after_optimizer"
    after_train["round"] = round_index
    eval_module.append_trial(after_train, data_home / "trials.jsonl")
    write_json(round_dir / "train-after.json", after_train)

    # ── ⑨ 验证集评分 ──────────────────────────
    after_eval, _ = eval_module.evaluate("current", "eval")
    after_eval["run_type"] = "eval_after_optimizer"
    after_eval["round"] = round_index
    eval_module.append_trial(after_eval, data_home / "trials.jsonl")
    write_json(round_dir / "eval-after.json", after_eval)

    improvement = after_train["avg_score"] - before_train["avg_score"]
    summary = {
        "round": round_index,
        "status": "completed",
        "optimizer_changes": optimizer_changes,
        "before_train": before_train,
        "after_train": after_train,
        "after_eval": after_eval,
        "train_avg_score_delta": round(improvement, 3),
    }

    # ── ⑩ 存档 ────────────────────────────────
    failure_report_module.write_failure_report("current", "train", 5, failure_report)
    snapshot_round_artifacts(round_dir, data_home)
    update_training_curve(data_home, round_index, before_train, after_train, after_eval)
    write_json(round_dir / "summary.json", summary)

    # 每轮单文件 copy_out：把运行槽里的 policy.py 持久化回 exp 目录
    copy_out_policy(data_home)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Snake Arena Heuristic Learning loop.")
    parser.add_argument(
        "--exp",
        type=str,
        help=f"Experiment name; data lives at {DATA_HOME_BASE}/<exp>/ (required unless --aggregate-only).",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Regenerate experiments/all-curves.html from existing training_curve.json files and exit.",
    )
    parser.add_argument(
        "--new-from",
        type=str,
        metavar="BASE",
        help=(
            "Create a fresh experiment by forking from this existing experiment "
            "(its .git is excluded so the new exp starts without remote/history). "
            "Fails if --exp already exists."
        ),
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--optimizer", choices=("claude", "none"), default="claude")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompt without invoking optimizer.")
    parser.add_argument("--max-budget-usd", type=float, default=999.0)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    if args.aggregate_only:
        update_aggregate_curve()
        target = DATA_HOME_BASE / "all-curves.html"
        print(f"Aggregate curve regenerated: {target}")
        return

    if not args.exp:
        parser.error("--exp is required (unless --aggregate-only)")

    data_home = data_home_for(args.exp)

    if args.new_from:
        if data_home.exists():
            raise RuntimeError(
                f"--new-from cannot overwrite an existing experiment at {data_home}. "
                f"Use a fresh --exp name, or remove that directory manually first."
            )
        base = data_home_for(args.new_from)
        validate_data_home(base)
        shutil.copytree(base, data_home, ignore=shutil.ignore_patterns(".git", ".claude"))
        print(f"Forked experiment '{args.exp}' from '{args.new_from}' at {data_home}")

    validate_data_home(data_home)
    copy_in_policy(data_home)
    _load_snake_hl_modules()  # snake_hl.policy 现在存在，可以 import 了
    print(f"Loaded experiment '{args.exp}' from {data_home}")

    # 每次运行创建带时间戳的 run dir（在 exp 目录下）
    if args.run_dir:
        run_dir = args.run_dir if args.run_dir.is_absolute() else data_home / args.run_dir
    else:
        run_dir = data_home / "runs" / utc_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    cached_before_train: dict | None = None
    try:
        for round_index in range(1, args.rounds + 1):
            summary = run_round(
                run_dir=run_dir,
                round_index=round_index,
                optimizer=args.optimizer,
                dry_run=args.dry_run,
                max_budget_usd=args.max_budget_usd,
                data_home=data_home,
                cached_before_train=cached_before_train,
            )
            cached_before_train = summary.get("after_train")
            summaries.append(summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        # 即使中途崩溃，把当前运行槽的 policy.py 拷回去
        if not args.dry_run:
            try:
                copy_out_policy(data_home)
            except Exception as e:
                print(f"WARNING: final copy_out failed: {e}", file=sys.stderr)

    write_json(run_dir / "run-summary.json", summaries)
    print(f"run_dir={run_dir}")
    print(f"Data home: {data_home}")


if __name__ == "__main__":
    main()
