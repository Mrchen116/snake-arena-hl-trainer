from __future__ import annotations

# ============================================================
# train.py — Heuristic Learning 多轮训练编排器
#
# 数据/代码同仓布局：
#   ~/Repos/snake-arena-hl/                trainer code
#   ├── train.py
#   ├── snake_hl/                          trainer modules
#   │   ├── env.py, eval.py, baselines.py, ...
#   │   └── policy_runtime.py              ← 动态 loader，读 SNAKE_POLICY_PATH
#   ├── experiments/                       ← gitignored
#   │   ├── <exp-A>/                       ← 一个实验 = 一个目录（可独立 git）
#   │   │   ├── policy.py                  ← canonical state + runtime（agent 直接 edit）
#   │   │   ├── heuristic_notes.md
#   │   │   ├── runs/, replays/, reports/
#   │   │   ├── trials.jsonl, training_curve.{json,html}
#   │   │   └── .claude/settings.local.json   train.py 启动时重新生成
#   │   └── <exp-B>/ ...
#   └── tests/
#
# 工作流（无全局 slot，支持多实验并发）：
#   1. train.py 启动：os.environ["SNAKE_POLICY_PATH"] = experiments/<exp>/policy.py
#   2. trainer-side import snake_hl.eval / failure_report / replay 时，
#      policy_runtime 通过该 env var 加载本 exp 的 policy 文件
#   3. 写 experiments/<exp>/.claude/settings.local.json（动态枚举 trainer 模块 + siblings）
#   4. 启动 optimizer subprocess，cwd=experiments/<exp>/，env 继承 SNAKE_POLICY_PATH
#      - optimizer 直接编辑 cwd 下的 policy.py（即 experiments/<exp>/policy.py）
#      - 其他 trainer 路径被 settings + sandbox 双层 deny
#   5. 每轮结束：policy_runtime.reload() 让 trainer 进程内 import 看到新代码
#
# 并发安全：每个 train.py 进程关心自己的 SNAKE_POLICY_PATH。不再有共享 slot。
#
# 【DL 类比】
#   模型参数 (weights)    experiments/<exp>/policy.py（canonical + runtime 合二为一）
#   损失函数 (loss)       -avg_score
#   优化器 (optimizer)    Claude Code CLI
#   一个 epoch            一轮 run_round()
#   模型检查点            每轮结束在 runs/<ts>/round-NN/policy.py 留快照
# ============================================================

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

# 这三个模块都依赖 SNAKE_POLICY_PATH，main() 设好 env 后才 import。
eval_module = None  # type: ignore
failure_report_module = None  # type: ignore
policy_runtime_module = None  # type: ignore


def _load_snake_hl_modules() -> None:
    """SNAKE_POLICY_PATH 设置好之后调用，绑定 trainer-side 模块到全局。"""
    global eval_module, failure_report_module, policy_runtime_module
    from snake_hl import eval as _em
    from snake_hl import failure_report as _frm
    from snake_hl import policy_runtime as _prm
    eval_module = _em
    failure_report_module = _frm
    policy_runtime_module = _prm


ROOT = Path(__file__).resolve().parent

# 实验数据目录根。固定在 trainer 内部，不允许通过环境变量覆盖。
DATA_HOME_BASE = ROOT / "experiments"

# Trainer venv 的 python 绝对路径，optimizer prompt + permissions allow 列表都要用到
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

# 旧 monolithic 仓的归档目录。出现时连同兄弟实验一起 deny。
ARCHIVE_DIR = Path.home() / "Repos" / "snake-arena-hl-archive"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


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
    """Optimizer 编辑 policy.py 后，让 trainer 进程内 import 看到新代码。

    eval.POLICIES["current"] 持有的是 policy_runtime.choose_action 这个函数对象，
    内部按 lookup-on-call 方式读 policy_runtime._module，所以只需 reload runtime。
    """
    policy_runtime_module.reload()


def load_dotenv(path: Path = ROOT / ".env") -> dict[str, str]:
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def claude_env() -> dict[str, str]:
    """构造 Claude CLI 子进程的环境变量。

    .env 提供默认值，shell 当前 environment 覆盖。
    SNAKE_POLICY_PATH 由 main() 写入 os.environ，所以这里自动透传给子进程。
    """
    return {**load_dotenv(), **os.environ}


MIN_IMPROVEMENT = 2.0


# ============================================================
# 实验目录校验
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


# ============================================================
# Optimizer 权限配置
# ============================================================


def write_optimizer_permissions(data_home: Path) -> None:
    """
    生成 experiments/<exp>/.claude/settings.local.json。

    分层防御：
    1. permission 层：Edit/Write 精确到自己的 exp dir（含 policy.py）；
       trainer .py 全部 Edit deny；Bash 工具整体放行（边界由 sandbox 收）。
    2. sandbox 层（OS 级 Seatbelt/bubblewrap）：
       - autoAllowBashIfSandboxed: bash 子进程不再被 prefix 规则拒，
         复合命令 / 重定向 / heredoc / run_in_background 一概放行。
       - filesystem.denyRead 整个 experiments/ + ARCHIVE，allowRead 只开自己 exp dir，
         即使 bash 用 `python -c "open(...)"` 也读不到兄弟实验数据。
       - sandbox 默认只让写 cwd（即 exp dir）；trainer dir 的 Edit deny
         同步成 sandbox denyWrite。

    回归测试见 tests/test_sandbox.py（pytest -m sandbox）。
    """
    # 所有 trainer .py 都禁止编辑（包括 policy_runtime.py）
    trainer_pkg = ROOT / "snake_hl"
    trainer_deny_edits = sorted(trainer_pkg.glob("*.py"))

    deny: list[str] = []
    for py in trainer_deny_edits:
        deny.append(f"Edit(/{py})")
    deny.append(f"Edit(/{ROOT}/train.py)")
    # 旧 monolithic 仓归档目录：sandbox.denyRead 是主防线，permission Read deny
    # 同时挡 CC 内置 Read/Glob/Grep 工具。
    if ARCHIVE_DIR.exists():
        deny.append(f"Read(/{ARCHIVE_DIR}/**)")
        deny.append(f"Glob(/{ARCHIVE_DIR}/**)")
        deny.append(f"Grep(/{ARCHIVE_DIR}/**)")

    allow = [
        # Bash / Monitor 整体放行；真实边界由 sandbox 在 OS 层强制。
        "Bash",
        "Monitor",
        # Edit/Write 只开 exp dir（包含 policy.py）。
        f"Edit(/{data_home}/**)",
        f"Write(/{data_home}/**)",
        # Read：CC 内置 Read 工具整体放行；sibling 实验由 sandbox.denyRead 隔离。
        "Read",
        "Glob",
        "Grep",
    ]

    sandbox_deny_read = [str(DATA_HOME_BASE)]
    if ARCHIVE_DIR.exists():
        sandbox_deny_read.append(str(ARCHIVE_DIR))

    settings = {
        "permissions": {
            "defaultMode": "dontAsk",
            "allow": allow,
            "deny": deny,
            "disableBypassPermissionsMode": "disable",
        },
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "failIfUnavailable": True,
            "filesystem": {
                "denyRead": sandbox_deny_read,
                "allowRead": [str(data_home)],
            },
        },
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
    optimizer 的 cwd 是 data_home (experiments/<exp>/)，policy.py 就在 cwd 下，
    所以 "编辑 policy.py" 全用 cwd 相对引用。
    trainer 代码（README/AGENTS）用绝对路径，验证命令用 trainer venv 绝对路径。
    """
    target_score = round(before_train["avg_score"] + MIN_IMPROVEMENT, 3)
    round_dir_in_exp = round_dir.relative_to(data_home).as_posix()
    trainer_readme = ROOT / "README.md"
    trainer_agents = ROOT / "AGENTS.md"
    trainer_eval_protocol = ROOT / "EVAL_PROTOCOL.md"
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

trainer 侧（绝对路径，只读）：
- {trainer_readme}
- {trainer_eval_protocol}
- {trainer_agents}

本实验侧（cwd 相对，可读写）：
- policy.py   ← **你要编辑的就是这个文件**（cwd 下，本实验的策略）
- heuristic_notes.md
- reports/train-failures.md

# 允许修改的文件

- **policy.py**（cwd 下，本实验的策略——你的所有 policy 改动都在这里）
- heuristic_notes.md（cwd 相对，本实验的跨轮笔记）
- {round_dir_in_exp}/journal.md（cwd 相对，本轮实验日志，下面有协议）
- {round_dir_in_exp}/scripts/（cwd 相对，你保留的诊断脚本写这里）

# 禁止事项

- 不要编辑 trainer 代码（{ROOT}/snake_hl/*.py、train.py 等）。
  settings.local.json 里的 Edit/Write allow 只覆盖本实验目录，
  trainer 路径在 dontAsk 模式下会被 CC 工具层自动拒绝。
- 不要硬编码特定 seed、replay 路径或当前 worst cases。
- 本轮不要针对 eval 做优化（eval 是 held-out）。
- 不要尝试读取你 cwd 之外的其他实验数据目录（已通过 deny 规则隔离）。

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
哪种更适合当前情况。但请注意 session 结束时 cwd 下的 policy.py 应该反映本轮
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
  找到 winner 后再用 Edit 一次性写入 policy.py。

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

trainer 子进程已通过 SNAKE_POLICY_PATH 环境变量指向 cwd 下的 policy.py，
所以 `--policy current` 自动加载你正在编辑的文件。

```bash
# pytest 在 trainer 那边，传 tests 目录的绝对路径让 pytest 找到测试
{VENV_PYTHON} -m pytest {trainer_tests}

# eval 默认使用 SNAKE_POLICY_PATH 指向的 policy.py（即 cwd 下的）
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
    # policy.py 直接在 exp 目录里（没有运行槽了），从那里拷
    shutil.copy2(data_home / "policy.py", round_dir / "policy.py")
    notes_src = data_home / "heuristic_notes.md"
    if notes_src.exists():
        shutil.copy2(notes_src, round_dir / "heuristic_notes.md")
    replays_src = data_home / "replays" / "current"
    replays_dst = round_dir / "replays"
    if replays_src.exists():
        if replays_dst.exists():
            shutil.rmtree(replays_dst)
        shutil.copytree(replays_src, replays_dst)


# ============================================================
# Optimizer subprocess
# ============================================================


# 触发 _run_claude 自动重试的瞬时错误特征。
# 仅匹配明确属于"网络/服务侧抖动、与本地代码无关"的 token。
_TRANSIENT_CLAUDE_ERROR_TOKENS = (
    "ECONNRESET",
    "ETIMEDOUT",
    "ENETUNREACH",
    "ENOTFOUND",
    "EAI_AGAIN",
    "socket hang up",
    "Unable to connect to API",
    "Connection error",
    "fetch failed",
    "Overloaded",
    "rate_limit_error",
    "503 Service Unavailable",
    "502 Bad Gateway",
    "504 Gateway Timeout",
)

_MAX_NETWORK_RETRIES = 3
_NETWORK_BACKOFF_SECONDS = (15, 45, 120)


def _is_transient_claude_error(combined_output: str) -> bool:
    if not combined_output:
        return False
    return any(tok in combined_output for tok in _TRANSIENT_CLAUDE_ERROR_TOKENS)


def _build_resume_retry_args(original_args: list[str], session_id: str) -> list[str]:
    """从原始 args 派生出"续 session"的重试 args。

    - 删掉旧的 `--resume <id>`（如果有）
    - 把末尾的位置参数（prompt）替换成"继续"提示
    - 在 `--print` 之后插入 `--resume <session_id>`
    """
    args = list(original_args)
    i = 0
    while i < len(args):
        if args[i] == "--resume" and i + 1 < len(args):
            del args[i:i + 2]
            continue
        i += 1
    if args and not args[-1].startswith("--"):
        args[-1] = "之前网络中断了，现在继续完成。"
    insert_at = 1 if args and args[0] == "--print" else 0
    args[insert_at:insert_at] = ["--resume", session_id]
    return args


def _run_claude(
    args: list[str],
    output: Path,
    cwd: Path,
) -> tuple[str, str | None]:
    """
    底层：执行 claude 子进程，--output-format json，返回 (text_result, session_id)。
    args 是 claude 命令行参数（不含 "claude" 本身）。

    瞬时网络/服务端错误（ECONNRESET / Overloaded / 5xx 等）自动重试：
    若上次拿到了 session_id，就改用 `--resume` 形态续上，避免重头再来。
    """
    env = claude_env()
    current_args = list(args)
    last_session_id: str | None = None

    for attempt in range(_MAX_NETWORK_RETRIES + 1):
        attempt_output = (
            output if attempt == 0
            else output.with_name(f"{output.stem}-netretry{attempt}{output.suffix}")
        )

        result = subprocess.run(
            ["claude", *current_args, "--output-format", "json"],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
        )
        text = result.stdout
        session_id: str | None = None
        is_error: bool | None = None
        json_parsed = False
        try:
            data = json.loads(result.stdout)
            text = data.get("result", result.stdout)
            session_id = data.get("session_id")
            is_error = bool(data.get("is_error", False))
            json_parsed = True
        except (json.JSONDecodeError, AttributeError):
            pass
        if session_id:
            last_session_id = session_id

        attempt_output.write_text(
            "STDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
            encoding="utf-8",
        )

        # 失败判定：优先看 is_error（JSON 语义），fallback 看 returncode（JSON 没解析出来）。
        failed = (is_error if json_parsed else result.returncode != 0)
        if not failed:
            return text, session_id

        # 瞬时判定：JSON 解析成功时只看 result 字段（窄、信噪比高）；
        # JSON 没解出来时回退到 stdout+stderr。
        error_text = text if json_parsed else (
            (result.stdout or "") + "\n" + (result.stderr or "")
        )
        if attempt < _MAX_NETWORK_RETRIES and _is_transient_claude_error(error_text):
            backoff = _NETWORK_BACKOFF_SECONDS[
                min(attempt, len(_NETWORK_BACKOFF_SECONDS) - 1)
            ]
            resume_hint = (
                f", resuming {last_session_id[:8]}…" if last_session_id else ""
            )
            print(
                f"  ↻ Transient Claude API error{resume_hint}; sleeping {backoff}s "
                f"then retry ({attempt + 1}/{_MAX_NETWORK_RETRIES}). "
                f"See {attempt_output}"
            )
            time.sleep(backoff)
            if last_session_id:
                current_args = _build_resume_retry_args(args, last_session_id)
            continue

        raise RuntimeError(
            f"Claude subprocess failed ({result.returncode}). See {attempt_output}"
        )

    raise RuntimeError(
        f"Claude subprocess failed after {_MAX_NETWORK_RETRIES} network retries. "
        f"See {output}"
    )


def run_claude_optimizer(
    prompt: str, output: Path, max_budget_usd: float, cwd: Path
) -> tuple[str, str | None]:
    """
    首次启动优化器。cwd 指向 experiments/<exp>/，optimizer 读到该目录下的
    .claude/settings.local.json，Edit/Write 被限制在 exp dir（含 policy.py）。

    --permission-mode dontAsk：allow 列表内自动批准，不在列表里的自动拒绝。
    --model sonnet：alias 经由 ANTHROPIC_DEFAULT_SONNET_MODEL 重映射到 provider 模型。
    """
    return _run_claude(
        [
            "--print",
            "--permission-mode", "dontAsk",
            "--model", "sonnet",
            "--max-budget-usd", str(max_budget_usd),
            prompt,
        ],
        output=output,
        cwd=cwd,
    )


def send_optimizer_feedback(
    session_id: str,
    feedback: str,
    output: Path,
    cwd: Path,
) -> tuple[str, str | None]:
    """
    用 --resume 把反馈发回给已有 session，让 optimizer 在完整上下文里继续改。
    用于：分数不够时的迭代反馈。
    """
    return _run_claude(
        [
            "--print",
            "--resume", session_id,
            "--permission-mode", "dontAsk",
            "--model", "sonnet",
            feedback,
        ],
        output=output,
        cwd=cwd,
    )


# ============================================================
# Training curve（每实验一份，写在 data_home 下）
# ============================================================


def update_training_curve(
    data_home: Path,
    round_index: int,
    before_train: dict,
    after_train: dict,
    after_eval: dict,
    duration_seconds: float | None = None,
) -> None:
    """追加一轮结果到 experiments/<exp>/training_curve.json。

    Schema 包含完整 DL 面板需要的指标：score / food / survival / steps / deaths / 耗时，
    便于 dashboard/index.html 多面板可视化（不止 avg_score 一项）。

    HTML 不在这里生成——dashboard 是静态 viewer，运行时 fetch JSON。
    """
    curve_json = data_home / "training_curve.json"

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
        "train_food": round(after_train.get("avg_food", 0), 3),
        "eval_food": round(after_eval.get("avg_food", 0), 3),
        "train_survival": round(after_train.get("survival_rate", 0), 4),
        "eval_survival": round(after_eval.get("survival_rate", 0), 4),
        "train_steps": round(after_train.get("avg_steps", 0), 2),
        "eval_steps": round(after_eval.get("avg_steps", 0), 2),
        "train_deaths": after_train.get("death_reasons", {}),
        "eval_deaths": after_eval.get("death_reasons", {}),
        "duration_seconds": round(duration_seconds, 1) if duration_seconds is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    curve_json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def start_dashboard_server(port: int = 8765) -> subprocess.Popen | None:
    """
    起一个 python -m http.server 子进程，bind 127.0.0.1，serve trainer root。
    dashboard/index.html 通过 fetch 拉 experiments/*/training_curve.json 渲染面板。

    返回 Popen 句柄；main() 在 finally 里 terminate 它。
    端口被占用等异常时打印警告并返回 None（不阻断训练）。
    """
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", "--bind", "127.0.0.1", str(port)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"WARNING: dashboard server failed to start: {e}", file=sys.stderr)
        return None
    print(f"📊 Dashboard: http://localhost:{port}/dashboard/")
    return proc


def stop_dashboard_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


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
      ⑤ 加载新参数   policy_runtime.reload()
      ⑥ 回归测试     pytest
      ⑦ 训练集评分   after_train
      ⑧ 验证集评分   eval split
      ⑨ 存档         快照 + 训练曲线
    """
    round_dir = run_dir / f"round-{round_index:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    failure_report = data_home / "reports" / "train-failures.md"
    round_start = datetime.now(timezone.utc)

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

    # ── ④⑤⑥⑦ 优化 + 评分（最多 MAX_FEEDBACK 次反馈循环）──────────
    MAX_FEEDBACK = 3
    session_id: str | None = None
    after_train: dict = {}
    after_eval: dict = {}

    for attempt in range(MAX_FEEDBACK + 1):
        output_file = round_dir / (
            "claude-output.txt" if attempt == 0 else f"claude-output-fb{attempt}.txt"
        )

        if attempt == 0:
            if optimizer == "claude":
                _, session_id = run_claude_optimizer(
                    prompt, output_file, max_budget_usd, cwd=data_home
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            if session_id is None:
                print("  ✗ No session_id to resume (provider may not support --resume).")
                break
            print(f"  → Feedback attempt {attempt}/{MAX_FEEDBACK} (session {session_id[:8]}…)")
            _, session_id = send_optimizer_feedback(
                session_id, feedback_msg, output_file, cwd=data_home
            )

        # ── ⑤ 加载新参数 ────────────────────────
        reload_policy_modules()

        # ── ⑥ 回归测试 ──────────────────────────
        run_command([sys.executable, "-m", "pytest"], output=round_dir / "pytest.txt")

        # ── ⑦ 训练集评分 ────────────────────────
        after_train, _ = eval_module.evaluate("current", "train")
        after_train["run_type"] = "train_after_optimizer"
        after_train["round"] = round_index
        eval_module.append_trial(after_train, data_home / "trials.jsonl")
        write_json(round_dir / "train-after.json", after_train)

        improvement = after_train["avg_score"] - before_train["avg_score"]
        target_score = round(before_train["avg_score"] + MIN_IMPROVEMENT, 3)

        if improvement >= MIN_IMPROVEMENT:
            break  # 达标

        if attempt >= MAX_FEEDBACK:
            raise RuntimeError(
                f"Optimizer failed after {MAX_FEEDBACK} feedback attempts. "
                f"Last improvement: {improvement:+.3f} (target: +{MIN_IMPROVEMENT}). "
                f"Inspect round dir: {round_dir}"
            )

        feedback_msg = (
            f"当前 train avg_score = {after_train['avg_score']:.3f}，"
            f"比本轮开始前提升了 {improvement:+.3f} 分，"
            f"但目标是至少 +{MIN_IMPROVEMENT} 分（达到 {target_score}）。\n\n"
            f"请继续分析 failure report，尝试不同方向。"
            f"若参数微调已到瓶颈，考虑结构性或算法性改动。"
            f"先读 journal.md，不要重试已失败的方向。"
        )

    # ── ⑧ 验证集评分 ──────────────────────────────────────────────────────────
    after_eval, _ = eval_module.evaluate("current", "eval")
    after_eval["run_type"] = "eval_after_optimizer"
    after_eval["round"] = round_index
    eval_module.append_trial(after_eval, data_home / "trials.jsonl")
    write_json(round_dir / "eval-after.json", after_eval)

    improvement = after_train["avg_score"] - before_train["avg_score"]
    summary = {
        "round": round_index,
        "status": "completed",
        "before_train": before_train,
        "after_train": after_train,
        "after_eval": after_eval,
        "train_avg_score_delta": round(improvement, 3),
    }

    duration_seconds = (datetime.now(timezone.utc) - round_start).total_seconds()
    summary["duration_seconds"] = round(duration_seconds, 1)

    # ── ⑨ 存档 ────────────────────────────────
    failure_report_module.write_failure_report("current", "train", 5, failure_report)
    snapshot_round_artifacts(round_dir, data_home)
    update_training_curve(
        data_home, round_index, before_train, after_train, after_eval,
        duration_seconds=duration_seconds,
    )
    write_json(round_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Snake Arena Heuristic Learning loop.")
    parser.add_argument(
        "--exp",
        type=str,
        help=f"Experiment name; data lives at {DATA_HOME_BASE}/<exp>/ (required unless --serve).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the dashboard HTTP server and keep running (no training). Ctrl-C to stop.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8765,
        help="Port for the dashboard HTTP server (default: 8765).",
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

    if args.serve:
        proc = start_dashboard_server(args.dashboard_port)
        if proc is None:
            sys.exit(1)
        try:
            proc.wait()
        except KeyboardInterrupt:
            stop_dashboard_server(proc)
        return

    if not args.exp:
        parser.error("--exp is required (unless --serve)")

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
    elif not data_home.exists():
        baseline = data_home_for("_baseline")
        if not baseline.is_dir() or not (baseline / "policy.py").is_file():
            raise RuntimeError(
                f"Experiment '{args.exp}' not found at {data_home} and no _baseline exists.\n"
                f"Create experiments/_baseline/policy.py first, or use --new-from <existing-exp>."
            )
        shutil.copytree(baseline, data_home, ignore=shutil.ignore_patterns(".git", ".claude"))
        print(f"Auto-forked experiment '{args.exp}' from '_baseline' at {data_home}")

    validate_data_home(data_home)

    # 关键：设置 SNAKE_POLICY_PATH，让 trainer 自身的 import + 所有子进程都看到同一个
    # exp policy.py。多个 train.py 并发跑各自的 SNAKE_POLICY_PATH 互不干扰。
    os.environ["SNAKE_POLICY_PATH"] = str(data_home / "policy.py")

    _load_snake_hl_modules()
    print(f"Loaded experiment '{args.exp}' from {data_home}")

    if args.run_dir:
        run_dir = args.run_dir if args.run_dir.is_absolute() else data_home / args.run_dir
    else:
        run_dir = data_home / "runs" / utc_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    dashboard_proc = None if args.dry_run else start_dashboard_server(args.dashboard_port)

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
        stop_dashboard_server(dashboard_proc)

    write_json(run_dir / "run-summary.json", summaries)
    print(f"run_dir={run_dir}")
    print(f"Data home: {data_home}")


if __name__ == "__main__":
    main()
