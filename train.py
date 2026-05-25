from __future__ import annotations

# ============================================================
# train.py — Heuristic Learning 多轮训练编排器
#
# 数据/代码分离架构：
# - Trainer 仓库（本仓库）：train.py + snake_hl/* 模块 + tests/
# - 数据仓库：~/Repos/snake-data/<exp-name>/
#     包含 policy.py、heuristic_notes.md、runs/、replays/ 等所有可变数据
#
# 每次运行必须指定 --exp <name>。trainer 在调用 optimizer 之前先把
# 数据仓库的内容拷贝进 trainer 的工作槽（snake_hl/policy.py、experiments/*），
# 每轮结束后再拷贝回数据仓库做持久化。trainer 自身不再 auto-commit；
# 数据仓库的 git 提交由用户在数据仓库里手动管理。
#
# 【DL 类比】
#   模型参数 (weights)    snake_hl/policy.py（拷自 data dir）
#   损失函数 (loss)       -avg_score
#   优化器 (optimizer)    Claude Code CLI（读 failure report，改 policy）
#   一个 epoch            一轮 run_round()
#   训练集                train split（200 个固定 seed）
#   验证集                eval split（200 个固定 seed，永远 held-out）
#   梯度                  failure report + Claude 的诊断推理
#   模型检查点            每轮 copy_out 持久化到 ~/Repos/snake-data/<exp>/
#   参数约束 / 梯度裁剪   assert_optimizer_boundary（hash-based）
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

# snake_hl.eval / failure_report / policy 都需要 snake_hl/policy.py 文件存在才能 import。
# 在 train.py 启动时 policy.py 是从数据仓库拷贝进来的，所以这些模块必须**延迟 import**
# ——只有 copy_in() 跑完之后才能用。下面 _load_snake_hl_modules() 会在 main() 里调用。
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
DATA_HOME_BASE = Path.home() / "Repos" / "snake-data"

# Claude CLI 启动时注入的额外环境变量
CLAUDE_ENV_DEFAULTS = {
    "ENABLE_TOOL_SEARCH": "false",
    "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding/",  # 走 Kimi 兼容接口
}

# 数据同步表：每行 (trainer 工作槽相对路径, 数据仓库相对路径)
# copy_in 时从右拷到左，copy_out 时从左拷到右。
SYNC_PATHS: tuple[tuple[str, str], ...] = (
    ("snake_hl/policy.py", "policy.py"),
    ("experiments/heuristic_notes.md", "heuristic_notes.md"),
    ("experiments/runs", "runs"),
    ("experiments/reports", "reports"),
    ("experiments/replays", "replays"),
    ("experiments/replay_viewer", "replay_viewer"),
    ("experiments/trials.jsonl", "trials.jsonl"),
    ("experiments/training_curve.json", "training_curve.json"),
    ("experiments/training_curve.html", "training_curve.html"),
    ("experiments/baseline_snapshot.json", "baseline_snapshot.json"),
    ("experiments/baseline_snapshot.md", "baseline_snapshot.md"),
)

# Optimizer 训练过程中可以修改的文件白名单（hash-based 边界检查使用）
# 其他生成产物目录（runs/、reports/、replays/、replay_viewer/）由 GENERATED_PREFIXES
# 过滤，不参与 hash 比对，因此 optimizer 在那些目录下的活动不受白名单限制。
ALLOWED_OPTIMIZER_EDITS = {
    "snake_hl/policy.py",
    "experiments/heuristic_notes.md",
    "experiments/trials.jsonl",
}

# Hash 比对时跳过这些目录（虚拟环境、缓存、git 等不需要监控）
IGNORED_SNAPSHOT_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "snake_arena_hl.egg-info",
}

# 这些路径下是生成产物，optimizer 修改它们不算越界
GENERATED_PREFIXES = (
    "experiments/replays/",
    "experiments/replay_viewer/",
    "experiments/reports/",
    "experiments/runs/",
)

TRAINING_CURVE_JSON = ROOT / "experiments/training_curve.json"
TRAINING_CURVE_HTML = ROOT / "experiments/training_curve.html"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def repo_path(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    return path.relative_to(ROOT).as_posix()


def should_snapshot(path: Path) -> bool:
    """判断某个文件是否要纳入 hash 快照（用于检测 optimizer 改了哪些文件）。"""
    relative = path.relative_to(ROOT)
    parts = set(relative.parts)
    if parts & IGNORED_SNAPSHOT_PARTS:
        return False
    if any(repo_path(path).startswith(prefix) for prefix in GENERATED_PREFIXES):
        return False
    return path.is_file()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_repo() -> dict[str, str]:
    """对整个 trainer dir（排除忽略项）生成 {相对路径: 哈希}。"""
    return {
        repo_path(path): file_digest(path)
        for path in ROOT.rglob("*")
        if should_snapshot(path)
    }


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    paths = set(before) | set(after)
    return sorted(path for path in paths if before.get(path) != after.get(path))


def run_command(args: list[str], *, output: Path | None = None) -> subprocess.CompletedProcess[str]:
    """执行 shell 命令；如果指定了 output 路径，把 stdout/stderr 写入文件。失败则抛异常。"""
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
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
    """
    在当前 Python 进程内重新加载 policy 模块。
    类比：把 optimizer 更新后的参数加载进推理图（inference graph）。
    """
    importlib.reload(policy_module)
    importlib.reload(eval_module)
    importlib.reload(failure_report_module)


def load_dotenv(path: Path = ROOT / ".env") -> dict[str, str]:
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def claude_env() -> dict[str, str]:
    """合并 .env、当前环境变量和 CLAUDE_ENV_DEFAULTS，作为 Claude CLI 的运行环境。"""
    env = {**load_dotenv(), **os.environ}
    env.update(CLAUDE_ENV_DEFAULTS)
    return env


MIN_IMPROVEMENT = 2.0


# ============================================================
# 数据仓库 ↔ trainer 工作槽 同步
# ============================================================


def data_home_for(exp_name: str) -> Path:
    return DATA_HOME_BASE / exp_name


def validate_data_home(data_home: Path) -> None:
    """确认数据仓库存在且包含必需的 policy.py。"""
    if not data_home.is_dir():
        raise RuntimeError(
            f"Experiment directory not found: {data_home}\n"
            f"To create a fresh experiment, copy from an existing baseline:\n"
            f"  cp -r {DATA_HOME_BASE}/r14-baseline {data_home}\n"
            f"Then re-run with --exp {data_home.name}."
        )
    policy_file = data_home / "policy.py"
    if not policy_file.is_file():
        raise RuntimeError(f"Required file missing in data dir: {policy_file}")


def _replace(src: Path, dst: Path) -> None:
    """把 src 拷到 dst，dst 已存在则先删（文件或目录都处理）。src 不存在则跳过。"""
    if not src.exists():
        return
    if dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_in(data_home: Path) -> None:
    """从数据仓库拷入 trainer 工作槽。在每次 train.py 启动时调用一次。"""
    for trainer_rel, data_rel in SYNC_PATHS:
        _replace(data_home / data_rel, ROOT / trainer_rel)
    # 确保 experiments/ 存在（下游代码会往里写 reports、trials.jsonl 等）
    (ROOT / "experiments").mkdir(parents=True, exist_ok=True)


def copy_out(data_home: Path) -> None:
    """把 trainer 工作槽里的修改持久化回数据仓库。每轮结束都调用。"""
    for trainer_rel, data_rel in SYNC_PATHS:
        _replace(ROOT / trainer_rel, data_home / data_rel)


# ============================================================
# Optimizer prompt + 实验日志模板 + 每轮 snapshot
# ============================================================


def build_optimizer_prompt(
    *,
    round_dir: Path,
    round_index: int,
    before_train: dict,
    failure_report: Path,
) -> str:
    """
    构建发给 Claude optimizer 的任务 prompt。
    类比：把当前 loss 值和 hard examples 打包成"梯度信号"交给优化器。
    prompt 包含：当前 train avg_score、失败案例报告、约束条件、方法论指引。
    """
    target_score = round(before_train["avg_score"] + MIN_IMPROVEMENT, 3)
    round_dir_rel = repo_path(round_dir)
    return f"""你是 Snake Arena HL 这个 Heuristic Learning 闭环中的一轮优化器。

# 目标

- 改进显式 Snake 启发式系统，而不是训练神经网络。
- 本轮开始前，当前策略在 train split 上的 avg_score 是：{before_train["avg_score"]}。
- 本轮目标：train avg_score 必须达到 {target_score}（至少提升 {MIN_IMPROVEMENT} 分）。

# 必读项目文件

- README.md
- EVAL_PROTOCOL.md
- AGENTS.md
- snake_hl/policy.py
- experiments/heuristic_notes.md
- {repo_path(failure_report)}

# 允许修改的文件

- snake_hl/policy.py
- experiments/heuristic_notes.md
- {round_dir_rel}/journal.md（实验日志，下面有协议）
- {round_dir_rel}/scripts/（你保留的临时脚本写这里）

# 禁止事项

- 不要修改 snake_hl/env.py、snake_hl/eval.py、snake_hl/baselines.py、评分公式、train/eval seed 定义。
- 不要手工修改生成产物，比如 reports、replays、replay_viewer、runs 顶层；这些只能通过命令生成。
- 不要硬编码特定 seed、replay 路径或当前 worst cases。
- 本轮不要针对 eval 做优化（eval 是 held-out）。

# 实验日志协议（必读，硬性要求）

文件路径：{round_dir_rel}/journal.md（train.py 已经为你预创建好了模板）

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
哪种更适合当前情况。但请注意 session 结束时 snake_hl/policy.py 应该反映本轮
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

4. 更新 experiments/heuristic_notes.md，记录你改了什么、为什么应该泛化、
   有什么风险。这是给未来轮次看的高层笔记，与 journal 的逐条实验记录互补。

5. 必须运行测试和 train 评估（见下方验收命令）。

6. 如果 train avg_score 没有达到 {target_score}，不要结束任务；继续诊断、
   修改、测试、评估，直到达标。

7. 本轮不要运行 eval；eval 是外层在 train 确认提升后做的 held-out check。

# 实验效率（避免常见的反模式）

- **批量参数扫描**：每次 .venv/bin/python -m snake_hl.eval 约耗时 130 秒。
  如果你要在同一处试多个候选值（如阈值取 2/3/4/5），不要做 4 次 Edit + eval + revert
  ——这是最慢的方式。正确做法是写一个 python3 -c 内联脚本，在同一个 Python
  进程里 monkey-patch 目标函数为不同值，连续调用 evaluate() 多次，打印
  (value, score) 列表。多个候选只占 1 次进程启动开销，节省 70% 以上 wall time。
  找到 winner 后再用 Edit 一次性写入 policy.py。

- **禁止 Edit→eval→revert 颠簸**：journal 里如果出现 3 次以上 "Edit A → eval →
  Edit A 回原值"，必须停下来用上面的批量扫描重做。

- **中途刷新 failure report**：累积涨分超过 1 时，重跑
    .venv/bin/python -m snake_hl.failure_report --policy current --split train --limit 5
  看最新瓶颈——之前的失败 seed 可能已经解决，继续优化它就是浪费。

- **历史轮次对照**：experiments/runs/ 下有以前各轮的产物
  （policy.py 快照、heuristic_notes.md 快照、replays）。需要做 round-over-round
  对比时直接读这些目录，不要靠 git。

# 临时脚本

- 一次性诊断用 python3 -c 内联，跑完即弃。
- 如果确实需要保留脚本（比如可复用的失败分析工具），写到 {round_dir_rel}/scripts/ 下，
  不要写到 repo 根目录。

# 验收命令

- .venv/bin/python -m pytest
- .venv/bin/python -m snake_hl.eval --policy current --split train

# 完成条件

- pytest 通过。
- train avg_score 必须 >= {target_score}（即比本轮开始前至少提升 {MIN_IMPROVEMENT} 分）。
- 如果没有达到这个条件，你必须继续迭代，而不是总结失败后退出。
- {round_dir_rel}/journal.md 已经完整记录每一个尝试。

# 完成后请用中文简要总结

- 你改了什么 heuristic（标注是参数 / 结构 / 算法类型）。
- 为什么它应该能泛化，而不是只适配某个 seed。
- 你运行了哪些命令。
- 修改后 train avg_score 是多少，比 {before_train["avg_score"]} 提升了多少。
- 你认为下一轮应该观察什么。

本轮产物目录：{round_dir_rel}
"""


def write_journal_template(journal_path: Path, round_index: int, before_train: dict) -> None:
    """
    在 round 开始时预创建实验日志模板。
    这是 optimizer 在 conversation 压缩之外的长期记忆载体。
    """
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


def snapshot_round_artifacts(round_dir: Path) -> None:
    """
    把当轮结束时的 policy.py、heuristic_notes.md、和 replays 拷贝到 round_dir。
    类比 DL：每个 epoch 结束时存模型权重 checkpoint + 推理样本，供后续做
    round-over-round 对比。
    """
    shutil.copy(ROOT / "snake_hl/policy.py", round_dir / "policy.py")
    shutil.copy(
        ROOT / "experiments/heuristic_notes.md",
        round_dir / "heuristic_notes.md",
    )

    replays_src = ROOT / "experiments/replays/current"
    replays_dst = round_dir / "replays"
    if replays_src.exists():
        if replays_dst.exists():
            shutil.rmtree(replays_dst)
        shutil.copytree(replays_src, replays_dst)


# ============================================================
# Optimizer subprocess
# ============================================================


def require_claude_env() -> None:
    """启动 optimizer 前检查 ANTHROPIC_API_KEY 是否配置。"""
    env = claude_env()
    api_key = env.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "replace-with-your-api-key":
        raise RuntimeError(
            "Missing required environment variable: ANTHROPIC_API_KEY. "
            "Put it in .env or export it before running optimizer mode."
        )


def run_claude_optimizer(prompt: str, output: Path, max_budget_usd: float) -> None:
    """
    启动 Claude CLI 子进程，执行一轮参数优化。
    --print：非交互模式，完成后直接退出。
    --dangerously-skip-permissions：跳过工具调用确认。
    --max-budget-usd：API 调用费用上限。
    """
    require_claude_env()
    env = claude_env()
    result = subprocess.run(
        [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--model",
            "claude-sonnet-4-6",
            "--max-budget-usd",
            str(max_budget_usd),
            prompt,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    output.write_text(
        "STDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude optimizer failed ({result.returncode}). See {repo_path(output)}")


def assert_optimizer_boundary(changes: Iterable[str]) -> None:
    """
    参数约束检查（类比梯度裁剪 / 参数冻结）。
    确保 optimizer 只"更新"了白名单里的参数文件，没有动 trainer 代码。
    任何越界修改都会立刻抛异常，阻止后续流程。
    """
    illegal = sorted(path for path in changes if path not in ALLOWED_OPTIMIZER_EDITS)
    if illegal:
        formatted = "\n".join(f"- {path}" for path in illegal)
        raise RuntimeError(f"Optimizer changed files outside the allowed boundary:\n{formatted}")


# ============================================================
# Training curve
# ============================================================


def update_training_curve(
    round_index: int,
    before_train: dict,
    after_train: dict,
    after_eval: dict,
) -> None:
    """
    追加一轮结果到训练曲线 JSON，然后重新生成 HTML 图表。
    类比：TensorBoard / wandb 每个 epoch 写入 train_loss 和 val_loss。
    """
    records: list[dict] = []
    if TRAINING_CURVE_JSON.exists():
        records = json.loads(TRAINING_CURVE_JSON.read_text(encoding="utf-8"))

    global_round = len(records) + 1  # 跨 run 的全局轮次编号
    records.append({
        "round": global_round,
        "train_before": round(before_train["avg_score"], 3),
        "train_after": round(after_train["avg_score"], 3),
        "eval_after": round(after_eval["avg_score"], 3),
        "train_delta": round(after_train["avg_score"] - before_train["avg_score"], 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    TRAINING_CURVE_JSON.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    generate_training_curve_html(records)


def generate_training_curve_html(records: list[dict]) -> None:
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
<title>Snake HL Training Curve</title>
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
<h1>Snake Arena HL — Training Curve</h1>
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
    TRAINING_CURVE_HTML.write_text(html, encoding="utf-8")


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
    执行单轮 HL 优化：

      ① 前向推理     train set 评分（= loss_before）
                     多轮运行时直接复用上一轮的 after 分，不重跑
      ② 误差分析     生成 failure report = hard examples
      ③ 梯度信号     写 journal 模板 + 构建 prompt
      ④ 参数更新     Claude 反复迭代直到 train 分超过基准
      ⑤ 参数约束检查 hash 对比，确认只改了白名单文件
      ⑥ 加载新参数   reload policy 模块
      ⑦ 回归测试     pytest
      ⑧ 训练集评分   after_train
      ⑨ 验证集评分   每轮必跑 eval split
      ⑩ 存档         snapshot + 训练曲线 + copy_out 到数据仓库
    """
    round_dir = run_dir / f"round-{round_index:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # ── ① 前向推理：确定 before 基准分 ─────────────────
    if cached_before_train is not None:
        before_train = cached_before_train
    else:
        before_train, _ = eval_module.evaluate("current", "train")
        before_train["run_type"] = "train_before_optimizer"
        before_train["round"] = round_index
    write_json(round_dir / "train-before.json", before_train)

    # ── ② 误差分析：找最差 episode ─────────────────────
    failure_report = ROOT / "experiments/reports/train-failures.md"
    failure_report_module.write_failure_report("current", "train", 5, failure_report)

    # ── ③ 梯度信号：日志模板 + prompt ──────────────────
    write_journal_template(round_dir / "journal.md", round_index, before_train)
    prompt = build_optimizer_prompt(
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
            "prompt": repo_path(prompt_path),
            "before_train": before_train,
        }

    # ── ④ 参数更新：启动 Claude（带内循环的 optimizer step）──
    before_hashes = snapshot_repo()
    if optimizer == "claude":
        run_claude_optimizer(prompt, round_dir / "claude-output.txt", max_budget_usd)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # ── ⑤ 参数约束：hash 对比 ─────────────────────────
    after_hashes = snapshot_repo()
    optimizer_changes = changed_files(before_hashes, after_hashes)
    assert_optimizer_boundary(optimizer_changes)

    # ── ⑥ 加载新参数 ──────────────────────────────────
    reload_policy_modules()

    # ── ⑦ 回归测试 ────────────────────────────────────
    run_command([sys.executable, "-m", "pytest"], output=round_dir / "pytest.txt")

    # ── ⑧ 训练集评分 ──────────────────────────────────
    after_train, _ = eval_module.evaluate("current", "train")
    after_train["run_type"] = "train_after_optimizer"
    after_train["round"] = round_index
    eval_module.append_trial(after_train, ROOT / "experiments/trials.jsonl")
    write_json(round_dir / "train-after.json", after_train)

    # ── ⑨ 验证集评分 ──────────────────────────────────
    after_eval, _ = eval_module.evaluate("current", "eval")
    after_eval["run_type"] = "eval_after_optimizer"
    after_eval["round"] = round_index
    eval_module.append_trial(after_eval, ROOT / "experiments/trials.jsonl")
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

    # ── ⑩ 存档：刷新 failure report + 快照 + 训练曲线 + copy_out ──
    failure_report_module.write_failure_report("current", "train", 5, failure_report)
    snapshot_round_artifacts(round_dir)
    update_training_curve(round_index, before_train, after_train, after_eval)
    write_json(round_dir / "summary.json", summary)

    # Per-round 持久化：把工作槽里的修改写回数据仓库
    copy_out(data_home)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Snake Arena Heuristic Learning loop.")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help=f"Experiment name; data lives at {DATA_HOME_BASE}/<exp>/",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--optimizer", choices=("claude", "none"), default="claude")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompt without invoking optimizer.")
    parser.add_argument("--max-budget-usd", type=float, default=999.0)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    data_home = data_home_for(args.exp)
    validate_data_home(data_home)
    copy_in(data_home)
    _load_snake_hl_modules()  # snake_hl.policy 现在存在，可以 import 了
    print(f"Loaded experiment '{args.exp}' from {data_home}")

    # 每次运行创建带时间戳的目录
    run_dir = args.run_dir or Path("experiments/runs") / utc_stamp()
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
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
        # 即使中途崩溃，也把当前工作槽 copy_out 一次，让用户可从最新状态恢复
        if not args.dry_run:
            try:
                copy_out(data_home)
            except Exception as e:
                print(f"WARNING: final copy_out failed: {e}", file=sys.stderr)

    write_json(run_dir / "run-summary.json", summaries)
    print(f"run_dir={run_dir.as_posix()}")
    print(f"Data home: {data_home}")


if __name__ == "__main__":
    main()
