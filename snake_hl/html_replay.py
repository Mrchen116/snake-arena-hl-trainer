from __future__ import annotations

import argparse
import json
from pathlib import Path

from snake_hl.env import EpisodeResult, SnakeState, simulate_episode
from snake_hl.replay import POLICIES


VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Snake Replay Viewer</title>
  <style>
    :root {
      --bg: #101216;
      --panel: #1b2028;
      --grid: #2d3440;
      --text: #eef2f8;
      --muted: #9aa4b2;
      --snake: #54d17a;
      --head: #f7d154;
      --food: #ff5d6c;
      --danger: #ff8a3d;
      --button: #edf2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    main {
      width: min(980px, 100%);
      display: grid;
      grid-template-columns: minmax(360px, 520px) 1fr;
      gap: 20px;
      align-items: start;
    }
    h1 { margin: 0 0 8px; font-size: 22px; }
    .meta, .controls, .legend, .timeline {
      background: var(--panel);
      border: 1px solid #303746;
      border-radius: 8px;
      padding: 14px;
    }
    .board {
      display: grid;
      width: min(520px, 100%);
      aspect-ratio: 1;
      border: 1px solid #3a4252;
      background: #171b24;
      border-radius: 8px;
      overflow: hidden;
    }
    .cell {
      border: 1px solid var(--grid);
      min-width: 0;
      min-height: 0;
    }
    .snake { background: var(--snake); }
    .head { background: var(--head); }
    .food { background: var(--food); }
    .death { outline: 3px solid var(--danger); outline-offset: -3px; }
    .side { display: grid; gap: 12px; }
    .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }
    .label { color: var(--muted); font-size: 12px; }
    .value { font-weight: 700; font-size: 17px; }
    .controls-row { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--button);
      color: #111318;
      padding: 9px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    select, input[type="range"] { width: 100%; }
    select {
      border: 1px solid #3a4252;
      border-radius: 6px;
      background: #141922;
      color: var(--text);
      padding: 9px 10px;
      margin-bottom: 10px;
    }
    input[type="file"] { max-width: 100%; color: var(--muted); }
    .legend-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; color: var(--muted); }
    .swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
    .error { color: #ffb4a1; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      color: var(--muted);
      max-height: 220px;
      overflow: auto;
    }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      .board { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <section>
      <h1 id="title">Snake replay</h1>
      <div id="board" class="board" aria-label="Snake replay board"></div>
    </section>
    <section class="side">
      <div class="meta">
        <div class="stats">
          <div><div class="label">Step</div><div class="value" id="step"></div></div>
          <div><div class="label">Action</div><div class="value" id="action"></div></div>
          <div><div class="label">Food</div><div class="value" id="food"></div></div>
          <div><div class="label">Score</div><div class="value" id="score"></div></div>
          <div><div class="label">Death</div><div class="value" id="death"></div></div>
          <div><div class="label">Seed</div><div class="value" id="seed"></div></div>
        </div>
      </div>
      <div class="controls">
        <select id="episodeSelect" aria-label="Choose replay"></select>
        <div class="controls-row">
          <button id="play">Play</button>
          <input id="file" type="file" accept="application/json,.json">
        </div>
        <input id="slider" type="range" min="0" value="0">
        <p class="label" id="source"></p>
      </div>
      <div class="legend">
        <div class="legend-row"><span class="swatch" style="background: var(--head)"></span> head</div>
        <div class="legend-row"><span class="swatch" style="background: var(--snake)"></span> body</div>
        <div class="legend-row"><span class="swatch" style="background: var(--food)"></span> food</div>
        <div class="legend-row"><span class="swatch" style="background: transparent; outline: 3px solid var(--danger);"></span> final death cell</div>
      </div>
      <div class="timeline">
        <div class="label">Last 30 actions</div>
        <pre id="tail"></pre>
      </div>
    </section>
  </main>
  <script>
    const board = document.getElementById("board");
    const slider = document.getElementById("slider");
    const play = document.getElementById("play");
    const fileInput = document.getElementById("file");
    const episodeSelect = document.getElementById("episodeSelect");
    const fields = {
      title: document.getElementById("title"),
      step: document.getElementById("step"),
      action: document.getElementById("action"),
      food: document.getElementById("food"),
      score: document.getElementById("score"),
      death: document.getElementById("death"),
      seed: document.getElementById("seed"),
      source: document.getElementById("source"),
      tail: document.getElementById("tail"),
    };
    let episode = null;
    let replayIndex = [];
    let timer = null;
    let index = 0;

    function actionAt(i) {
      return i === 0 ? "START" : episode.actions[i - 1];
    }

    function stop() {
      clearInterval(timer);
      timer = null;
      play.textContent = "Play";
    }

    function setError(message) {
      stop();
      fields.title.textContent = "Snake replay";
      fields.source.innerHTML = `<span class="error">${message}</span>`;
      board.innerHTML = "";
    }

    function setEpisode(nextEpisode, sourceLabel) {
      stop();
      episode = nextEpisode;
      index = 0;
      slider.max = episode.states.length - 1;
      board.style.gridTemplateColumns = `repeat(${episode.states[0].width}, 1fr)`;
      board.style.gridTemplateRows = `repeat(${episode.states[0].height}, 1fr)`;
      fields.title.textContent = `${episode.policy} seed ${episode.seed}`;
      fields.source.textContent = sourceLabel;
      render(0);
    }

    async function loadEpisode(path) {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      setEpisode(await response.json(), path);
    }

    function optionLabel(item) {
      const death = item.death_reason || "survived";
      return `${item.policy} / ${item.split || "single"} / seed ${String(item.seed).padStart(4, "0")} / score ${Number(item.score).toFixed(1)} / ${death}`;
    }

    function renderReplayOptions(selectedPath) {
      episodeSelect.innerHTML = "";
      if (!replayIndex.length) {
        const option = document.createElement("option");
        option.textContent = "No indexed replays";
        option.value = "";
        episodeSelect.appendChild(option);
        episodeSelect.disabled = true;
        return;
      }
      episodeSelect.disabled = false;
      for (const item of replayIndex) {
        const option = document.createElement("option");
        option.value = item.path;
        option.textContent = optionLabel(item);
        option.selected = item.path === selectedPath;
        episodeSelect.appendChild(option);
      }
    }

    function render(i) {
      if (!episode) return;
      index = Math.max(0, Math.min(i, episode.states.length - 1));
      slider.value = index;
      const state = episode.states[index];
      const snake = new Map(state.snake.map((point, idx) => [point.join(","), idx]));
      const foodKey = state.food.join(",");
      const finalState = episode.states[episode.states.length - 1];
      const deathKey = finalState.death_reason ? finalState.snake[0].join(",") : null;

      board.innerHTML = "";
      for (let y = 0; y < state.height; y += 1) {
        for (let x = 0; x < state.width; x += 1) {
          const cell = document.createElement("div");
          const key = `${x},${y}`;
          cell.className = "cell";
          if (key === foodKey) cell.classList.add("food");
          if (snake.has(key)) cell.classList.add(snake.get(key) === 0 ? "head" : "snake");
          if (index === episode.states.length - 1 && key === deathKey) cell.classList.add("death");
          board.appendChild(cell);
        }
      }

      fields.step.textContent = `${state.steps} / ${episode.steps}`;
      fields.action.textContent = actionAt(index);
      fields.food.textContent = state.food_eaten;
      fields.score.textContent = episode.score.toFixed(1);
      fields.death.textContent = episode.death_reason || "survived";
      fields.seed.textContent = episode.seed;
      const start = Math.max(0, index - 30);
      fields.tail.textContent = episode.actions
        .slice(start, index)
        .map((action, offset) => `${start + offset + 1}: ${action}`)
        .join("\\n") || "No actions yet.";
    }

    play.addEventListener("click", () => {
      if (!episode) return;
      if (timer) {
        stop();
        return;
      }
      play.textContent = "Pause";
      timer = setInterval(() => {
        if (index >= episode.states.length - 1) {
          stop();
          return;
        }
        render(index + 1);
      }, 90);
    });
    slider.addEventListener("input", () => render(Number(slider.value)));
    fileInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      try {
        setEpisode(JSON.parse(await file.text()), file.name);
      } catch (error) {
        setError(`Could not load ${file.name}: ${error.message}`);
      }
    });
    episodeSelect.addEventListener("change", async () => {
      if (!episodeSelect.value) return;
      try {
        await loadEpisode(episodeSelect.value);
      } catch (error) {
        setError(`Could not load ${episodeSelect.value}: ${error.message}`);
      }
    });

    async function loadReplayIndex() {
      const params = new URLSearchParams(window.location.search);
      const indexPath = params.get("index") || "../replays/index.json";
      try {
        const response = await fetch(indexPath);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        replayIndex = await response.json();
        renderReplayOptions(params.get("episode"));
      } catch (error) {
        replayIndex = [];
        renderReplayOptions("");
        fields.source.textContent = `Replay index unavailable: ${error.message}. Use the JSON file picker.`;
      }
    }

    async function loadInitialEpisode() {
      const params = new URLSearchParams(window.location.search);
      const episodePath = params.get("episode");
      const firstIndexedPath = replayIndex.length ? replayIndex[0].path : null;
      const path = episodePath || firstIndexedPath;
      if (!path) return;
      try {
        await loadEpisode(path);
        renderReplayOptions(path);
      } catch (error) {
        setError(`Could not load ${path}: ${error.message}. If opened from file://, run a local web server or use the file picker.`);
      }
    }

    async function init() {
      await loadReplayIndex();
      await loadInitialEpisode();
    }

    init();
  </script>
</body>
</html>
"""


def state_to_dict(state: SnakeState) -> dict:
    return {
        "width": state.width,
        "height": state.height,
        "snake": [list(point) for point in state.snake],
        "food": list(state.food),
        "direction": state.direction,
        "steps": state.steps,
        "food_eaten": state.food_eaten,
        "done": state.done,
        "death_reason": state.death_reason,
    }


def result_payload(policy_name: str, result: EpisodeResult) -> dict:
    return {
        "policy": policy_name,
        "seed": result.seed,
        "score": result.score,
        "food_eaten": result.food_eaten,
        "steps": result.steps,
        "survived": result.survived,
        "death_reason": result.death_reason,
        "actions": list(result.actions),
        "states": [state_to_dict(state) for state in result.states],
    }


def replay_payload(policy_name: str, seed: int) -> dict:
    result = simulate_episode(POLICIES[policy_name], seed=seed, keep_states=True)
    return result_payload(policy_name, result)


def write_episode_json(policy_name: str, seed: int, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = replay_payload(policy_name, seed)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output


def replay_index_entry(replay_root: Path, replay_path: Path) -> dict:
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    relative = replay_path.relative_to(replay_root).as_posix()
    parts = replay_path.relative_to(replay_root).parts
    split = parts[1] if len(parts) > 2 else None
    return {
        "path": f"../replays/{relative}",
        "policy": payload["policy"],
        "split": split,
        "seed": payload["seed"],
        "score": payload["score"],
        "food_eaten": payload["food_eaten"],
        "steps": payload["steps"],
        "survived": payload["survived"],
        "death_reason": payload["death_reason"],
    }


def write_replay_index(replay_root: Path) -> Path:
    replay_root.mkdir(parents=True, exist_ok=True)
    entries = [
        replay_index_entry(replay_root, path)
        for path in sorted(replay_root.rglob("*.json"))
        if path.name != "index.json"
    ]
    entries.sort(key=lambda item: (item["policy"], item["split"] or "", item["seed"]))
    index_path = replay_root / "index.json"
    index_path.write_text(json.dumps(entries, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return index_path


def write_viewer(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(VIEWER_HTML, encoding="utf-8")
    return output


def write_html(policy_name: str, seed: int, output: Path) -> Path:
    """Compatibility wrapper: writes JSON beside the shared viewer path."""
    return write_episode_json(policy_name, seed, output.with_suffix(".json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Snake replay JSON and the reusable viewer.")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="current")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--viewer", type=Path, default=Path("experiments/replay_viewer/index.html"))
    args = parser.parse_args()

    replay_output = args.output or Path(f"experiments/replays/{args.policy}/seed-{args.seed:04d}.json")
    viewer_path = write_viewer(args.viewer)
    replay_path = write_episode_json(args.policy, args.seed, replay_output)
    if "replays" in replay_path.parts:
        replay_root = replay_path.parents[len(replay_path.parts) - replay_path.parts.index("replays") - 2]
        write_replay_index(replay_root)
    print(replay_path)
    print(viewer_path)


if __name__ == "__main__":
    main()
