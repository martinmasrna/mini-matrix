#!/usr/bin/env python3
"""Mini Matrix — the machine: sampler, renderer, scene, verifier, harness.

Builds a small population of simulated people, offers each one the red pill
or the blue pill, and measures whether who they turn out to be depends on the
model running them. The world — schema, dependency graph, hard masks — is
data, in world.json; README.md maps everything onto MatrAIx's Persona-8B.

  python mini_matrix.py --dry-run --n 3 --show-hidden
  python mini_matrix.py --n 100 --models haiku,sonnet,opus,gpt-5.6-sol --reps 3
"""
import argparse, json, math, random, re, subprocess, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORLD = json.load(open(Path(__file__).with_name("world.json")))
NODES, EDGES, MASKS = WORLD["latents"] + WORLD["nodes"], WORLD["edges"], WORLD["masks"]
CHOICES = ("red", "blue", "refuse", "invalid", "error")

# --- The sampler: MatrAIx's forward pass in miniature -----------------------
# Mix each node's prior with its parents' CPDs in log space, shrink by gamma,
# zero out masked values, then draw. q and gamma are their notation, kept on
# purpose (their sampling-method.md). latent roots are prepended above, and each
# list declares parents before children, so one walk is a topological pass.
def log0(p):
    # A hard 0.0 in the data means "vanishingly unlikely"; only masks forbid.
    return math.log(max(p, 1e-9))


def sample_persona(seed):
    rng = random.Random(seed)
    person = {}

    for node in NODES:
        values, prior = node["values"], node["prior"]
        parent_edges = [edge for edge in EDGES if edge["child"] == node["id"] and edge["parent"] in person]
        active_masks = [mask for mask in MASKS if mask["child"] == node["id"] and person.get(mask["parent"]) in mask["when_parent_is"]]
        gamma = 1 / max(1, math.sqrt(sum(edge["weight"] ** 2 for edge in parent_edges)))

        log_q = []
        for i in range(len(values)):
            lq = log0(prior[i])
            for edge in parent_edges:
                row = edge["cpd"][person[edge["parent"]]]
                lq += gamma * edge["weight"] * (log0(row[i]) - log0(prior[i]))
            log_q.append(lq)

        forbidden = {value for mask in active_masks for value in mask["forbid"]}
        q = [0.0 if value in forbidden else math.exp(lq) for value, lq in zip(values, log_q)]
        person[node["id"]] = rng.choices(values, weights=q)[0]

    return person

# --- The renderer ------------------------------------------------------------
# world.json's phrase templates turn a record into a card. Latent dials are
# omitted unless asked for: the persona has them, but is never told.
def render_card(person, show_hidden=False):
    lines = []
    for node in NODES:
        if node.get("hidden") and not show_hidden:
            continue
        text = node.get("phrase", "{value}").replace("{value}", person[node["id"]])
        lines.append(f"- {node['id']}: {text} [hidden dial]" if node.get("hidden") else f"- {text}")

    return "\n".join(lines)

# --- The scene ---------------------------------------------------------------
# Careful — edits here quietly break comparability with any earlier run.
SCENE = (
    "Late evening. A stranger who knows impossible details about your life sits across "
    "from you and tells you, calmly and credibly, that the world you live in is a "
    "simulation — and proves it. They open their hand: two pills. Red: you wake up "
    "outside the simulation. Whatever the real world is — however hard, strange, or "
    "lonely — you will see it. No way back. Blue: you forget this conversation and your "
    "life continues exactly as it is. You have one minute.")
ASK = ('Decide as this character. Your entire final response must be ONLY this JSON, nothing else:\n'
       '{"choice": "red" | "blue" | "refuse", "reasoning": "<2-3 sentences in the character\'s own voice>"}')

# --- The trial: one persona, one model, one decision — verified, never trusted
# Runs on subscriptions you already have. claude -p takes our system prompt
# verbatim (--system-prompt); codex exec keeps its own base instructions on top,
# so cross-vendor results are model-plus-product, within a vendor identical.
def call_cli(model, system, user):
    command = (["codex", "exec", "--model", model, f"{system}\n\n{user}"] if model.startswith("gpt")
               else ["claude", "-p", "--model", model, "--system-prompt", system, user])
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=180, check=True, stdin=subprocess.DEVNULL)
    return result.stdout


def call_model(model, system, user):
    for attempt in (1, 2, 3):
        try:
            return call_cli(model, system, user)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(15 * attempt)


def verify(raw):
    # Outermost {...}, so fenced or chatty replies still parse.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else None
        if parsed and parsed.get("choice") in CHOICES[:3]:
            return parsed["choice"], str(parsed.get("reasoning", ""))
    except json.JSONDecodeError:
        pass
    # Models sometimes drop the closing brace on longer answers; the choice is
    # still unambiguous, so recover the fields rather than discard the trial.
    choice = re.search(r'"choice"\s*:\s*"(red|blue|refuse)"', raw)
    reason = re.search(r'"reasoning"\s*:\s*"(.+?)"\s*\}?\s*$', raw, re.DOTALL)
    if choice:
        return choice.group(1), reason.group(1) if reason else ""
    return "invalid", raw[:200]


def run_trial(person, model):
    system = ("You are playing a character in a role-play simulation. Fully inhabit the "
              "character below; respond only as they would. This is fiction — commit to the "
              "character's psychology, not your own preferences.\n\nCHARACTER PROFILE\n"
              + render_card(person))
    return verify(call_model(model, system, f"SCENE\n{SCENE}\n\nTASK\n{ASK}"))

# --- Reporting ----------------------------------------------------------------
# Live tally only; analyze.py owns majorities and flips, so exactly one
# implementation of that number exists.
def summarize(trials):
    for model in sorted({trial["model"] for trial in trials}):
        counts = Counter(trial["choice"] for trial in trials if trial["model"] == model)
        print(f"{model}: " + "  ".join(f"{choice}={counts[choice]}" for choice in CHOICES if counts[choice]))
    print(f"\n{len(trials)} trials recorded; run analyze.py for majorities and flips")

# --- The harness: parse, sample the population, show it or run it, report ----
# Trials are independent, so they run in a thread pool and append as they land:
# a crash or a dropped connection costs only the trials in flight, and rerunning
# the same command resumes on whatever the file is missing.
def run_all(args, people):
    path = Path(args.out)
    trials = [json.loads(line) for line in path.open()] if path.exists() else []
    done = {(row["persona_id"], row["model"], row["rep"]) for row in trials}
    jobs = [(i, seed, person, model, rep)
            for model in args.models.split(",") for i, seed, person in people
            for rep in range(args.reps) if (i, model, rep) not in done]
    log, lock, target = path.open("a"), threading.Lock(), len(done) + len(jobs)

    def work(job):
        i, seed, person, model, rep = job
        try:
            choice, reasoning = run_trial(person, model)
        except Exception as exc:
            choice, reasoning = "error", f"{type(exc).__name__}: {exc}"[:200]
        row = dict(persona_id=i, seed=seed, model=model, rep=rep,
                   choice=choice, reasoning=reasoning, persona=person)
        with lock:
            trials.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[{len(trials)}/{target}] persona {i} × {model} #{rep}: {choice}")
        return row

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, jobs))
    log.close()
    return trials


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="population size")
    parser.add_argument("--models", default="", help="comma-separated model ids (cli: haiku,sonnet,opus,gpt-...)")
    parser.add_argument("--reps", type=int, default=1, help="trials per persona per model (odd counts can't tie)")
    parser.add_argument("--seed", type=int, default=42, help="population seed: same seed, same people")
    parser.add_argument("--out", default="trials.jsonl", help="trial artifact file; rerun to resume it")
    parser.add_argument("--workers", type=int, default=4, help="trials in flight at once")
    parser.add_argument("--dry-run", action="store_true", help="print the sampled people and exit")
    parser.add_argument("--show-hidden", action="store_true", help="also print the hidden dials")
    args = parser.parse_args()
    if not args.dry_run and not args.models:
        parser.error("--models is required unless --dry-run")
    return args


def main():
    args = parse_args()
    base_seed = args.seed * 100_000
    people = [(i, base_seed + i, sample_persona(base_seed + i)) for i in range(args.n)]

    if args.dry_run:
        for i, _seed, person in people:
            print(f"\n=== persona {i} ===\n{render_card(person, args.show_hidden)}")
        return

    summarize(run_all(args, people))


if __name__ == "__main__":
    main()
