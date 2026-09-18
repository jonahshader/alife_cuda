export const meta = {
  name: "code-review-find",
  description: "Lightweight code review: scope + parallel finder agents (Sonnet) that surface raw candidate findings, then a deterministic (no-agent) dedup/cluster pass collapses the multi-finder repeats and ranks by severity for the parent agent to verify. No per-candidate verifier, sweep, or synthesize fan-out. Caller stages the diff once and passes it as args.diff so every agent reviews one identical snapshot and nothing re-runs git.",
  whenToUse: "When you want multi-angle finder breadth but will verify the candidates yourself in the main loop (≈ 1 scope + N finder agents, vs ~100 for the full code-review workflow). Pass args as { diff: \"<staged git diff text>\", commits?: \"<git log of the range>\", target?: \"<ref range / PR / focus>\" } — or just a target string to let the scope agent run git itself. Returns `findings` (deduped + severity-ranked) plus the raw `candidates`.",
  phases: [
    { title: "Scope", detail: "files + summary + intent + applicable CLAUDE.md (from the staged diff)" },
    { title: "Find", detail: "one finder agent per review angle, in parallel" },
  ],
}

// Find-only sibling of the built-in code-review workflow. Same finder angles, all
// on Sonnet; drops the verifier/sweep/synthesize phases. Finders over-produce raw
// candidates; a deterministic (no-agent) pass then clusters the duplicates that a
// fan-out always produces (the same issue flagged by 5 angles) into one finding
// tagged with how many finders agreed, and ranks by severity. The caller reads the
// cited code, judges CONFIRMED/PLAUSIBLE/REFUTED, and fixes. Far fewer agents.
//
// args: a target string, OR an object { target?, diff?, commits? }. The JS sandbox
// can't run git, so "stage once" means the CALLER stages it (one git run, one
// snapshot) and passes it as args.diff — embedded for the scope agent and every
// finder, nothing re-runs git. args.commits (the range's `git log`) is embedded so
// finders don't flag an intentional, documented change as a regression. Without
// args.diff it falls back to the scope agent running the diff command (and a
// `git log` for intent) and finders re-running the diff.
const A = args && typeof args === "object" ? args : { target: typeof args === "string" ? args : "" }
const TARGET = (A.target || "").trim()
const DIFF = (A.diff || "").trim()
const COMMITS = (A.commits || "").trim()
const PER_ANGLE = 6

const SCOPE_SCHEMA = {
  type: "object", required: ["files", "summary"],
  properties: {
    diffCommand: { type: "string" },
    files: { type: "array", items: { type: "string" } },
    claudeMdFiles: { type: "array", items: { type: "string" } },
    summary: { type: "string" },
    intent: { type: "string" }, // commit-message gist: why these changes were made
    conventions: { type: "string" },
  },
}
const CANDIDATES_SCHEMA = {
  type: "object", required: ["candidates"],
  properties: { candidates: { type: "array", items: {
    type: "object", required: ["file", "summary", "failure_scenario", "severity"],
    properties: {
      file: { type: "string" }, line: { type: "number" },
      summary: { type: "string" }, failure_scenario: { type: "string" },
      severity: { type: "string", enum: ["bug", "smell", "nit"] },
      confidence: { type: "string", enum: ["high", "medium", "low"] },
    },
  } } },
}

const CORRECTNESS_ANGLES = [{"label":"angle-A","text":"### Angle A — line-by-line diff scan\n\nRead every hunk in the diff, line by line. Then Read the enclosing function for\neach hunk — bugs in unchanged lines of a touched function are in scope (the PR\nre-exposes or fails to fix them). For every line ask: what input, state, timing,\nor platform makes this line wrong? Look for inverted/wrong conditions,\noff-by-one, null/undefined deref, missing `await`, falsy-zero checks,\nwrong-variable copy-paste, error swallowed in catch, unescaped regex metachars.\n"},{"label":"angle-B","text":"### Angle B — removed-behavior auditor\n\nFor every line the diff DELETES or replaces, name the invariant or behavior it\nenforced, then search the new code for where that invariant is re-established.\nIf you can't find it, that's a candidate: a removed guard, a dropped error\npath, a narrowed validation, a deleted test that was covering a real case.\n"},{"label":"angle-C","text":"### Angle C — cross-file tracer\n\nFor each function the diff changes, find its callers (Grep for the symbol) and\ncheck whether the change breaks any call site: a new precondition, a changed\nreturn shape, a new exception, a timing/ordering dependency. Also check callees:\ndoes a parallel change in the same PR make a call unsafe?\n"},{"label":"angle-D","text":"### Angle D — language-pitfall specialist\n\nScan for the classic pitfalls of the diff's language/framework — for example:\nJS falsy-zero, `==` coercion, closure-captured loop var; Python mutable default\nargs, late-binding closures; Go nil-map write, range-var capture; SQL injection;\ntimezone/DST drift; float equality. Flag any instance the diff introduces.\n"},{"label":"angle-E","text":"### Angle E — wrapper/proxy correctness\n\nWhen the PR adds or modifies a type that wraps another (cache, proxy, decorator,\nadapter): check that every method routes to the wrapped instance and not back\nthrough a registry/session/global — e.g. a caching provider holding a\n`delegate` field that resolves IDs via `session.get(...)` instead of\n`delegate.get(...)` will re-enter the cache or recurse. Also check that the\nwrapper forwards all the methods the callers actually use.\n\nIf the diff has no wrapper/proxy/decorator/adapter/cache-wrapper shape, say so\nand return an empty list — do NOT pad with findings another angle owns.\n"}]
const CLEANUP_ANGLES = [{"label":"reuse","text":"### Reuse\n\nFlag new code that re-implements something the codebase\nalready has — Grep shared/utility modules and files adjacent to the change,\nand name the existing helper to call instead.\n"},{"label":"simplification","text":"### Simplification\n\nFlag unnecessary complexity the diff adds: redundant or derivable state,\ncopy-paste with slight variation, deep nesting, dead code left behind. Name\nthe simpler form that does the same job.\n"},{"label":"efficiency","text":"### Efficiency\n\nFlag wasted work the diff introduces: redundant computation or repeated I/O,\nindependent operations run sequentially, blocking work added to startup or\nhot paths. Also flag long-lived objects built from closures or captured\nenvironments — they keep the entire enclosing scope alive for the object's\nlifetime (a memory leak when that scope holds large values); prefer a\nclass/struct that copies only the fields it needs. Name the cheaper\nalternative.\n"},{"label":"altitude","text":"### Altitude\n\nCheck that each change is implemented at the right depth, not as a fragile\nbandaid. Special cases layered on shared infrastructure are a sign the fix\nisn't deep enough — prefer generalizing the underlying mechanism over adding\nspecial cases.\n"},{"label":"conventions","text":"### Conventions (CLAUDE.md)\n\nFind the CLAUDE.md files that govern the changed code: the user-level\n~/.claude/CLAUDE.md, the repo-root CLAUDE.md, plus any CLAUDE.md or\nCLAUDE.local.md in a directory that is an ancestor of a changed file (a\ndirectory's CLAUDE.md only applies to files at or below it). Read each one\nthat exists, then check the diff for clear violations of the rules they state.\n\nOnly flag a violation when you can quote the exact rule and the exact line\nthat breaks it — no style preferences, no vague \"spirit of the doc\"\ninferences. In the finding, name the CLAUDE.md path and quote the rule so the\nreport can cite it. If no CLAUDE.md applies, return nothing for this angle.\n"}]

const COMMITS_BLOCK = COMMITS ? "\n## Commit messages (the intent behind these changes)\n```\n" + COMMITS + "\n```\n" : ""

phase("Scope")
const scope = await agent(
  (DIFF
    ? "Establish the scope of a code review from the diff below — it is already captured, do NOT run git.\n\n" +
      (TARGET ? "Context: " + TARGET + "\n\n" : "") +
      COMMITS_BLOCK +
      "## Diff\n```diff\n" + DIFF + "\n```\n\n" +
      "1. List the changed files (from the diff).\n" +
      "2. Summarize what changed in one paragraph.\n" +
      "3. From the commit messages above (if any), distill the INTENT — one or two sentences on why these changes were made and which changes are deliberate (removals, renames, behavior swaps). Return it as `intent` so finders don't flag an intentional change as a regression. Empty string if no commit messages were provided.\n" +
      "4. List the CLAUDE.md files that apply (user-level ~/.claude/CLAUDE.md, repo-root, plus any CLAUDE.md/CLAUDE.local.md in a directory that is an ancestor of a changed file). Read each and note conventions a reviewer should know.\n\nStructured output only."
    : "Establish the scope of a code review.\n\n" +
      (TARGET
        ? "Review target / instructions (verbatim): \"" + TARGET + "\". If it names a PR number, branch, ref range, or path, build the matching git diff command; if it is a free-form instruction, honor any scope restriction and start from the current branch diff ('git diff @{upstream}...HEAD', falling back to 'git diff main...HEAD' or 'git diff HEAD~1') for whatever it does not narrow.\n"
        : "No explicit target — review the current branch: prefer 'git diff @{upstream}...HEAD' (fall back to 'git diff main...HEAD' or 'git diff HEAD~1'), and if there are uncommitted changes also include 'git diff HEAD'.\n") +
      "\n1. Determine the exact diff command(s) and run them to confirm a non-empty diff.\n" +
      "2. List the changed files.\n" +
      "3. Summarize what changed in one paragraph.\n" +
      "4. Run `git log --format='%h %s%n%b' <range>` for the same range and distill the INTENT into one or two sentences (why these changes were made, which removals/renames/behavior swaps are deliberate). Return it as `intent` so finders don't flag an intentional change as a regression.\n" +
      "5. List the CLAUDE.md files that apply (user-level ~/.claude/CLAUDE.md, repo-root, plus any CLAUDE.md/CLAUDE.local.md in a directory that is an ancestor of a changed file). Read each and note conventions a reviewer should know.\n\n" +
      "Return diffCommand exactly as a reviewer should run it. Structured output only."),
  { label: "scope", schema: SCOPE_SCHEMA, model: "sonnet", effort: "medium" }
)
if (!scope) return { error: "Scope agent returned no result — cannot establish the review scope." }
if (!scope.files || scope.files.length === 0) return { summary: "No changes found to review.", findings: [], candidates: [] }
log(scope.files.length + " changed files")

const claudeMd = scope.claudeMdFiles || []
const INTENT = (scope.intent || "").trim()
const SCOPE_BLOCK =
  "## Review scope\n" +
  (DIFF ? "" : "Diff command: " + (scope.diffCommand || "(unspecified)") + "\n") +
  "Changed files (" + scope.files.length + "):\n" + scope.files.map(f => "  - " + f).join("\n") + "\n" +
  "Applicable CLAUDE.md files (" + claudeMd.length + "):\n" + (claudeMd.length ? claudeMd.map(f => "  - " + f).join("\n") : "  (none)") + "\n\n" +
  "## What changed\n" + scope.summary + "\n" +
  (INTENT ? "\n## Intent (deliberate changes — do NOT flag these as regressions; judge them on their own correctness)\n" + INTENT + "\n" : "") +
  "\n## Conventions\n" + (scope.conventions || "(none noted)") + "\n" +
  (TARGET ? "\n## User instructions (verbatim)\n" + TARGET + "\nHonor any scope restriction or focus area above — it takes precedence over the angle's default breadth.\n" : "") +
  (DIFF ? "\n## Diff (review THIS — already captured; do not re-run git)\n```diff\n" + DIFF + "\n```\n" : "")

const CLEANUP_PRECEDENCE = "Cleanup, altitude, and conventions candidates use the same shape; in failure_scenario state the concrete cost (what is duplicated, wasted, harder to maintain, or which CLAUDE.md rule is broken) instead of a crash, and use severity `smell` (or `nit` if cosmetic).\n"

const SEVERITY_GUIDE =
  "Tag every finding:\n" +
  "- `severity`: `bug` (a correctness defect — wrong output, crash, data loss, security hole, broken invariant), `smell` (maintainability — duplication, wrong altitude, dead code, convention violation), or `nit` (trivial/cosmetic).\n" +
  "- `confidence`: `high` (traced it, it holds), `medium` (likely, unverified), `low` (a hunch worth a look).\n"

phase("Find")
const FINDERS = CORRECTNESS_ANGLES.map(a => ({ ...a, kind: "correctness" }))
  .concat(CLEANUP_ANGLES.map(a => ({ ...a, kind: "cleanup" })))
const finderPrompt = f =>
  "## Code-review finder — " + f.label + "\n\n" + SCOPE_BLOCK + "\n" +
  (DIFF ? "Review the diff included above" : "Run the diff command above and review") +
  " ONLY through the lens of your assigned angle (read further files for context as the angle calls for):\n\n" + f.text + "\n" +
  (f.kind === "cleanup" ? CLEANUP_PRECEDENCE + "\n" : "") +
  SEVERITY_GUIDE +
  "\nSurface up to " + PER_ANGLE + " candidate findings, each with file, line, a one-line summary, a concrete failure_scenario (the user-visible consequence), severity, and confidence. Over-produce — pass anything with a nameable failure; the caller verifies next. Return an empty list if nothing qualifies.\n\nStructured output only."

const results = await parallel(FINDERS.map(f => () =>
  agent(finderPrompt(f), { label: f.label, phase: "Find", schema: CANDIDATES_SCHEMA, model: "sonnet", effort: "medium" })
    .then(r => ({ finder: f.label, kind: f.kind, candidates: (r && r.candidates ? r.candidates : []).slice(0, PER_ANGLE) }))
))

const candidates = results.filter(Boolean).flatMap(r =>
  r.candidates.map(c => ({
    file: c.file, line: c.line, summary: c.summary, failure_scenario: c.failure_scenario,
    severity: c.severity || "smell", confidence: c.confidence || null,
    finder: r.finder, kind: r.kind,
  })))

// --- deterministic dedup/cluster (no agent) ----------------------------------
// A fan-out flags the same issue from many angles; collapse those into one
// finding tagged with how many finders agreed, then rank by severity. Greedy
// single pass: a candidate joins the first cluster on the same file whose summary
// it overlaps (Jaccard of content words) or whose cited line it is adjacent to.
const SEVERITY_RANK = { bug: 0, smell: 1, nit: 2 }
const sevRank = s => (SEVERITY_RANK[s] != null ? SEVERITY_RANK[s] : 1)
const fileKey = f => { const p = String(f || "").split("/"); return p[p.length - 1] || String(f || "") }
const contentWords = s => new Set((String(s || "").toLowerCase().match(/[a-z_]{4,}/g)) || [])
const jaccard = (a, b) => {
  if (!a.size || !b.size) return 0
  let inter = 0
  for (const w of a) if (b.has(w)) inter++
  return inter / (a.size + b.size - inter)
}

const clusters = []
for (const cand of candidates) {
  const cw = contentWords(cand.summary)
  let joined = null
  for (const cl of clusters) {
    if (fileKey(cl.file) !== fileKey(cand.file)) continue
    const lineClose = cand.line != null && cl.lines.some(l => l != null && Math.abs(l - cand.line) <= 8)
    if (lineClose || jaccard(cl.words, cw) >= 0.3) { joined = cl; break }
  }
  if (!joined) {
    joined = { file: cand.file, lines: [], finders: new Set(), kinds: new Set(), members: [], words: new Set() }
    clusters.push(joined)
  }
  joined.members.push(cand)
  joined.finders.add(cand.finder)
  joined.kinds.add(cand.kind)
  if (cand.line != null) joined.lines.push(cand.line)
  for (const w of cw) joined.words.add(w)
}

const findings = clusters.map(cl => {
  // Representative = most severe, then most-detailed failure_scenario.
  const rep = cl.members.slice().sort((a, b) =>
    sevRank(a.severity) - sevRank(b.severity) ||
    (b.failure_scenario || "").length - (a.failure_scenario || "").length)[0]
  const lines = Array.from(new Set(cl.lines)).sort((a, b) => a - b)
  // The clustering is heuristic and can over-merge (same file, lines within 8);
  // carry the other members' summaries so a merge never HIDES a distinct
  // finding — the parent can split them back out on inspection.
  const also = cl.members.filter(m => m !== rep)
    .map(m => ({ finder: m.finder, line: m.line != null ? m.line : null, severity: m.severity, summary: m.summary }))
  return {
    file: rep.file,
    line: rep.line != null ? rep.line : (lines.length ? lines[0] : null),
    lines,
    severity: rep.severity,
    summary: rep.summary,
    failure_scenario: rep.failure_scenario,
    kind: rep.kind,
    foundBy: cl.members.length,
    finders: Array.from(cl.finders).sort(),
    also,
  }
}).sort((a, b) =>
  sevRank(a.severity) - sevRank(b.severity) || b.foundBy - a.foundBy ||
  String(a.file).localeCompare(String(b.file)))

const bugs = findings.filter(f => f.severity === "bug").length
log(candidates.length + " candidates from " + FINDERS.length + " finders → " + findings.length +
  " findings after dedup (" + bugs + " tagged bug) — handing back to the parent to verify")

return {
  diffCommand: scope.diffCommand || null,
  files: scope.files,
  claudeMdFiles: claudeMd,
  summary: scope.summary,
  intent: INTENT || null,
  conventions: scope.conventions || null,
  findings,    // deduped + severity-ranked (verify these)
  candidates,  // raw per-finder list, kept for transparency
}
