# GraphGuard UI redesign brief

The current dashboard reads as a generic AI-generated dark SaaS template. Redesign it
as a **research instrument**: calm, precise, evidence-first. Every screen should answer
"what did we find, and how sure are we?" before anything else.

Keep the stack (Next.js App Router, TypeScript, Tailwind, Recharts, Cytoscape.js).
All AGENTS.md rules still apply: no numbers in TSX, everything from the API.

## 1. Bugs to fix first

- `app/page.tsx` hardcoded fallbacks (`"203,769"`, `"234,355"`, `"~23% Labelled"`,
  `"49 Steps"`, `"~2017-2018"`). Remove them. Show a skeleton while loading and an
  error state on failure. Delete the date range unless a source is cited.
- Score colour uses `score >= 0.5`. Use **the model's own threshold** from the API
  everywhere a score is coloured or labelled.
- Hardcoded transaction IDs (`69746416`, `30585812`, `106364923`) in nav, explorer,
  explainability, transaction pages. Add `GET /api/examples` (backend) returning
  example txIds chosen by code (top-scored true positive, a false positive, a false
  negative, an unknown) from saved predictions. Nav "Investigate" goes to a search page,
  not a fixed ID.
- Footer claims "All numbers ... with paired block bootstrap validation". Not true for
  dataset counts; remove.

## 2. Visual direction

- **Theme:** light "paper" default (`#FAFAF7` background, `#111827` ink, `#E5E7EB` rules),
  optional dark mode via `class` strategy. No gradients, no glassmorphism, no glow.
- **Type:** headings in a serif (Source Serif 4 or Fraunces via `next/font`), UI in
  Inter or IBM Plex Sans, IDs and numbers in IBM Plex Mono with `tabular-nums`.
  Clear scale: 32 / 24 / 18 / 14 / 12.
- **Colour is semantic only:** illicit `#B42318`, licit `#1D4ED8`, unknown `#6B7280`
  drawn with a **hatched or dashed** treatment, one accent (`#0F766E`) for interactive
  elements. Nothing else coloured.
- **Known vs predicted** must never be confused: known label = filled shape;
  model prediction = outline ring in the predicted colour + score.
- Remove emoji, the icon-per-heading pattern, and the uniform card grids. Use rules,
  whitespace and alignment instead of boxes. Max content width ~1200px; generous spacing.
- Banner: one quiet line under the header ("Research demo on historical Elliptic data.
  Not a live monitoring or compliance system."), no emoji, no amber box.

## 3. Shared components (`frontend/components/`)

`PageHeader`, `Section`, `Stat` (label, value, footnote, source), `ClassBadge`
(known/predicted variants), `ScoreBar` (0–1 bar with a tick at the model threshold),
`CIPlot` (dot + whisker, optional base-rate reference line), `DataTable` (sortable,
tabular nums), `Skeleton`, `EmptyState`, `ErrorState`, `Disclaimer`. Pages compose
these; no page should define its own card styles.

## 4. Pages

**Overview — lead with the answer.**
1. Research question in one sentence.
2. Headline figure: PR-AUC on `paper_refit` for every model family as a dot-and-whisker
   plot (mean ± CI), with the test base rate as a dashed reference line. One-line
   takeaway generated from the comparison table (e.g. which differences are significant).
3. Three small stats (transactions, labelled share, time steps), all from the API.
4. Three entry points: Investigate a transaction · Explore the graph · See all results.
The model registry and inference log move to a secondary "System" section at the bottom.

**Investigate (`/investigate` search + `/transaction/[id]`) — a case file.**
Search box with example chips from `/api/examples`. The case page has three columns
on desktop (stacked on mobile):
- *Verdict:* ScoreBar with threshold tick, predicted class, known label (or "unknown"),
  time step, model selector.
- *Neighbourhood:* the k-hop graph (k = 1/2 toggle) centred on the transaction.
- *Why:* top feature attributions as horizontal bars (feature names are anonymised;
  say so), neighbour contributions (occlusion Δscore), pattern flags, explainer
  fidelity shown next to the explanation, disclaimer text.

**Graph Explorer.** Replace `cose` with a layout centred on the focus node
(`concentric` by hop distance, or `breadthfirst` following edge direction). Edge
arrows show payment direction. Node size by degree (capped). Click a node to open a
side panel with its case summary and a link to the case page. Legend always visible.

**Benchmark.** Model × feature-set dot-whisker plots, a sortable table, and the
comparison table showing block CI + "better on k/n steps". PR and ROC curves in a
secondary tab. Every chart titled with what it shows and which split.

**Temporal.** Small multiples per model: PR-AUC by test step, with a bar strip of
illicit counts per step underneath, and steps with < 30 illicit marked as low-confidence.
Shade the late-period regime change. Forward-chaining cut results as a second view.

**Explainability.** Global SHAP importance (bars), then a gallery of example cases
from `/api/examples` (TP, FP, FN), each linking to its case page. Explanations always
show fidelity and the "attribution, not causation" note.

## 5. Quality bar

- Loading skeletons, empty and error states on every page; URL holds state
  (selected model, k, tab) so views are shareable.
- Keyboard navigable; visible focus rings; WCAG AA contrast; charts have text
  alternatives (a caption with the key numbers from the API).
- Responsive down to 375px.
- `npm run build` and `npm run lint` pass.
- **Screenshot every page at 1440px and 390px before and after**, save them to
  `docs/ui_screenshots/`, and check them yourself before reporting done.

## 6. Reference mockups

Visual reference for sections 2–4 (Overview and Investigate), built with real values
from this repo's results: https://claude.ai/artifact/4hxbYyqx6kwne6mr4GSu5t
Match the direction (type, colour use, layout, honesty notes), not pixel values.
Every number in the real app still comes from the API.

## 7. Plain language (required)

The site must make sense to someone with no ML or finance background. Pattern, as in the
reference mockups:
- **Plain first, technical second.** Headings and main text in everyday words
  ("How good is each model at finding illicit payments?"); the technical name follows
  in smaller grey text ("Technical: test PR-AUC · paper_refit · mean ± 1 s.d.").
- Say "payment" in UI copy, not "transaction node"; "alert line", not "threshold";
  "false alarm (false positive)"; "connected payments", not "neighbourhood/edges".
- Every page opens with a 2–3 sentence **"In short"** summary generated from API data.
- Every chart has a **"How to read this"** `<details>` block: what one mark means,
  which direction is better, and what "random guessing" scores.
- A **glossary** ("Words used on this site") reachable from every page; terms in body
  copy link to their entry.
- Explanations always say what they are *not*: "what the model paid attention to,
  not proof of what happened". Anonymised features are explained once in plain words.
- Never show a bare number without its scale (e.g. "0.996 out of 1", "random ≈ 0.07").
