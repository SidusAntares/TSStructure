# Unified TimeMatch Phase Design

## Goal

Keep one complete TimeMatch training implementation with two experiment settings:
`alpha_candidates={0}` for exact Original TimeMatch degeneration, and
`alpha_candidates={0,0.25,0.5,0.75,1}` for AM-selected nonlinear residual Domain Phase.

## Runtime boundary

The semantic path is always `Input -> PSE -> LTAE -> classifier`. Stage 1 is source-only
Original TimeMatch training. Stage 2 retains TimeMatch IS/AM scalar shift, EMA teacher,
confidence pseudo-labeling, focal loss, optimizer and scheduler. Geometry never enters the
classifier and never receives gradients.

With only alpha zero, registration and the frozen geometry branch are not constructed.
The timestamp transform is exactly `u + delta / 365`. With the full alpha bank, a frozen
copy of the Stage-1 PSE estimates one bootstrap-only shared residual phase. Each epoch first
selects scalar `delta`, then AM selects alpha at fixed delta.

## Files and cleanup

One source-only Stage 1 program produces the shared checkpoint, and one canonical Stage 2
program owns both alpha-zero and nonlinear adaptation. One four-GPU launcher composes the
complete flow and runs
AT1->DK1, DK1->FR2, FR1->AT1 and FR2->FR1. Generic dataset/split/registration helpers are
moved out of historical diagnostic scripts. Old StructureDA models, trainers, objectives,
13A/13B/14/15, oracle, V3 and obsolete launchers/tests are removed once unreachable.

## Verification

Tests prove alpha-zero timestamp and logits equivalence, zero-mode geometry bypass,
full-bank registration use, target-label isolation, four-task mapping and absence of legacy
imports. Compileall, focused pytest, full pytest where the local environment permits,
shell syntax and diff checks are required. No real-data training runs locally.
