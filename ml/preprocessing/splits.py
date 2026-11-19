"""Chronological split definitions.

Two protocols (see docs/00_design_analysis.md §7):

* ``paper``            train 1–29, val 30–34, test 35–49   (Weber et al. 2019 boundary)
* ``paper_refit``      train 1–34, no val,   test 35–49   (final refit with val-chosen HPs)
* ``fc_T``             forward chaining: train 1..T-5, val T-4..T, test T+1..T+5

Every split is checked to be disjoint and strictly ordered in time:
max(train) < min(val) <= max(val) < min(test).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Split:
    name: str
    train_steps: tuple[int, ...]
    val_steps: tuple[int, ...]
    test_steps: tuple[int, ...]
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        check_chronological(self)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("train_steps", "val_steps", "test_steps"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Split":
        return cls(
            name=d["name"],
            train_steps=tuple(d["train_steps"]),
            val_steps=tuple(d["val_steps"]),
            test_steps=tuple(d["test_steps"]),
            meta=d.get("meta", {}),
        )


def _rng(a: int, b: int) -> tuple[int, ...]:
    return tuple(range(a, b + 1)) if b >= a else ()


def check_chronological(s: Split) -> None:
    tr, va, te = set(s.train_steps), set(s.val_steps), set(s.test_steps)
    if not tr or not te:
        raise ValueError(f"{s.name}: train and test must be non-empty")
    if tr & va or tr & te or va & te:
        raise ValueError(f"{s.name}: split step sets overlap")
    if va and not (max(tr) < min(va) and max(va) < min(te)):
        raise ValueError(f"{s.name}: steps not ordered train < val < test")
    if not va and not max(tr) < min(te):
        raise ValueError(f"{s.name}: steps not ordered train < test")


def paper_splits(cfg: dict[str, Any]) -> list[Split]:
    p = cfg["protocols"]["paper"]
    t0, t1 = p["train_steps"]
    v0, v1 = p["val_steps"]
    s0, s1 = p["test_steps"]
    return [
        Split("paper", _rng(t0, v0 - 1), _rng(v0, v1), _rng(s0, s1),
              meta={"protocol": "paper"}),
        Split("paper_refit", _rng(t0, t1), (), _rng(s0, s1),
              meta={"protocol": "paper", "refit": True}),
    ]


def forward_chaining_splits(cfg: dict[str, Any], max_step: int = 49) -> list[Split]:
    fc = cfg["protocols"]["forward_chaining"]
    vw, tw = int(fc["val_width"]), int(fc["test_width"])
    out = []
    for cut in fc["cuts"]:
        test_end = min(cut + tw, max_step)
        out.append(Split(
            f"fc_{cut}",
            _rng(1, cut - vw),
            _rng(cut - vw + 1, cut),
            _rng(cut + 1, test_end),
            meta={"protocol": "forward_chaining", "cut": cut},
        ))
    return out


def all_splits(cfg: dict[str, Any], max_step: int = 49) -> list[Split]:
    return paper_splits(cfg) + forward_chaining_splits(cfg, max_step)


def save_splits(splits: list[Split], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in splits], f, indent=2)


def load_splits(path) -> dict[str, Split]:
    with open(path, encoding="utf-8") as f:
        return {d["name"]: Split.from_dict(d) for d in json.load(f)}
