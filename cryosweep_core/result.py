from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

class Gate(BaseModel):
    model_config = ConfigDict(frozen=True)
    need: str
    reason: str
    remedy: dict[str, str] = Field(default_factory=dict)

class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)
    file: str
    sha256: str
    app_version: str | None
    config: dict[str, Any] = Field(default_factory=dict)

class FitResult(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())  # allow field named "model"
    model: str
    params: dict[str, float]
    sigma: dict[str, float] = Field(default_factory=dict)
    covariance: list[list[float]] = Field(default_factory=list)
    r2: float | None = None
    chi2_red: float | None = None
    n_points: int = 0
    fit_range: list[float] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)

class Diagnostic(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: str                                       # e.g. "outliers"
    severity: str = "info"                          # "info" | "warning"
    scope: str = ""                                 # human label, e.g. "bridge1 ρ(T) 0.5 Oe"
    message: str = ""                               # one-line human summary
    data: dict[str, Any] = Field(default_factory=dict)   # structured payload (outlier_stats, etc.)

class Result(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Literal["ok", "gated", "low_confidence", "error"]
    confidence: float = 0.0
    confidence_parts: dict[str, float | None] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    gate: list[Gate] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    provenance: Provenance

EXIT_CODES = {"ok": 0, "gated": 10, "low_confidence": 11, "error": 2}
