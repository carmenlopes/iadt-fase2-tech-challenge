"""Carregamento e divisao reproduzivel dos dados da Fase 1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

MALIGNANT_LABEL = 1
BENIGN_LABEL = 0
CLASS_NAMES = {BENIGN_LABEL: "benigno", MALIGNANT_LABEL: "maligno"}


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: list[int]
    test_indices: list[int]
    dataset_sha256: str
    split_seed: int
    test_size: float


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """Carrega o CSV Kaggle e converte M=1 e B=0, como na Fase 1."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)
    if "diagnosis" not in frame:
        raise ValueError("Coluna obrigatoria 'diagnosis' nao encontrada.")
    labels = set(frame["diagnosis"].dropna().unique())
    if labels != {"M", "B"}:
        raise ValueError(f"Rotulos inesperados em diagnosis: {sorted(labels)}")
    y = frame["diagnosis"].map({"M": MALIGNANT_LABEL, "B": BENIGN_LABEL})
    X = frame.drop(columns=["diagnosis", "id", "Unnamed: 32"], errors="ignore")
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("O dataset contem valores ausentes apos a limpeza.")
    return X, y.astype(int), digest


def make_original_split(
    path: str | Path, *, seed: int = 42, test_size: float = 0.20
) -> DatasetSplit:
    """Recria exatamente a divisao estratificada documentada na Fase 1."""
    X, y, digest = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=[int(i) for i in X_train.index],
        test_indices=[int(i) for i in X_test.index],
        dataset_sha256=digest,
        split_seed=seed,
        test_size=test_size,
    )

