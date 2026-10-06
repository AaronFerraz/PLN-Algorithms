"""
TF-IDF com Regressão Logística Multiclasse
- TF-IDF de unigramas/bigramas

c1 / c234 / c5
2 n-gramas x 3 min_df x 3 max_df x 4 valores de C = 72 combinações
72 combinações x 5 folds = 360 treinamentos

Execução:
python baseline_classifier.py --input "Dataset\\train.xlsx" --test "Dataset\\test1.xlsx"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline



def build_model() -> Pipeline:
    """Cria o pipeline TF-IDF de palavras seguido de regressão logística."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
            ),
        ),
        (
            "logistic_regression",
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])


def normalized_groups(texts: pd.Series) -> pd.Series:
    """Agrupa respostas iguais para impedir vazamento entre folds."""
    return (
        texts.fillna("").astype(str)
        .str.normalize("NFKC")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def print_grid_results(search: GridSearchCV) -> None:
    for number, (params, mean, std) in enumerate(
        zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        ),
        start=1,
    ):
        configuration = " | ".join(
            f"{name}={value}" for name, value in params.items()
        )
        print(
            f"[{number:02d}] {configuration} | "
            f"accuracy={mean:.2%} ± {std:.2%}"
        )


def execute_grid_search(
    model: Pipeline,
    param_grid: dict[str, list[Any]],
    texts: pd.Series,
    labels: pd.Series,
    groups: pd.Series,
    *,
    n_folds: int = 5,
    n_jobs: int = 1,
) -> GridSearchCV:
    """Seleciona hiperparâmetros sem misturar textos repetidos entre folds."""
    cv = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42,
    )
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
        return_train_score=False,
        verbose=1,
    )
    search.fit(texts, labels, groups=groups)
    print_grid_results(search)
    return search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="train.xlsx")
    parser.add_argument("--test", type=Path, required=True, help="test1.xlsx")
    parser.add_argument(
        "--output", type=Path, default=Path("test1_rotulado.xlsx"),
        help="Planilha .xlsx de entrega.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Processos paralelos do Grid Search. Em Windows, 1 é mais estável.",
    )
    args = parser.parse_args()

    train = pd.read_excel(args.input, usecols=["resp_text", "clarity"])
    train = train.dropna(subset=["resp_text", "clarity"]).copy()
    texts = train["resp_text"].fillna("").astype(str)
    labels = train["clarity"].astype(str)
    groups = normalized_groups(texts)

    # 2 x 3 x 3 x 4 = 72 configurações, avaliadas em cinco folds.
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__max_df": [0.95, 0.98, 1.0],
        "logistic_regression__C": [0.01, 0.1, 1.0, 10.0],
    }
    result = execute_grid_search(
        model=build_model(),
        param_grid=param_grid,
        texts=texts,
        labels=labels,
        groups=groups,
        n_folds=5,
        n_jobs=args.n_jobs,
    )

    print("\n" + "=" * 70)
    print("MELHOR MODELO ENCONTRADO")
    print("=" * 70)
    print(f"Accuracy média OOF: {result.best_score_:.2%}")
    print("Hiperparâmetros:", result.best_params_)

    # refit=True já treinou o vencedor usando todas as linhas do treino.
    best_model = result.best_estimator_
    test = pd.read_excel(args.test)
    if "resp_text" not in test.columns or "clarity" not in test.columns:
        raise ValueError("O teste precisa conter as colunas resp_text e clarity.")
    if test["clarity"].notna().any():
        raise ValueError("A coluna clarity do teste deve estar vazia antes da previsão.")

    test["clarity"] = best_model.predict(
        test["resp_text"].fillna("").astype(str)
    )
    if set(test["clarity"].unique()) - set(labels.unique()):
        raise RuntimeError("O modelo produziu um rótulo ausente no treino.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    test.to_excel(args.output, index=False)
    joblib.dump(best_model, args.output.with_suffix(".joblib"))
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "oof_accuracy": float(result.best_score_),
                "best_params": result.best_params_,
                "classes": sorted(labels.unique().tolist()),
                "rows_predicted": int(len(test)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Planilha de entrega: {args.output}")


if __name__ == "__main__":
    main()