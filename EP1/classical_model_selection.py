"""
Seleção justa de classificadores clássicos para train.xlsx.

Execução:
- python classical_model_selection.py --input "Dataset\\train.xlsx" --test "Dataset\\test1.xlsx" --output ".\\classical_selection"

Todos os modelos usam StratifiedGroupKFold, agrupando textos normalizados.
Assim, uma resposta repetida não aparece ao mesmo tempo no treino e na validação.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=180_000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )


def char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.995,
        max_features=220_000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )


def lexical_features() -> FeatureUnion:
    """Duas visões complementares do texto, sem dados externos."""
    return FeatureUnion([
        ("word", word_tfidf()),
        ("char", char_tfidf()),
    ])


def lexical_feature_profiles() -> list[dict[str, list[object]]]:
    """Configurações estratégicas de TF-IDF de palavras e caracteres.

    Usar perfis evita um produto cartesiano gigante e redundante, mas ainda
    testa cobertura lexical, expressões e variações ortográficas.
    """
    profiles = [
        ((1, 1), 1, 0.98, (3, 5), 1, 0.995, {"word": 1.0, "char": 0.5}),
        ((1, 1), 2, 0.98, (3, 5), 2, 0.995, {"word": 1.0, "char": 1.0}),
        ((1, 2), 1, 0.98, (3, 5), 1, 0.995, {"word": 1.0, "char": 0.5}),
        ((1, 2), 2, 0.98, (3, 5), 2, 0.995, {"word": 1.0, "char": 1.0}),
        ((1, 2), 3, 0.95, (3, 5), 2, 0.995, {"word": 1.0, "char": 1.0}),
        ((1, 2), 2, 1.00, (3, 6), 1, 0.995, {"word": 0.75, "char": 1.0}),
        ((1, 2), 3, 1.00, (3, 6), 2, 0.995, {"word": 0.5, "char": 1.0}),
        ((1, 2), 1, 0.95, (2, 5), 1, 0.995, {"word": 1.0, "char": 1.0}),
        ((1, 2), 2, 0.98, (2, 5), 2, 0.995, {"word": 1.0, "char": 0.75}),
    ]
    return [
        {
            "features__word__ngram_range": [word_ngram],
            "features__word__min_df": [word_min_df],
            "features__word__max_df": [word_max_df],
            "features__char__ngram_range": [char_ngram],
            "features__char__min_df": [char_min_df],
            "features__char__max_df": [char_max_df],
            "features__transformer_weights": [weights],
        }
        for (
            word_ngram,
            word_min_df,
            word_max_df,
            char_ngram,
            char_min_df,
            char_max_df,
            weights,
        ) in profiles
    ]


def with_model_grid(
    feature_profiles: list[dict[str, list[object]]],
    model_grid: dict[str, list[object]],
) -> list[dict[str, list[object]]]:
    """Combina perfis de representação com os parâmetros do classificador."""
    return [{**profile, **model_grid} for profile in feature_profiles]


def load_data(path: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    frame = pd.read_excel(path)
    required = {"resp_text", "clarity"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")

    frame = frame.dropna(subset=["resp_text", "clarity"]).copy()
    texts = frame["resp_text"].astype(str)
    labels = frame["clarity"].astype(str)
    groups = (
        texts.str.normalize("NFKC")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return texts, labels, groups


def candidates() -> list[tuple[str, Pipeline, dict[str, list[object]] | list[dict[str, list[object]]]]]:
    lexical_profiles = lexical_feature_profiles()
    return [
        (
            "TF-IDF palavras + Regressão Logística",
            Pipeline([
                ("features", word_tfidf()),
                ("model", LogisticRegression(
                    solver="saga", max_iter=2000, random_state=42,
                )),
            ]),
            {
                "features__ngram_range": [(1, 1), (1, 2)],
                "features__min_df": [1, 2, 3],
                "features__max_df": [0.95, 0.98, 1.0],
                "model__C": [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
            },
        ),
        (
            "TF-IDF palavras+caracteres + Regressão Logística",
            Pipeline([
                ("features", lexical_features()),
                ("model", LogisticRegression(
                    solver="saga", max_iter=2000, random_state=42,
                )),
            ]),
            with_model_grid(
                lexical_profiles,
                {"model__C": [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]},
            ),
        ),
        (
            "TF-IDF palavras+caracteres + LinearSVC",
            Pipeline([
                ("features", lexical_features()),
                ("model", LinearSVC()),
            ]),
            with_model_grid(
                lexical_profiles,
                {"model__C": [0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]},
            ),
        ),
        (
            "TF-IDF palavras+caracteres + ComplementNB",
            Pipeline([
                ("features", lexical_features()),
                ("model", ComplementNB()),
            ]),
            with_model_grid(
                lexical_profiles,
                {"model__alpha": [0.005, 0.01, 0.03, 0.1, 0.3, 1.0]},
            ),
        ),
        (
            "TF-IDF palavras+caracteres + SGD modified-huber",
            Pipeline([
                ("features", lexical_features()),
                ("model", SGDClassifier(
                    loss="modified_huber", max_iter=2000, tol=1e-3,
                    random_state=42,
                )),
            ]),
            with_model_grid(
                lexical_profiles[:6],
                {
                    "model__alpha": [3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
                    "model__penalty": ["l2", "elasticnet"],
                    "model__l1_ratio": [0.15],
                },
            ),
        ),
    ]


def export_test_predictions(model: Pipeline, test_path: Path, output_dir: Path) -> Path:
    """Gera a planilha de entrega sem modificar a planilha de teste original."""
    test = pd.read_excel(test_path)
    required = {"resp_text", "clarity"}
    if missing := required - set(test.columns):
        raise ValueError(f"Colunas ausentes no teste: {sorted(missing)}")
    if test["clarity"].notna().any():
        raise ValueError("A coluna clarity do teste deve estar vazia.")

    test["clarity"] = model.predict(test["resp_text"].fillna("").astype(str))
    destination = output_dir / "test1_rotulado.xlsx"
    test.to_excel(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Use 1 no Windows para LinearSVC; paralelismo pode esgotar RAM.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    texts, labels, groups = load_data(args.input)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    results: list[dict[str, object]] = []
    best_search: GridSearchCV | None = None
    best_name = ""

    for name, pipeline, param_grid in candidates():
        print(f"\n### {name}")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=args.n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False,
            verbose=1,
        )
        search.fit(texts, labels, groups=groups)
        result = {
            "model": name,
            "mean_accuracy": float(search.best_score_),
            "std_accuracy": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "best_params": search.best_params_,
        }
        results.append(result)
        print(
            f"Melhor: {result['mean_accuracy']:.2%} ± "
            f"{result['std_accuracy']:.2%} | {result['best_params']}"
        )

        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search = search
            best_name = name

    assert best_search is not None
    results.sort(key=lambda item: float(item["mean_accuracy"]), reverse=True)
    
    print("\n=== RANKING FINAL ===")
    for position, result in enumerate(results, start=1):
        print(
            f"{position}. {result['model']}: "
            f"{result['mean_accuracy']:.2%} ± {result['std_accuracy']:.2%}"
        )

    joblib.dump(best_search.best_estimator_, args.output / "best_classical_model.joblib")
    predictions_path = export_test_predictions(
        best_search.best_estimator_, args.test, args.output
    )
    (args.output / "summary.json").write_text(
        json.dumps({
            "method": "5-fold StratifiedGroupKFold, groups=texto normalizado",
            "winner": best_name,
            "winner_accuracy": float(best_search.best_score_),
            "all_results": results,
            "test_file": str(args.test),
            "predictions_file": str(predictions_path),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nVencedor salvo em: {args.output / 'best_classical_model.joblib'}")
    print(f"Teste rotulado salvo em: {predictions_path}")


if __name__ == "__main__":
    main()