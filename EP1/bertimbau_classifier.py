"""
Fine-tuning leakage-aware do BERTimbau para train.xlsx.

Execução:
python bertimbau_gridsearch.py --input "Dataset\train.xlsx" --test "Dataset\test1.xlsx" --output ".\modelo_bertimbau"

Requisitos:
- pip install torch transformers datasets accelerate scikit-learn pandas openpyxl

O GridSearchCV executa 5 folds para cada configuração. Ele deve usar GPU e
permanece sequencial (n_jobs=1) para não disputar memória de vídeo.

Descobrir se a GPU é CUDA: 
- python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU nao detectada')"
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL = "neuralmind/bert-base-portuguese-cased"


def print_grid_results(search: GridSearchCV, metric: str = "accuracy") -> None:
    """Exibe cada configuração testada pela busca."""
    for index, (params, score, std) in enumerate(
        zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        ),
        start=1,
    ):
        params_text = " | ".join(
            f"{name}={value}" for name, value in params.items()
        )
        print(
            f"[Configuração {index:02d}] {params_text} | "
            f"{metric} média={score:.2%} ± {std:.2%}"
        )


def execute_grid_search(
    model: Any,
    param_grid: dict[str, list[Any]],
    texts: Any,
    labels: Any,
    groups: Any,
    *,
    metric: str = "accuracy",
    n_folds: int = 5,
) -> GridSearchCV:
    """Executa uma busca sem vazamento entre textos repetidos.

    `groups` deve conter o texto normalizado. Assim, versões idênticas de uma
    resposta nunca ficam uma no treino e outra na validação de um mesmo fold.
    """
    cv = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=metric,
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        verbose=2,
    )
    search.fit(texts, labels, groups=groups)
    print_grid_results(search, metric)
    return search


class BertimbauClassifier(ClassifierMixin, BaseEstimator):
    """Adaptador scikit-learn para fine-tuning de BERTimbau.

    O estimador recebe documentos completos. Internamente, textos maiores que
    `max_length` são quebrados em janelas sobrepostas. Na previsão, a média dos
    logits de todas as janelas produz uma única previsão por documento.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        learning_rate: float = 2e-5,
        num_train_epochs: int = 3,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        max_length: int = 512,
        stride: int = 128,
        warmup_ratio: float = 0.1,
        dataloader_num_workers: int = 4,
        tokenizer_batch_size: int = 256,
        random_state: int = 42,
        work_dir: str = "./bertimbau_grid_work",
    ) -> None:
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.stride = stride
        self.warmup_ratio = warmup_ratio
        self.dataloader_num_workers = dataloader_num_workers
        self.tokenizer_batch_size = tokenizer_batch_size
        self.random_state = random_state
        self.work_dir = work_dir

    @staticmethod
    def _as_text_list(texts: Iterable[Any]) -> list[str]:
        return pd.Series(texts).fillna("").astype(str).tolist()

    def _tokenize(
        self,
        texts: list[str],
        labels: np.ndarray | None = None,
    ) -> tuple[Dataset, np.ndarray]:
        # Tokenizar todo o fold de uma vez desperdiça memória de CPU e pode
        # falhar para coleções grandes. Os índices mantêm a associação janela
        # -> documento mesmo quando a tokenização é feita em lotes.
        all_encoded: dict[str, list[Any]] = {}
        document_index_parts: list[np.ndarray] = []
        for start in range(0, len(texts), self.tokenizer_batch_size):
            stop = min(start + self.tokenizer_batch_size, len(texts))
            encoded = self.tokenizer_(
                texts[start:stop],
                truncation=True,
                max_length=self.max_length,
                stride=self.stride,
                return_overflowing_tokens=True,
            )
            mapping = np.asarray(
                encoded.pop("overflow_to_sample_mapping"), dtype=np.int64
            ) + start
            document_index_parts.append(mapping)
            for name, values in encoded.items():
                all_encoded.setdefault(name, []).extend(values)

        document_index = np.concatenate(document_index_parts)

        if labels is not None:
            all_encoded["labels"] = labels[document_index].tolist()

        return Dataset.from_dict(all_encoded), document_index

    def fit(self, texts: Any, labels: Any) -> "BertimbauClassifier":
        if not 1 <= self.stride < self.max_length <= 512:
            raise ValueError("Use 0 < stride < max_length <= 512.")

        set_seed(self.random_state)
        has_cuda = torch.cuda.is_available()
        use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
        if has_cuda:
            # TF32 usa Tensor Cores nas GPUs NVIDIA Ampere ou posteriores sem
            # reduzir a estabilidade numérica materialmente neste fine-tuning.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.empty_cache()
        text_list = self._as_text_list(texts)
        self.classes_ = np.unique(np.asarray(labels))
        if len(self.classes_) < 2:
            raise ValueError("O treino exige ao menos duas classes.")

        label_to_id = {
            label: index for index, label in enumerate(self.classes_)
        }
        numeric_labels = np.asarray(
            [label_to_id[label] for label in labels], dtype=np.int64
        )

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        train_dataset, _ = self._tokenize(text_list, numeric_labels)

        id_to_label = {
            index: str(label) for index, label in enumerate(self.classes_)
        }
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.classes_),
            id2label=id_to_label,
            label2id=label_to_id,
        )

        run_dir = Path(self.work_dir) / "current_fit"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Transformers muda alguns nomes/opções entre versões. Filtrar a
        # configuração pela assinatura instalada preserva os ganhos que ela
        # suporta sem impedir o treinamento em versões mais antigas.
        training_options = {
            "output_dir": str(run_dir),
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size * 2,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": 1.0,
            "save_strategy": "no",
            "eval_strategy": "no",
            "logging_strategy": "steps",
            "logging_steps": 50,
            "report_to": "none",
            "bf16": use_bf16,
            "fp16": has_cuda and not use_bf16,
            "tf32": has_cuda,
            "optim": "adamw_torch_fused" if has_cuda else "adamw_torch",
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": has_cuda,
            "seed": self.random_state,
        }
        supported = set(inspect.signature(TrainingArguments.__init__).parameters)
        args = TrainingArguments(**{
            key: value for key, value in training_options.items()
            if key in supported
        })

        trainer_options = {
            "model": self.model_,
            "args": args,
            "train_dataset": train_dataset,
            "processing_class": self.tokenizer_,
            "tokenizer": self.tokenizer_,
            "data_collator": DataCollatorWithPadding(tokenizer=self.tokenizer_),
        }
        trainer_supported = set(inspect.signature(Trainer.__init__).parameters)
        self.trainer_ = Trainer(**{
            key: value for key, value in trainer_options.items()
            if key in trainer_supported
        })
        self.trainer_.train()
        return self

    def predict_proba(self, texts: Any) -> np.ndarray:
        if not hasattr(self, "trainer_"):
            raise RuntimeError("Chame fit antes de predict_proba.")

        text_list = self._as_text_list(texts)
        dataset, document_index = self._tokenize(text_list)
        logits = self.trainer_.predict(dataset).predictions

        document_logits = np.zeros(
            (len(text_list), len(self.classes_)), dtype=np.float64
        )
        np.add.at(document_logits, document_index, logits)
        chunks_per_document = np.bincount(
            document_index,
            minlength=len(text_list),
        )
        document_logits /= chunks_per_document[:, None]

        shifted = document_logits - document_logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, texts: Any) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        return self.classes_[np.argmax(probabilities, axis=1)]


def load_data(path: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    frame = pd.read_excel(path)
    required_columns = {"resp_text", "clarity"}
    missing = required_columns - set(frame.columns)
    if missing:
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


def export_test_predictions(
    model: BertimbauClassifier,
    test_path: Path,
    output_path: Path,
) -> Path:
    """Prediz o teste e preserva todas as suas colunas na planilha de saída."""
    test = pd.read_excel(test_path)
    required_columns = {"resp_text", "clarity"}
    missing = required_columns - set(test.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no teste: {sorted(missing)}")
    if test["clarity"].notna().any():
        raise ValueError(
            "A coluna clarity do teste deve estar vazia; o script não sobrescreve rótulos."
        )

    test["clarity"] = model.predict(test["resp_text"].fillna("").astype(str))
    result_path = output_path / "test1_rotulado.xlsx"
    test.to_excel(result_path, index=False)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--test", type=Path, required=True,
        help="test1.xlsx com as colunas resp_text e clarity vazia.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch por GPU. Use 32 em GPU com VRAM suficiente; reduza se der falta de memória.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Acumula gradientes para simular batch maior sem aumentar VRAM.",
    )
    parser.add_argument(
        "--dataloader-workers", type=int,
        default=min(4, os.cpu_count() or 1),
        help="Processos de preparação de lotes pela CPU.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("AVISO: nenhuma GPU CUDA detectada; o treinamento será lento.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    texts, labels, groups = load_data(args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    estimator = BertimbauClassifier(
        work_dir=str(args.output / "grid_work"),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_workers,
    )
    param_grid = {
        "learning_rate": [1e-5, 2e-5, 3e-5],
        "num_train_epochs": [3, 4],
        "weight_decay": [0.0, 0.01],
    }

    search = execute_grid_search(
        model=estimator,
        param_grid=param_grid,
        texts=texts,
        labels=labels,
        groups=groups,
        metric="accuracy",
        n_folds=5,
    )

    # GridSearchCV refaz automaticamente o melhor modelo com todos os dados.
    best_model = search.best_estimator_
    best_model.model_.save_pretrained(args.output / "model")
    best_model.tokenizer_.save_pretrained(args.output / "model")
    predictions_path = export_test_predictions(best_model, args.test, args.output)

    summary = {
        "best_params": search.best_params_,
        "cross_validation_accuracy": float(search.best_score_),
        "classes": best_model.classes_.tolist(),
        "model_name": best_model.model_name,
        "test_file": str(args.test),
        "predictions_file": str(predictions_path),
        "rows_predicted": int(len(pd.read_excel(predictions_path))),
        "notes": (
            "A acurácia é a média de 5 folds StratifiedGroupKFold, "
            "agrupados por texto normalizado."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # O diretório de trabalho contém apenas arquivos temporários do Trainer.
    shutil.rmtree(args.output / "grid_work", ignore_errors=True)

    print("\nMelhor configuração:", search.best_params_)
    print(f"Acurácia média validada: {search.best_score_:.2%}")
    print(f"Modelo salvo em: {args.output / 'model'}")
    print(f"Teste rotulado salvo em: {predictions_path}")


if __name__ == "__main__":
    main()