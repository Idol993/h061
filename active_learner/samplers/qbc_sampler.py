from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QBCResult:
    scores: np.ndarray
    committee_votes: np.ndarray
    disagreement: np.ndarray

    def __len__(self) -> int:
        return len(self.scores)


class QBCSampler:
    def __init__(self, seed: int = 42, n_committee: int = 5) -> None:
        self.seed = seed
        self.n_committee = n_committee

    def _train_committee(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_committee: int,
    ) -> list:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB

        base_estimators = [
            LogisticRegression(max_iter=1000, random_state=self.seed),
            RandomForestClassifier(n_estimators=50, random_state=self.seed, n_jobs=-1),
            DecisionTreeClassifier(random_state=self.seed),
            GaussianNB(),
            GradientBoostingClassifier(n_estimators=50, random_state=self.seed),
        ]

        committee = []
        rng = np.random.RandomState(self.seed)
        n_samples = len(X_train)

        for i in range(n_committee):
            if i < len(base_estimators):
                clf = base_estimators[i]
            else:
                clf = DecisionTreeClassifier(random_state=self.seed + i)

            if n_samples > 100:
                bootstrap_idx = rng.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_train[bootstrap_idx]
                y_boot = y_train[bootstrap_idx]
            else:
                X_boot = X_train
                y_boot = y_train

            try:
                clf.fit(X_boot, y_boot)
                committee.append(clf)
            except Exception:
                continue

        if not committee:
            clf = LogisticRegression(max_iter=1000, random_state=self.seed)
            clf.fit(X_train, y_train)
            committee.append(clf)

        return committee

    def _disagreement_kl_divergence(self, all_probs: np.ndarray) -> np.ndarray:
        n_models, n_samples, n_classes = all_probs.shape

        avg_probs = all_probs.mean(axis=0)
        avg_probs = np.clip(avg_probs, 1e-12, 1.0)
        avg_probs = avg_probs / avg_probs.sum(axis=1, keepdims=True)

        kl_values = np.zeros(n_samples, dtype=np.float64)
        for m in range(n_models):
            m_probs = np.clip(all_probs[m], 1e-12, 1.0)
            m_probs = m_probs / m_probs.sum(axis=1, keepdims=True)
            kl = (m_probs * (np.log(m_probs) - np.log(avg_probs))).sum(axis=1)
            kl_values += kl

        return kl_values / n_models

    def _vote_entropy(self, votes: np.ndarray) -> np.ndarray:
        n_samples, n_models = votes.shape
        entropies = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            unique, counts = np.unique(votes[i], return_counts=True)
            probs = counts / counts.sum()
            probs = np.clip(probs, 1e-12, 1.0)
            ent = -(probs * np.log2(probs)).sum()
            max_ent = np.log2(len(unique)) if len(unique) > 1 else 1.0
            if max_ent > 0:
                entropies[i] = ent / max_ent
            else:
                entropies[i] = 0.0

        return entropies

    def score(
        self,
        X_unlabeled: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        baseline_predictor=None,
    ) -> QBCResult:
        X_unlabeled = np.asarray(X_unlabeled, dtype=np.float64)
        if X_unlabeled.ndim != 2:
            X_unlabeled = X_unlabeled.reshape(len(X_unlabeled), -1)

        committee = []
        if X_train is not None and y_train is not None and len(X_train) > 0:
            try:
                committee = self._train_committee(X_train, y_train, self.n_committee)
            except Exception:
                committee = []

        if not committee and baseline_predictor is not None:
            all_probs_list = []
            votes_list = []
            for _ in range(max(self.n_committee, 3)):
                try:
                    probs = baseline_predictor.predict_proba(X_unlabeled)
                    noise = np.random.RandomState(self.seed + _).normal(0, 0.05, size=probs.shape)
                    noisy_probs = np.clip(probs + noise, 0, 1)
                    noisy_probs = noisy_probs / noisy_probs.sum(axis=1, keepdims=True)
                    all_probs_list.append(noisy_probs)
                    votes_list.append(np.argmax(noisy_probs, axis=1))
                except Exception:
                    pass

            if all_probs_list:
                all_probs = np.stack(all_probs_list, axis=0)
                votes = np.stack(votes_list, axis=1)
                disagreement = self._disagreement_kl_divergence(all_probs)
                if disagreement.max() > 0:
                    scores = disagreement / disagreement.max()
                else:
                    scores = self._vote_entropy(votes)

                return QBCResult(
                    scores=scores.astype(np.float32),
                    committee_votes=votes,
                    disagreement=disagreement.astype(np.float32),
                )

        if not committee:
            rng = np.random.RandomState(self.seed)
            n = len(X_unlabeled)
            dummy_scores = rng.rand(n).astype(np.float32)
            dummy_votes = rng.randint(0, 2, size=(n, 3))
            return QBCResult(
                scores=dummy_scores,
                committee_votes=dummy_votes,
                disagreement=dummy_scores,
            )

        all_probs_list = []
        votes_list = []

        for clf in committee:
            try:
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_unlabeled)
                else:
                    preds = clf.predict(X_unlabeled)
                    n_classes = len(np.unique(preds)) if len(np.unique(preds)) > 1 else 2
                    probs = np.zeros((len(X_unlabeled), n_classes), dtype=np.float64)
                    for i, p in enumerate(preds):
                        probs[i, int(p) % n_classes] = 1.0
                all_probs_list.append(np.asarray(probs, dtype=np.float64))
                votes_list.append(np.argmax(probs, axis=1))
            except Exception:
                continue

        if not all_probs_list:
            rng = np.random.RandomState(self.seed)
            n = len(X_unlabeled)
            return QBCResult(
                scores=rng.rand(n).astype(np.float32),
                committee_votes=rng.randint(0, 2, size=(n, 3)),
                disagreement=rng.rand(n).astype(np.float32),
            )

        n_classes = max(p.shape[1] for p in all_probs_list)
        padded_probs = []
        for p in all_probs_list:
            if p.shape[1] < n_classes:
                pad = np.zeros((p.shape[0], n_classes - p.shape[1]), dtype=np.float64)
                padded_probs.append(np.concatenate([p, pad], axis=1))
            else:
                padded_probs.append(p[:, :n_classes])

        all_probs = np.stack(padded_probs, axis=0)
        votes = np.stack(votes_list, axis=1)

        disagreement = self._disagreement_kl_divergence(all_probs)
        vote_ent = self._vote_entropy(votes)

        if disagreement.max() > 0:
            d_norm = disagreement / disagreement.max()
        else:
            d_norm = vote_ent

        scores = (0.7 * d_norm + 0.3 * vote_ent).astype(np.float32)

        return QBCResult(
            scores=scores,
            committee_votes=votes,
            disagreement=disagreement.astype(np.float32),
        )

    def rank_indices(
        self,
        X_unlabeled: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        baseline_predictor=None,
    ) -> np.ndarray:
        result = self.score(X_unlabeled, X_train, y_train, baseline_predictor)
        return np.argsort(-result.scores)

    def select_top_k(
        self,
        X_unlabeled: np.ndarray,
        k: int,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        baseline_predictor=None,
    ) -> np.ndarray:
        ranked = self.rank_indices(X_unlabeled, X_train, y_train, baseline_predictor)
        return ranked[:k]
