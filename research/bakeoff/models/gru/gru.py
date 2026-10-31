"""GRU contender: OOF for one split, or the final fit on all eligible events.

usage: CUDA_VISIBLE_DEVICES=k .venv/bin/python models/gru/gru.py cv event|mission
       CUDA_VISIBLE_DEVICES=k .venv/bin/python models/gru/gru.py final
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BAKE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, BAKE); sys.path.insert(0, os.path.join(HERE, "final"))
import numpy as np, torch
import harness as H
import gru_lib as G
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

torch.set_num_threads(4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ev = H.load_events(); keys = ev.event_key.values; y = ev.y_tier.values.astype(np.int64); mis = ev.mission_id.values
S = H.load_sequences()
assert (S["event_key"] == keys).all()
M = S["mask"]; L = S["lengths"].astype(np.int64)
F = G.engineer(S["X"], M, S["feature_names"])
cfg = dict(G.CFG)
cfg.update(json.loads(os.environ.get("GRU_CFG", "{}")))   # inner-split tuning overrides; the chosen config is frozen into gru_lib.CFG
G.CFG.update(cfg)


def inner_split(trn, grouped, seed):
    """inner early-stopping split of the training positions (mission-grouped on the mission split / final fit)"""
    if grouped:
        itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(trn, groups=mis[trn]))
    else:
        itr, iva = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(trn, y[trn]))
    return trn[itr], trn[iva]


def fit_ensemble(trn, grouped, log):
    mu, sd = G.fit_norm(F[trn], M[trn])                 # standardisation on the training folds only
    Z = G.apply_norm(F, M, mu, sd)
    w = G.class_weights(y[trn])
    models = []
    for s in range(cfg["n_seeds"]):
        a, b = inner_split(trn, grouped, s)
        m, ep, vl = G.train_one(Z[a], L[a], y[a], Z[b], L[b], y[b], w, seed=s, dev=dev, cfg=cfg)
        pv = G.proba(G.predict_logits(m, Z[b], L[b], dev), w)[:, 2]; yc = (y[b] == 2).astype(int)
        log.append(dict(seed=s, best_epoch=ep, val_wce=round(vl, 4), n_inner_val=int(len(b)), crit_inner_val=int(yc.sum()),
                        val_auc=round(float(roc_auc_score(yc, pv)), 4) if 0 < yc.sum() < len(yc) else None,
                        val_ap=round(float(average_precision_score(yc, pv)), 4) if yc.sum() > 0 else None))
        models.append(m)
    return models, Z, w, mu, sd


def main():
    mode = sys.argv[1]
    t0 = time.time()
    if mode == "cv":
        split = sys.argv[2]
        P = np.zeros((len(y), 3)); log = {}
        for k, trn, tst in H.folds(split):
            fl = []
            models, Z, w, _, _ = fit_ensemble(trn, split == "mission", fl)
            P[tst] = np.mean([G.proba(G.predict_logits(m, Z[tst], L[tst], dev), w) for m in models], 0)
            log[int(k)] = fl
            print(split, "fold", k, fl, round(time.time() - t0, 1), "s", flush=True)
        P /= P.sum(1, keepdims=True)
        H.write_oof("gru", split, keys, P)
        json.dump(dict(split=split, folds=log, runtime_s=time.time() - t0, cfg=cfg),
                  open(os.path.join(HERE, f"cv_{split}_log.json"), "w"), indent=1)
    elif mode == "tune":   # inner-split metrics only; outer test folds are never predicted
        split, tag = sys.argv[2], sys.argv[3]
        log = {}
        for k, trn, tst in H.folds(split):
            fl = []; fit_ensemble(trn, split == "mission", fl); log[int(k)] = fl
        allr = [r for v in log.values() for r in v]
        summ = dict(tag=tag, split=split, cfg=cfg, mean_val_wce=float(np.mean([r["val_wce"] for r in allr])),
                    mean_val_auc=float(np.nanmean([r["val_auc"] or np.nan for r in allr])),
                    mean_val_ap=float(np.nanmean([r["val_ap"] or np.nan for r in allr])),
                    mean_best_epoch=float(np.mean([r["best_epoch"] for r in allr])), runtime_s=time.time() - t0, folds=log)
        os.makedirs(os.path.join(HERE, "tune"), exist_ok=True)
        json.dump(summ, open(os.path.join(HERE, "tune", f"{tag}_{split}.json"), "w"), indent=1)
        print({k: v for k, v in summ.items() if k != "folds"})
    elif mode == "final":
        fl = []
        models, Z, w, mu, sd = fit_ensemble(np.arange(len(y)), True, fl)
        out = os.path.join(HERE, "final"); os.makedirs(out, exist_ok=True)
        for s, m in enumerate(models):
            torch.save(m.state_dict(), os.path.join(out, f"seed{s}.pt"))
        np.savez(os.path.join(out, "norm.npz"), mu=mu, sd=sd, class_weight=w)
        json.dump(dict(cfg=cfg, n_features=int(F.shape[2]), raw_names=list(G.RAW_NAMES), seeds=fl,
                       n_train=int(len(y)), runtime_s=time.time() - t0), open(os.path.join(out, "meta.json"), "w"), indent=1)
        # sanity: in-sample prediction through the saved artefacts happens in predict.py
        print("final", fl, round(time.time() - t0, 1), "s")
    print("done", mode, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
