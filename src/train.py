import argparse
import json
import random
import time
import platform
import subprocess
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .data import prepare, targets, RESULTS, ARTIFACTS, ROOT, SEED
from .models import ItemCF, LightGBMRec, create_lru
from .metrics import topk, ranking_metrics


def atomic_dump(obj, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(obj, temporary)
    temporary.replace(path)


def evaluate(score_fn, bundle, target, name, split):
    records = []
    for user, relevant in target.items():
        ranked = topk(score_fn(user), bundle['seen'].get(user, ()))
        records.append(dict(user_id=user, model=name, split=split,
                            **ranking_metrics(ranked, relevant)))
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError('No eligible evaluation users')
    summary = frame[['Recall@10','NDCG@10','HitRate@10']].mean().to_dict()
    return dict(model=name, split=split, users=len(frame), **summary), frame


def sequence_windows(histories, length):
    xx, yy = [], []
    for history in histories.values():
        seq = np.asarray(history, dtype=np.int64) + 1
        for start in range(0, max(0,len(seq)-1), length):
            chunk = seq[start:start+length+1]
            if len(chunk) < 2:
                continue
            x, y = np.zeros(length,dtype=np.int64), np.zeros(length,dtype=np.int64)
            x[-(len(chunk)-1):], y[-(len(chunk)-1):] = chunk[:-1], chunk[1:]
            xx.append(x); yy.append(y)
    return TensorDataset(torch.from_numpy(np.stack(xx)), torch.from_numpy(np.stack(yy)))


def lru_scores(model, histories, users, n_items, length, device):
    model.eval()
    output = {}
    with torch.no_grad():
        for start in range(0,len(users),64):
            batch_users = users[start:start+64]
            seqs = np.zeros((len(batch_users),length),dtype=np.int64)
            for row,user in enumerate(batch_users):
                history = histories[user][-length:] + 1
                seqs[row,-len(history):] = history
            scores = model(torch.from_numpy(seqs).to(device))[0][:,-1,1:].float().cpu().numpy()
            for user,score in zip(batch_users,scores):
                output[user] = score
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--max-len',type=int,default=64)
    parser.add_argument('--hidden',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--tune-users',type=int,default=500)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(8)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    bundle = prepare()
    val, val_coverage = targets(bundle,'val')
    test, test_coverage = targets(bundle,'test')
    rng = np.random.default_rng(SEED)
    tune_ids = sorted(rng.choice(list(val),min(args.tune_users,len(val)),replace=False).tolist())
    tune = {u:val[u] for u in tune_ids}
    trials, rows, all_metrics = [], [], []
    n_items = len(bundle['catalog'])
    started = time.perf_counter()
    best_cf, best_metric = None, -1
    for k in [40,80]:
        t = time.perf_counter()
        cf = ItemCF(k).fit(bundle['train'],n_items)
        metric,_ = evaluate(lambda u:cf.score(bundle['histories'][u]),bundle,tune,'ItemCF','validation_tuning')
        trials.append(dict(**metric,config={'neighbors':k},seconds=time.perf_counter()-t))
        print('CF',k,metric,flush=True)
        if metric['NDCG@10']>best_metric:
            best_cf,best_metric = cf,metric['NDCG@10']
    atomic_dump(best_cf,ARTIFACTS/'itemcf.joblib')
    best_lgb, best_metric = None,-1
    for leaves in [31,63]:
        t = time.perf_counter()
        lgb = LightGBMRec().fit(bundle,leaves)
        metric,_ = evaluate(lgb.score,bundle,tune,'LightGBM','validation_tuning')
        trials.append(dict(**metric,config={'num_leaves':leaves},seconds=time.perf_counter()-t))
        print('LightGBM',leaves,metric,flush=True)
        if metric['NDCG@10']>best_metric:
            best_lgb,best_metric = lgb,metric['NDCG@10']
    atomic_dump(best_lgb,ARTIFACTS/'lightgbm.joblib')
    model = create_lru(n_items,args.hidden).to(device)
    dataset = sequence_windows(bundle['histories'],args.max_len)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    best_metric, stale, history, best_epoch = -1,0,[],0
    print('LRURec device:',device,'windows:',len(dataset),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train(); epoch_start=time.perf_counter(); total=0; tokens=0
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores=model(x)[0]
            loss=loss_fn(scores.reshape(-1,n_items+1),y.reshape(-1))
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            count=int((y!=0).sum()); total+=loss.item()*count; tokens+=count
        scores=lru_scores(model,bundle['histories'],tune_ids,n_items,args.max_len,device)
        metric,_=evaluate(scores.__getitem__,bundle,tune,'LRURec','validation_tuning')
        history.append(dict(epoch=epoch,loss=total/tokens,seconds=time.perf_counter()-epoch_start,**metric))
        print('LRURec',history[-1],flush=True)
        if metric['NDCG@10']>best_metric:
            best_metric=metric['NDCG@10']; stale=0; best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'hidden':args.hidden,'max_len':args.max_len,
                        'n_items':n_items,'epoch':epoch},ARTIFACTS/'lrurec.pt')
        else:
            stale+=1
            if stale>=3: break
    checkpoint=torch.load(ARTIFACTS/'lrurec.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    for split,target in [('validation',val),('test',test)]:
        lru=lru_scores(model,bundle['histories'],list(target),n_items,args.max_len,device)
        for name,fn in [('ItemCF',lambda u:best_cf.score(bundle['histories'][u])),
                        ('LightGBM',best_lgb.score),('LRURec',lru.__getitem__)]:
            metric,frame=evaluate(fn,bundle,target,name,split)
            all_metrics.append(metric); rows.append(frame)
            print('FINAL',metric,flush=True)
    pd.DataFrame(all_metrics).to_csv(RESULTS/'metrics.csv',index=False)
    pd.concat(rows).to_csv(RESULTS/'per_user_metrics.csv',index=False)
    pd.DataFrame(history).to_csv(RESULTS/'training_history.csv',index=False)
    (RESULTS/'tuning.json').write_text(json.dumps(trials,indent=2),encoding='utf-8')
    protocol=dict(seed=SEED,device=device,torch=torch.__version__,python=platform.python_version(),
                  gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
                  elapsed_seconds=time.perf_counter()-started,validation=val_coverage,test=test_coverage,
                  tuning_users=len(tune),selected_cf_neighbors=best_cf.k,
                  selected_lgb_leaves=best_lgb.model.num_leaves,lgb_training=best_lgb.training_info,
                  lru=dict(hidden=args.hidden,blocks=2,max_len=args.max_len,batch_size=args.batch_size,
                           best_epoch=best_epoch,epochs_run=len(history),lr=.001,weight_decay=.01,patience=3),
                  upstream_commit=subprocess.check_output(['git','-C',str(ROOT/'vendor/LRURec'),'rev-parse','HEAD'],text=True).strip(),
                  limitations=['Warm-start evaluation excludes customers without positive training history.',
                               'Training-only catalog excludes unseen held-out items.',
                               'LRURec uses 64-step windows by default, not the paper default 200.',
                               'LightGBM uses fixed inner-training metadata/history features and sampled negatives.',
                               'Validation observations are not appended to test histories; all models remain frozen.',
                               'CUDA kernels may introduce small nondeterministic differences.'])
    (RESULTS/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    atomic_dump(bundle,ARTIFACTS/'bundle.joblib')
    print('Saved all artifacts and results.',flush=True)


if __name__=='__main__':
    main()
