"""Export reproducible aggregate figures and an evaluation error breakdown."""
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import RESULTS, ARTIFACTS


def main():
    b=joblib.load(ARTIFACTS/'bundle.joblib')
    out=RESULTS/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':150,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    b['ratings'].rating.value_counts().sort_index().plot.bar(ax=axs[0],color='#567bc3',rot=0)
    axs[0].set(title='MovieLens 1M: rating distribution',xlabel='Rating',ylabel='Interactions')
    b['movies'].genres.str.split('|').explode().value_counts().sort_values().plot.barh(ax=axs[1],color='#69a89b')
    axs[1].set(title='Movie genres (multi-label)',xlabel='Movies',ylabel='')
    fig.tight_layout(); fig.savefig(out/'data_overview.png'); plt.close(fig)
    metrics=pd.read_csv(RESULTS/'metrics.csv')
    test=metrics.query("split == 'test'").set_index('model')
    fig,ax=plt.subplots(figsize=(8,4.5))
    test[['NDCG@10','Recall@10']].plot.bar(ax=ax,color=['#567bc3','#69a89b'],rot=0)
    ax.set(title='Full-catalog test evaluation · 990 eligible users',ylabel='Macro average',xlabel='')
    fig.tight_layout(); fig.savefig(out/'model_comparison.png'); plt.close(fig)
    history=pd.read_csv(RESULTS/'training_history.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].plot(history.epoch,history.loss,'o-',color='#567bc3'); axs[0].set(title='LRURec training loss',xlabel='Epoch',ylabel='Cross entropy')
    axs[1].plot(history.epoch,history['NDCG@10'],'o-',color='#69a89b'); axs[1].set(title='Validation NDCG@10',xlabel='Epoch')
    fig.tight_layout(); fig.savefig(out/'training_curve.png'); plt.close(fig)
    comparison=[]
    for name in ['train','val','test']:
        df=b[name]
        comparison.append({'split':name,'rows':len(df),'users':df.user_id.nunique(),'items':df.movie_id.nunique(),
                           'positive_rows':int((df.rating>=4).sum()),'mean_rating':float(df.rating.mean()),
                           'missing_values':int(df.isna().sum().sum())})
    pd.DataFrame(comparison).to_csv(RESULTS/'split_comparison.csv',index=False)
    per_user=pd.read_csv(RESULTS/'per_user_metrics.csv').query("split == 'test'").copy()
    sizes={u:len(h) for u,h in b['histories'].items()}
    per_user['history_size']=per_user.user_id.map(sizes)
    per_user['history_group']=pd.cut(per_user.history_size,[0,20,100,float('inf')],labels=['1-20','21-100','101+'])
    summary=per_user.groupby(['model','history_group'],observed=True).agg(users=('user_id','size'),recall=('Recall@10','mean'),ndcg=('NDCG@10','mean'),hit_rate=('HitRate@10','mean'))
    summary.to_csv(RESULTS/'error_analysis.csv')
    print(summary.to_string())


if __name__=='__main__': main()
