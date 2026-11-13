import html
import json
import pandas as pd
import streamlit as st
from src.data import ARTIFACTS, RESULTS

st.set_page_config(page_title='CineMatch | Movie discovery',page_icon='🎬',layout='wide')
st.markdown('''<style>
.stApp {background:#f7f8fc;color:#18243b;}
.block-container {padding-top:4.5rem;max-width:1440px;}
[data-testid="stSidebar"] {background:#e8edf7;}
.eyebrow {color:#5977b2;font-size:12px;letter-spacing:2px;font-weight:700;}
.hero {background:linear-gradient(115deg,#14243e,#263f67);padding:36px 40px;border-radius:20px;color:white;margin:12px 0 28px;}
.hero h1 {color:white!important;font-size:42px;margin:0 0 8px;}
.hero p {color:#c9d7ed;font-size:16px;margin:0;}
.movie {background:white;border:1px solid #e4e9f2;border-radius:16px;padding:22px;min-height:210px;margin-bottom:14px;}
.movie .rank {color:#6b87b4;font-size:12px;letter-spacing:2px;font-weight:700;}
.movie h3 {color:#18243b;font-size:19px;line-height:1.4;padding:12px 0 4px;}
.movie p {font-size:13px;color:#63728a;}
.pill {display:inline-block;background:#eef2fa;border-radius:20px;color:#365b96;padding:4px 10px;font-size:11px;margin:4px 3px 0 0;}
[data-testid="stMetric"] {background:white;border:1px solid #e4e9f2;border-radius:14px;padding:18px;}
</style>''',unsafe_allow_html=True)
st.sidebar.markdown('## ◈ CineMatch')
st.sidebar.caption('CS5481 · MOVIELENS LAB')
if not (ARTIFACTS/'bundle.joblib').exists():
    st.title('CineMatch · 电影推荐实验室')
    st.info('模型尚未准备好。请先运行 python -m src.train，训练完成后刷新页面。')
    st.stop()

@st.cache_resource
def load_service():
    from src.service import RecommendationService
    return RecommendationService()

service=load_service()
bundle=service.bundle
users=sorted(bundle['histories'])
user=st.sidebar.selectbox('MovieLens 用户 / User',users,index=0)
model=st.sidebar.selectbox('推荐算法 / Algorithm',['ItemCF','LightGBM','LRURec'])
count=st.sidebar.slider('推荐数量 / Top K',5,20,10)
genres=['All genres']+sorted(set('|'.join(bundle['movies'].genres).split('|')))
genre=st.sidebar.selectbox('电影类型 / Genre',genres)
st.sidebar.divider()
st.sidebar.caption('只推荐该用户没有评分过的电影。模型使用训练数据；演示列表额外排除该用户全部已知评分。')
st.sidebar.markdown('[Source & reproduction ↗](https://github.com/winged-hussar-chat/cs5481_pro)')
st.markdown('<div class="eyebrow">DISCOVERY, GUIDED BY DATA</div>',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Your next great film.</h1><p>从百万条评分中，发现下一部值得看的电影。三种算法，同一个用户，更多可能。</p></div>',unsafe_allow_html=True)
tabs=st.tabs(['为你推荐 · Discover','算法对比 · Compare','数据探索 · Data','实验结果 · Results'])

with tabs[0]:
    c1,c2,c3=st.columns(3)
    c1.metric('当前用户',f'#{user}')
    c2.metric('训练期喜欢的电影',len(bundle['histories'][user]))
    c3.metric('候选电影库',f"{len(bundle['catalog']):,}")
    st.subheader(f'{model} 为你精选')
    st.caption('排序得分用于同一模型内部比较，不代表预测星级或跨模型可比的概率。')
    rec=service.recommend(user,model,count,genre)
    if len(rec)<5: st.info('该类型下可推荐电影不足 5 部，可以切换为 All genres。')
    cols=st.columns(5)
    for j,row in enumerate(rec.itertuples()):
        chips=''.join(f'<span class="pill">{html.escape(g)}</span>' for g in row.genres.split('|'))
        cols[j%5].markdown(f'<div class="movie"><div class="rank">PICK {row.rank:02d}</div><h3>{html.escape(row.title)}</h3>{chips}<p>Model score · {row.score:.3f}</p></div>',unsafe_allow_html=True)
    st.download_button('下载推荐列表 CSV',rec.to_csv(index=False).encode('utf-8-sig'),'recommendations.csv','text/csv')
    with st.expander('查看用户的训练期观影偏好'):
        liked=bundle['train'].query('user_id == @user and rating >= 4').merge(bundle['movies'],on='movie_id')
        st.dataframe(liked[['title','genres','rating']].head(30),hide_index=True,width='stretch')

with tabs[1]:
    st.subheader('Same viewer. Three perspectives.')
    st.caption('物品协同过滤寻找共同喜好；LightGBM 结合属性和历史特征；LRURec 建模观影序列。')
    lists={}
    for col,name in zip(st.columns(3),['ItemCF','LightGBM','LRURec']):
        frame=service.recommend(user,name,count,genre); lists[name]=set(frame.movie_id)
        with col:
            st.markdown(f'### {name}')
            st.dataframe(frame[['rank','title','genres']],hide_index=True,width='stretch')
    overlap=len(set.intersection(*lists.values()))
    st.info(f'三个推荐列表共同出现 {overlap} 部电影。列表差异反映模型学习到的偏好不同，并不直接等同于质量高低。')

with tabs[2]:
    st.subheader('A million ratings, made understandable.')
    stats=json.loads((RESULTS/'dataset.json').read_text())
    cs=st.columns(4)
    for c,label,value in zip(cs,['评分数','用户数','电影元数据','矩阵密度'],[f"{stats['ratings']:,}",f"{stats['users']:,}",f"{stats['movies_in_metadata']:,}",f"{stats['density']:.2%}"]): c.metric(label,value)
    a,b=st.columns(2)
    with a:
        st.markdown('#### 评分分布')
        st.bar_chart(bundle['ratings'].rating.value_counts().sort_index(),color='#567bc3')
    with b:
        st.markdown('#### 电影类型分布')
        counts=bundle['movies'].genres.str.split('|').explode().value_counts()
        st.bar_chart(counts,color='#69a89b')
    st.markdown('#### 时间切分与缺失值')
    st.dataframe(pd.DataFrame([{'split':k,'rows':v} for k,v in stats['splits'].items()]),hide_index=True)
    st.caption('探索图仅描述数据；模型特征使用训练数据计算。评分 ≥ 4 定义为喜欢。')
    st.dataframe(pd.read_csv(RESULTS/'data_dictionary.csv'),hide_index=True,width='stretch')
    st.markdown('[MovieLens 1M · GroupLens](https://grouplens.org/datasets/movielens/1m/)')

with tabs[3]:
    st.subheader('Evidence before preference.')
    metrics=pd.read_csv(RESULTS/'metrics.csv')
    split=st.radio('评估集',['test','validation'],horizontal=True)
    shown=metrics.query('split == @split').set_index('model')
    st.bar_chart(shown[['NDCG@10','Recall@10']],color=['#567bc3','#69a89b'])
    st.dataframe(shown.style.format({'NDCG@10':'{:.4f}','Recall@10':'{:.4f}','HitRate@10':'{:.4f}'}),width='stretch')
    protocol=json.loads((RESULTS/'protocol.json').read_text())
    st.caption('全候选集排名；统一时间划分；测试集只用于最终评估。仅评估有训练历史、且有可评估正例的用户。')
    coverage=protocol['test' if split=='test' else 'validation']
    st.info(f"本次评估覆盖 {coverage['evaluated_users']} / {coverage['total_relevant_users']} 位在该时段有正向评分的用户。冷启动用户不包含在指标中。")
    st.markdown('#### LRURec 训练曲线')
    history=pd.read_csv(RESULTS/'training_history.csv').set_index('epoch')
    left,right=st.columns(2)
    left.line_chart(history[['loss']]); right.line_chart(history[['NDCG@10']])
    with st.expander('实验配置与复现边界'):
        st.json(protocol)
    st.download_button('下载评估结果 CSV',metrics.to_csv(index=False),'metrics.csv','text/csv')

st.divider()
st.caption('CineMatch · CS5481 Project 1 · MovieLens 1M / GroupLens · Educational research demo')
