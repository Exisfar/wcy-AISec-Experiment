"""本机制作课堂汇报：读取容器导出的结果，同源输出 PPT、PDF 和讲稿。"""
from pathlib import Path
import json,sys,runpy,zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'presentation'))
from canvas import Deck,INK,BLUE,ORANGE,RED,GRAY,LIGHT,PAPER,WHITE,font_path

summary=pd.read_csv(ROOT/'results/appendixA_kaggle/recall_summary.csv')
mit=pd.read_csv(ROOT/'results/appendixA_mitigation/summary.csv')
rep=pd.read_csv(ROOT/'results/tables/agreement_representation.csv',index_col=0)
checks=pd.read_csv(ROOT/'results/tables/upstream_crosscheck.csv')
provenance=json.loads((ROOT/'results/provenance.json').read_text())
assert json.loads((ROOT/'results/appendixA_kaggle/verification.json').read_text())['complete']
assert json.loads((ROOT/'results/appendixA_mitigation/verification.json').read_text())['complete']

def score(exp,q,label,gender='Female'):
    return float(summary[(summary.experiment==exp)&(summary.fraction==q)&(summary.label==label)&(summary.gender==gender)]['mean'].iloc[0])
def weighted(method,gender,label):
    return float(mit[(mit.method==method)&(mit.gender==gender)&(mit.label==label)]['mean'].iloc[0])

hb=score(2,.5,'happy');hm=score(2,0,'happy');ab=score(3,.5,'angry');am=score(3,1,'angry')
mb=score(2,.5,'mean');mm=score(2,0,'mean')
values={'happy_balanced':f'{hb*100:.1f}','happy_missing':f'{hm*100:.1f}',
        'angry_balanced':f'{ab*100:.1f}','angry_missing':f'{am*100:.1f}',
        'macro_balanced':f'{mb*100:.1f}','macro_missing':f'{mm*100:.1f}',
        'weighted_female_before':f'{weighted("ERM","Female","happy")*100:.1f}',
        'weighted_female_after':f'{weighted("Reweighted","Female","happy")*100:.1f}',
        'weighted_male_before':f'{weighted("ERM","Male","happy")*100:.1f}',
        'weighted_male_after':f'{weighted("Reweighted","Male","happy")*100:.1f}'}
speech={k:v.format_map(values) for k,v in json.loads((ROOT/'presentation/speech_content.json').read_text()).items()}
D=Deck(ROOT);PAPER_SOURCE='Dominguez-Catena et al.  ·  TPAMI 2024  ·  arXiv:2303.15889v2'
TIMES=[25,30,75,55,100,80,80,90,70,60,70,80,80,90,95,60,75,80,90,70,45,45,90,45,120]
assert sum(TIMES[:21])==1500

def slide(title,source='',dark=False,cover=False):
    n=len(D.pages)+1
    section='汇报安排' if n==2 else '论文内容' if n<=9 else '复现工作' if n<=16 else '不足与改进' if n<=19 else '代码演示' if n==20 else '总结' if n==21 else '讨论'
    D.slide(title,section,TIMES[n-1],speech[str(n)],source=source,dark=dark,backup=n>21,cover=cover)

def big_num(x,y,num,label,c=BLUE,w=3.5):
    D.text(x,y,w,1.1,num,48,c,True)
    D.text(x,y+1.15,w,.65,label,19,INK)

def remark(s,y=6.15,c=BLUE):D.text(.75,y,11.8,.6,s,19,c,True)

# 01 / 参照课程封面的学校标识、深蓝标题区与浅色信息栏。
slide('论文复现\n数据集人口统计偏差',cover=True)
# 保留参考页顶部校徽及校名；重绘主题、英文论文名和汇报信息。
D.image(ROOT/'presentation/cover_reference.png',0,0,13.333,7.5)
COVER_BLUE='#194A6B'; COVER_LIGHT='#F7FAFC'; COVER_ACCENT='#9DD0F7'
D.rect(0,1.11,13.333,4.55,COVER_BLUE)
D.rect(0,5.66,13.333,1.84,COVER_LIGHT)
D.text(.6,1.78,12.13,1.05,'论文复现',47,WHITE,True,align='center')
D.text(.6,2.88,12.13,1.05,'数据集人口统计偏差',47,WHITE,True,align='center')
D.text(.7,4.17,11.93,1.12,'Metrics for Dataset Demographic Bias:\nA Case Study on Facial Expression Recognition',25,COVER_ACCENT,True,align='center')
D.text(.5,6.03,3.7,.48,'汇报组：第九组',21,INK,True,align='center')
D.line(4.28,6.02,4.28,6.39,'#4683BD',.8)
D.text(4.43,6.03,4.42,.48,'方向：AI公平性与偏差评估',18,INK,True,align='center')
D.line(9.02,6.02,9.02,6.39,'#4683BD',.8)
D.text(9.2,6.03,3.65,.48,'来源：IEEE TPAMI 2024',18,INK,True,align='center')
D.text(.7,6.83,11.93,.48,'论文汇报  |  2026年11月2日',22,INK,align='center')

# 封面后目录。
slide('汇报目录')
for index,(title,detail) in enumerate([
    ('论文内容','研究问题、指标体系与主要发现'),
    ('复现工作','数据与实验设计、指标对照及真实模型结果'),
    ('不足与改进','局限分析与重加权探索'),
    ('代码演示','三份Notebook的代码、注释与已完成输出'),
    ('讨论','两个问题及参考答案'),
]):
    y=1.95+index*.86
    D.text(.8,y,.65,.55,f'{index+1:02}',23,BLUE,True)
    D.text(1.7,y,3.25,.55,title,24,INK,True)
    D.text(5.15,y+.03,7.3,.5,detail,19,GRAY)

# 02 / 直接用数据例子说明研究问题。
slide('总体均衡，仍可能存在类别偏差',PAPER_SOURCE)
D.text(.75,1.95,5.6,.6,'A：每种表情都有男女样本',22,BLUE,True)
D.text(7,1.95,5.6,.6,'B：性别与表情绑定出现',22,RED,True)
D.table(['样本数','高兴','愤怒'],[['女性',25,25],['男性',25,25]],x=.75,y=2.7,widths=[2.0,1.6,1.6],rowh=.72,fs=21)
D.table(['样本数','高兴','愤怒'],[['女性',50,0],['男性',0,50]],x=7,y=2.7,widths=[2.0,1.6,1.6],rowh=.72,fs=21)
D.text(.8,5.3,11.7,.65,'两套数据都是：100张图片，男女各半，表情各半。',23,INK,True)
D.text(.8,6.1,11.7,.55,'但B可能让模型借助性别去猜表情。',22,RED)

# 03 / 论文贡献，先建立完整脉络。
slide('论文做了三件事',PAPER_SOURCE)
for y,num,title,body in [
    (2.0,'01','整理指标','把17类已有指标按测量对象分类。'),
    (3.5,'02','比较指标','在20个表情数据集上比较，推荐较精简的组合。'),
    (5.0,'03','检验下游影响','构造不同偏差的训练集，观察ResNet50的分组表现。')]:
    D.text(.8,y,.85,.8,num,32,ORANGE,True)
    D.text(1.95,y,2.8,.65,title,25,INK,True)
    D.text(5.0,y+.03,7.3,.9,body,22)

# 04 / 四个推荐指标的定义和解释。
slide('四个推荐指标：定义与含义',PAPER_SOURCE+'  ·  Sec.III')
for x,y,w,h in [(.75,1.95,5.6,1.82),(6.85,1.95,5.65,1.82),(.75,4.0,5.6,2.2),(6.85,4.0,5.65,2.2)]:
    D.rect(x,y,w,h,WHITE)
D.text(1,2.08,5.05,.48,'ENS：多样性／有效群体数',22,BLUE,True)
D.formula(r'$H=-\sum_g p_g\ln p_g,\qquad \mathrm{ENS}=e^H$',1,2.72,5.05,.48,24,INK)
D.text(1,3.30,5.05,.36,'男女各半时ENS=2，仅出现一组时ENS=1。',14,GRAY)
D.text(7.1,2.08,5.05,.48,'SEI：已出现群体的均匀程度',22,BLUE,True)
D.formula(r'$\mathrm{SEI}=\frac{H}{\ln R}$',7.1,2.65,5.05,.65,26,INK)
D.text(7.1,3.30,5.05,.36,'R>1时，比例越均匀越接近1。',14,GRAY)
D.text(1,4.13,5.05,.48,'V：人口属性与表情的整体关联',21,ORANGE,True)
D.formula(r'$V=\sqrt{\frac{\chi^2/n}{\min(r-1,c-1)}}$',1,4.72,5.05,.68,26,INK)
D.text(1,5.56,5.05,.42,'χ²为独立性统计量；V=0表示统计独立。',14,GRAY)
D.text(7.1,4.13,5.05,.48,'Z：某个群体×表情的超额或不足',20,ORANGE,True)
D.formula(r'$\delta=p_{gy}-p_gp_y,\quad Z_{gy}=\delta/d_{gy}$',7.1,4.68,5.05,.40,19,INK)
D.formula(r'$d_{gy}=\min(p_g,p_y)-p_gp_y\quad(\delta\geq0)$',7.1,5.18,5.05,.32,16,INK)
D.formula(r'$d_{gy}=p_gp_y-\max(0,p_g+p_y-1)\quad(\delta<0)$',7.1,5.66,5.05,.32,16,INK)
D.text(.8,6.42,11.7,.42,'p为样本比例，R为已出现群体数；n为样本量，r、c为计数表行列数。Z>0为超额，Z<0为不足。',15,GRAY)

# 05 / 作者的案例研究流程。
slide('作者怎样分析20个表情数据集',PAPER_SOURCE+'  ·  Sec.IV')
for x,title,body in [(0.8,'人脸图片','实验室 / 影视 / 互联网'),(5.0,'FairFace','预测年龄、种族、性别'),(9.15,'计数表','人口属性 × 表情')]:
    D.rect(x,2.35,3.35,1.55,WHITE)
    D.text(x+.15,2.55,3.05,.6,title,25,BLUE,True)
    D.text(x+.15,3.24,3.05,.55,body,16)
D.text(4.3,2.8,.6,.6,'→',27,GRAY);D.text(8.45,2.8,.6,.6,'→',27,GRAY)
D.text(.85,4.55,11.6,.65,'由计数表计算偏差，比较不同数据集、不同指标。',25,INK,True)
D.text(.85,5.65,11.6,.9,'人口属性是模型预测值；标签误差也会影响分析结果。',21,GRAY)

# 06 / 作者的核心结论之一。
slide('很多指标提供了相近的信息',PAPER_SOURCE+'  ·  Fig.3 / Fig.5')
D.table(['比较对象','论文中的观察','选择建议'],[
    ['ENS 与熵H','排序相关为1','多样性优先报告ENS'],
    ['SEI 与1−NSD','排序高度相关','用SEI描述均匀程度'],
    ['全局关联指标','对数据集的排序相近','优先使用易解释的V']],
    y=2.1,widths=[3.6,4.3,4.1],rowh=.85,fs=20)
D.text(.85,5.82,11.6,.85,'作者推荐：ENS、SEI、V、Z，分别保留不同角度的信息。',22,BLUE,True)

# 07 / 论文中的训练实验。
slide('用四种设置观察偏差怎样影响模型',PAPER_SOURCE+'  ·  Appendix A')
D.table(['实验','训练数据怎样变化'],[
    ['1  整体比例','所有表情中的男女比例一起变化'],
    ['2  单类偏差','只改变高兴类的男女比例'],
    ['3  两类反向','高兴偏向女性时，愤怒偏向男性'],
    ['4  两类同向','高兴与愤怒同时偏向同一群体']],
    y=1.95,widths=[3.1,8.9],rowh=.73,fs=20)
D.text(.85,6.0,11.6,.65,'11个比例点 × 3次重复；预训练ResNet50，每次训练20轮。',22,BLUE,True)

# 08 / 原文的主要发现。
slide('模型受影响的方式，取决于偏差发生在哪里',PAPER_SOURCE+'  ·  Appendix A')
for y,title,body in [(2.0,'整体比例变化','原文中，模型对部分整体性别比例变化仍能泛化。'),
                     (3.5,'类别内部缺失','某群体在特定表情中缺失时，该类召回率可能明显下降。'),
                     (5.0,'类别之间的差异','高兴、愤怒的敏感程度不同；多类偏差也可能影响其他类。')]:
    D.text(.85,y,11.6,.6,title,25,BLUE,True)
    D.text(.85,y+.67,11.6,.7,body,21)

# 09 / 从论文自然过渡到本组工作。
slide('我们复现了统计结果，也实际训练了模型','本组实验记录 · Notebook 01 / 02')
D.rect(.75,2.05,5.6,3.45,WHITE);D.rect(6.85,2.05,5.7,3.45,WHITE)
big_num(1,2.25,'20','公开数据集的统计分析',BLUE,5)
D.text(1,4.25,5,1.0,'17类指标；794,166条记录\n60套局部计数与关联矩阵',20)
big_num(7.1,2.25,'132','次真实人脸训练',ORANGE,5)
D.text(7.1,4.25,5,1.0,'4种设置 × 11个比例点 × 3次\n每个模型训练20轮',20)
D.text(.85,6.05,11.6,.65,'验证路径：重算指标并对照原文 → 控制训练比例、固定测试集 → 比较分组召回率。',18)

# 10 / 真实数据样例与使用规模。
slide('从Kaggle下载压缩数据集，开展图像实验','数据来源：fatihkgg/affectnet-yolo-format，第2版 · 图片为本次测试集样例')
D.text(.8,2.10,3.15,.5,'原始图像包',21,GRAY,True)
D.text(4.1,2.10,3.15,.5,'本次压缩包',21,BLUE,True)
D.text(.8,2.95,3.15,.95,'约122GB',36,GRAY,True)
D.text(4.1,2.95,3.15,.95,'259MB',36,BLUE,True)
D.text(.8,4.12,3.15,.46,'下载与存储成本较高',16,GRAY)
D.text(4.1,4.12,3.15,.46,'25,262张 · 96×96像素',15,GRAY)
D.rect(7.55,1.95,4.95,3.10,WHITE)
D.text(7.80,2.08,4.40,.43,'本次使用的数据样例',18,INK,True)
for x,name,label in [(7.90,'ffhq_103.png','标签：高兴'),(10.10,'ffhq_1196.png','标签：愤怒')]:
    D.image(ROOT/'presentation/assets'/name,x,2.70,1.65,1.65)
    D.text(x-.15,4.55,1.95,.38,label,16,GRAY,align='center')
D.line(.8,5.20,12.4,5.20,'#CFD8DC',1)
D.text(.8,5.48,11.6,.55,'处理后：训练14,903张  /  验证4,764张  /  测试2,421张',21,INK,True)
D.text(.8,6.28,11.65,.45,'采用下载包的划分并做文件去重；分辨率、划分与可采样规模和原文不同。',16,GRAY)

# 11 / 统计复现的证据，不讲文件、包和调试过程。
slide('指标计算与主要统计结果能够重现','本组统计复现 · Notebook 01')
for x,num,label in [(.8,'1,280','项作者实现对照通过'),(5.05,'45 / 45','Fig.3印刷值吻合'),(9.0,'21 / 21','Fig.5印刷值吻合')]:
    big_num(x,2.1,num,label,BLUE,3.55)
D.line(.85,4.1,12.4,4.1,'#CCD7DC',1)
D.text(.85,4.5,11.6,.7,'按论文公式计算，并调用作者开源实现逐项核对。',24,INK,True)
D.text(.85,5.5,11.6,1.05,'少数局部相关汇总受聚合口径影响；具体差异保留在Notebook中。',20,GRAY)

# 12 / 真实训练的具体协议与评价公式。
slide('怎样评价132次真实训练','本组图像实验 · Notebook 02')
D.table(['保持一致的设置','实际执行'],[
    ['模型与训练轮数','预训练ResNet50；20轮；max_lr=1e−4'],
    ['采样与重复','11个比例点；每点3个随机种子'],
    ['测试方式','同一测试集；只评价最后一轮']],y=1.85,widths=[4,8],rowh=.64,fs=20)
D.text(.85,4.60,11.6,.42,'本组采用AdamW、batch=256；原文未明确全部优化细节。',16,GRAY)
D.formula(r'$\mathrm{Recall}_{g,y}=\frac{TP_{g,y}}{N_{g,y}}$',.85,5.18,5.3,.67,28,BLUE)
D.formula(r'$\mathrm{MacroRecall}_{g}=\frac{1}{7}\sum_{y=1}^{7}\mathrm{Recall}_{g,y}$',6.65,5.18,5.85,.67,27,BLUE)
D.text(.85,6.25,11.6,.45,'g为性别组，y为表情类别；N为该组该类测试样本数。分类召回率与七类宏平均同时报告。',16,GRAY)

# 13 / 只重绘导出的汇总，不在本机运行实验。
from matplotlib import font_manager
font_manager.fontManager.addfont(font_path())
font=FontProperties(fname=font_path())
plt.rcParams.update({'font.family':font.get_name(),'font.size':18,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(14,5.0),sharey=True)
for exp,ax,title in [(1,axes[0],'整体改变男女比例'),(2,axes[1],'只改变高兴类男女比例')]:
    for gender,name,c in [('Female','女性',BLUE),('Male','男性',ORANGE)]:
        t=summary[(summary.experiment==exp)&(summary.gender==gender)&(summary.label=='happy')].sort_values('fraction')
        x=t.fraction.to_numpy();y=t['mean'].to_numpy();sd=t['std'].to_numpy()
        ax.plot(x,y,color=c,label=name,linewidth=2.8,marker='o',markersize=5)
        ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),color=c,alpha=.12)
    ax.set_title(title,fontproperties=font,fontsize=23,pad=15)
    ax.set_xticks([0,.5,1]);ax.set_ylim(0,1);ax.set_yticks([0,.5,1])
    ax.set_xlabel('受控类别中的女性比例',fontproperties=font,fontsize=19,labelpad=9)
    ax.legend(prop=FontProperties(fname=font_path(),size=20),frameon=False,loc='lower center',ncol=2)
axes[0].set_ylabel('高兴类召回率',fontproperties=font,fontsize=20)
fig.tight_layout();curve=ROOT/'presentation/class_recall_comparison.png';fig.savefig(curve,dpi=180,bbox_inches='tight',facecolor=PAPER);plt.close(fig)
slide('类别内部的样本缺失，会影响对应召回率','本组实验1、2 · 三次重复均值；阴影为种子间标准差')
D.image(curve,.7,1.85,12,4.5)
D.text(.85,6.35,11.6,.5,'高兴类中的群体缺失，比单独观察总体男女比例更值得留意。',18,BLUE,True)

# 14 / 保留具体对照，避免用“都下降”替代定量结果。
slide('主要下降趋势一致，具体降幅有差别','原文附录A第18、20页；本组为三个重复均值')
D.table(['设置','原文AffectNet','本组压缩数据'],[
    ['实验2：女性高兴\n均衡 → 该组缺失','>85% → <40%',f'{hb:.1%} → {hm:.1%}'],
    ['实验3：女性愤怒\n均衡 → 该组缺失','约40% → 约10%',f'{ab:.1%} → {am:.1%}']],
    y=2.0,widths=[4.5,3.75,3.75],rowh=.97,fs=20)
D.text(.85,5.3,11.6,.65,'两项对照都出现下降；实验3降幅接近，实验2下降较温和。',22,BLUE,True)
D.text(.85,6.15,11.6,.6,'支持论文的定性描述。原文为文字近似值，数据与实验条件并不完全相同。',17,GRAY)

# 15 / 一个直观的局部与平均对比。
slide('平均分会掩盖一部分群体损失','实验2 · 女性高兴训练样本从均衡变为缺失')
for y,label,before,after,c in [(2.25,'高兴类召回率',hb,hm,RED),(4.4,'七类宏平均',mb,mm,BLUE)]:
    D.text(.85,y,4.1,.65,label,25,INK,True)
    D.text(5.1,y,6.6,.85,f'{before:.2%}  →  {after:.2%}',34,c,True)
    D.rect(.85,y+1.0,10.2,.18,LIGHT)
    D.rect(.85,y+1.0,max(.05,(before-after)/.1*10.2),.18,c)
D.text(.85,6.4,11.6,.4,'两条色带按同一尺度表示下降量；只看平均分，容易低估具体类别的变化。',15,GRAY)

# 16 / 研究上的改进空间。
slide('论文方法的不足：覆盖和标签质量仍需检查','方法分析 · Notebook 03')
D.text(.8,2.0,5.4,.65,'边缘均衡，交叉群体仍可能缺失',21,BLUE,True)
D.table(['人数','年轻','年长'],[['女性',50,0],['男性',0,50]],
    x=.8,y=3.0,widths=[1.4,1.8,1.8],rowh=.75,fs=19)
D.text(.85,5.65,5.25,.65,'预设4种组合，只观察到2种。',20)
D.text(6.75,2.0,5.6,.6,'人口标签来自模型预测',23,INK,True)
D.text(6.75,2.8,5.6,1.0,'预测误差会影响指标；\n缺少主体ID，难以排查主体泄漏。',20)
D.text(6.75,4.35,5.6,.6,'年龄被当作离散类别',23,INK,True)
D.text(6.75,5.18,5.6,1.0,'名义变量指标不区分年龄组的远近，\n无法利用年龄的顺序信息。',19)
D.text(.85,6.4,11.6,.45,'覆盖不足先补数据；有样本但比例失衡时，再探索重加权。',17,BLUE)

# 17 / 权重估计、截断、类内归一与训练目标。
slide('重加权尝试：估计权重，再限制与归一','配对对照 · 同子集、同初始化种子、同测试集')
D.text(.85,1.95,11.6,.55,'设置：高兴类中，女性约占10%，男性约占90%。',23,INK,True)
D.rect(.85,2.68,1.1,.30,BLUE);D.rect(1.95,2.68,9.9,.30,ORANGE)
D.rect(.75,3.25,5.65,2.45,WHITE);D.rect(6.8,3.25,5.7,2.45,WHITE)
D.text(1,3.40,5.05,.48,'1. 按群体×类别估计并截断',20,BLUE,True)
D.formula(r'$\widetilde{w}_{gy}=\min\!\left(\frac{p(y)}{Kp(g,y)},5\right)$',1,4.08,5.05,.65,24,INK)
D.text(1,5.02,5.05,.43,'最终权重：女性高兴约5，男性约0.56。',14,GRAY)
D.text(7.05,3.40,5.1,.48,'2. 类内归一，保持类别总权重',19,BLUE,True)
D.formula(r'$w_{gy}=\frac{\widetilde{w}_{gy}n_y}{\sum_{g^\prime}n_{g^\prime y}\widetilde{w}_{g^\prime y}}$',7.05,4.08,5.1,.65,24,INK)
D.text(7.05,5.02,5.1,.43,'K=2；n为训练计数。缺失单元无法靠权重补齐。',13,GRAY)
D.formula(r'$\mathcal{L}=\frac{1}{N}\sum_i w_{g_i y_i}\,\ell(f_\theta(x_i),y_i)$',.85,5.98,11.6,.62,25,BLUE)
D.text(.85,6.70,11.6,.27,'额外训练3个模型，每个20轮。截断值在测试前固定；归一化后最大权重可略高于5。',12,GRAY)

# 18 / 如实展示改善和代价。
slide('女性组平均表现提高，男性组略有下降','本组重加权实验 · 三个配对重复的均值')
rows=[]
for gender,cn,label,name in [('Female','女性','happy','高兴'),('Male','男性','happy','高兴'),('Female','女性','mean','宏平均'),('Male','男性','mean','宏平均')]:
    b=weighted('ERM',gender,label);a=weighted('Reweighted',gender,label)
    rows.append([f'{cn} / {name}',f'{b:.1%}',f'{a:.1%}',f'{(a-b)*100:+.2f}个百分点'])
D.table(['指标','普通训练','重加权','变化'],rows,y=1.95,widths=[3.7,2.3,2.3,3.7],rowh=.73,fs=20)
D.text(.85,6.0,11.6,.75,'有效样本量约保留80%。有收益，也有群体间取舍，不能只挑改善项。',20,BLUE,True)

# 19 / 复用已经连接的开发容器，展示已完成的代码和输出。
slide('代码演示：三份实验Notebook','代码与已有结果展示')
D.table(['编号','Notebook内容','主要验证工作'],[
    ['01','二十个真实数据集完整指标复现','17类指标、作者对照与局部矩阵'],
    ['02','真实人脸表情识别模型训练验证','132次训练、召回率与配对对照'],
    ['03','不足分析与改进探索','交叉覆盖、标签敏感性与重加权']],
    y=2.0,widths=[1.1,5.5,5.4],rowh=.85,fs=18)
D.text(.85,5.72,11.6,.6,'VS Code已连接 bias-repro-gpu，直接打开 /workspace/notebooks。',19,BLUE,True)
D.text(.85,6.4,11.6,.45,'代码与注释、已完成的训练日志、分组召回率和真实模型曲线。',17,GRAY)

# 20 / 直接归纳本次研究结果。
slide('本次复现说明了什么','汇报小结')
for y,num,s in [(2.0,'1','论文的指标计算和主要统计分析可以重现。'),
                 (3.45,'2','真实训练支持论文的主要定性描述，具体降幅仍有差别。'),
                 (4.9,'3','重加权改变了群体表现，需要同时报告收益和代价。')]:
    D.text(.85,y,.7,.8,num,34,ORANGE,True)
    D.text(1.85,y+.03,10.5,1.0,s,25,INK,True)
D.text(.85,6.35,11.6,.5,'下面留两个问题，分别讨论数据偏差与模型表现。',18,GRAY)

# 21–24 / 只保留两个问题，每个问题与答案分开。
slide('问题一','讲后讨论 · 1 / 2',dark=True)
D.text(.85,2.6,11.6,1.8,'男性和女性各占50%，\n能说明数据集没有人口统计偏差吗？',32,WHITE,True)
D.text(.85,5.9,11.6,.65,'请结合表征偏差与刻板偏差讨论。',21,'#BFCFD6')
slide('答案一：总体比例均衡，还要看类别内组成','讲后讨论 · 1 / 2')
D.table(['样本数','高兴','愤怒'],[['女性',50,0],['男性',0,50]],x=.85,y=2.3,widths=[1.8,1.8,1.8],rowh=.82,fs=22)
D.text(7,2.3,5.3,1.25,'男女总数相同，\n性别与表情仍然完全关联。',25,BLUE,True)
D.text(7,4.15,5.3,1.1,'还要检查其他人口属性\n以及交叉群体是否缺失。',21)
D.text(.85,6.15,11.6,.6,'ENS / SEI 描述覆盖和比例；V / Z 描述与标签的关联。',20)
slide('问题二','讲后讨论 · 2 / 2',dark=True)
D.text(.85,2.4,11.6,2.1,'把重加权后的V降到0，\n就能保证模型对每个群体都更好吗？',31,WHITE,True)
D.text(.85,5.9,11.6,.65,'请区分数据分布与模型测试表现。',21,'#BFCFD6')
slide('答案二：数据分布改变，不保证每个群体都改善','讲后讨论 · 2 / 2')
D.text(.85,2.0,11.6,.65,'V描述人口属性与标签的统计关联，不直接约束模型错误率。',23,INK,True)
D.rect(.85,3.1,11.6,1.25,WHITE)
D.text(1.1,3.38,5.3,.7,'女性高兴：82.1% → 83.1%',24,BLUE,True)
D.text(6.9,3.38,5.2,.7,'男性高兴：88.5% → 86.3%',24,ORANGE,True)
D.text(.85,4.95,11.6,1.1,'判断改进效果，还要看：独立测试集、各群体各类别表现、\n整体准确率、群体覆盖和有效样本量。',22)
D.text(.85,6.4,11.6,.45,'进一步讨论因果公平时，还需要额外的因果假设。',16,GRAY)

notes=D.save()
(ROOT/'presentation/slide_plan.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2))
lines=['# PPT逐页演讲稿','',
       '24页：第1–8页介绍论文，第9–20页介绍本组复现、结果及补充实验。主讲25分钟；第22–25页为两个问题及各自答案，共5分钟。',
       '', '【】为操作或停顿提示，不必读出。','']
for n in notes:lines.extend([f'## 第{n["slide"]}页 · {n["title"]}','',n['script'],''])
(ROOT/'presentation/演示讲稿.md').write_text('\n'.join(lines))
runpy.run_path(str(ROOT/'presentation/export_speech.py'))
with zipfile.ZipFile(ROOT/'dist/PPT与演讲稿_两个问题版.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['presentation/论文复现汇报.pptx','presentation/论文复现汇报.pdf','presentation/PPT演讲稿.docx','presentation/演示讲稿.md','presentation/答辩问答.md']:
        z.write(ROOT/name,name)
print('本机已生成：PPT、PDF、Word讲稿；21页主讲25分钟，两个问答5分钟。')
