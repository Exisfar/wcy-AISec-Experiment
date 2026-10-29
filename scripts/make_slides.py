"""从实际结果生成可编辑中文 PPT；讲稿与计时表同步导出。"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiments import load_data,association,COMPONENTS,UNIVERSE,LAB,MOTION
import matplotlib.pyplot as plt
import seaborn as sns
figdir=ROOT/'results/figures'
# 专门为投影制作更简洁的结果图。
data=load_data();df=data['ferplus'];t=pd.crosstab(df.gender,df.label);_,_,z=association(t)
plt.figure(figsize=(10,2.8));sns.heatmap(pd.DataFrame(z,index=t.index,columns=t.columns),annot=True,fmt='.2f',cmap='vlag',vmin=-1,vmax=1,cbar=False,annot_kws={'size':16})
plt.title('FER+ | Gender x expression | Ducher Z',fontsize=18);plt.ylabel('');plt.xlabel('');plt.tight_layout();plt.savefig(figdir/'slide_ferplus_gender.png',dpi=180,bbox_inches='tight');plt.close()
rep=pd.read_csv(ROOT/'results/tables/representation.csv');glob=pd.read_csv(ROOT/'results/tables/global.csv')
a=[]
for n,d in data.items():
 for c in COMPONENTS:
  a.append({'dataset':n,'component':c,'collection':'LAB' if n in LAB else 'ITW-M' if n in MOTION else 'ITW-I',
            'ENS_over_K':float(rep.query('dataset==@n and component==@c').ENS.iloc[0])/UNIVERSE[c]})
plt.figure(figsize=(10,4));sns.barplot(data=pd.DataFrame(a),x='component',y='ENS_over_K',hue='collection',errorbar=None,hue_order=['LAB','ITW-M','ITW-I']);plt.ylim(0,1);plt.ylabel('Mean ENS / fixed K');plt.xlabel('');plt.title('Observed diversity by source (dataset-level mean)');plt.tight_layout();plt.savefig(figdir/'slide_sources.png',dpi=180);plt.close()

raw=pd.read_csv(ROOT/'results/tables/synthetic_raw.csv');syn=raw.groupby(['shift','method'])[['accuracy','worst_group_accuracy','counterfactual_flip']].mean()
rw=pd.read_csv(ROOT/'results/tables/reweighting.csv');sub=rw.query("dataset=='ferplus' and component=='gender' and cap==0").iloc[0]
oldv=float(glob.query("dataset=='ferplus' and component=='gender'").V.iloc[0])
checks=pd.read_csv(ROOT/'results/tables/upstream_crosscheck.csv')
loc=pd.read_csv(ROOT/'results/tables/local_agreement.csv')
rc=pd.read_csv(ROOT/'results/tables/agreement_representation.csv',index_col=0)
pr=json.loads((ROOT/'results/provenance.json').read_text())

NAVY='#142B43';TEAL='#087F8C';ORANGE='#D98532';LIGHT='#F3F6F8';INK='#183047';MUTED='#627487';WHITE='#FFFFFF'
def rgb(s):return RGBColor.from_string(s.strip('#'))
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
notes=[];elapsed=0

def rect(sl,x,y,w,h,fill,line=None,radius=False):
    sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill);sh.line.fill.background() if line is None else None
    if line:sh.line.color.rgb=rgb(line)
    return sh

def text(sl,x,y,w,h,s,size=22,color=INK,bold=False,align=None):
    box=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame
    tf.word_wrap=True;tf.margin_left=Inches(.02);tf.margin_right=Inches(.02);tf.margin_top=Inches(.02);tf.margin_bottom=0
    for i,line in enumerate(s.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
        p.font.name='Noto Sans CJK SC';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color)
        p.space_after=Pt(8)
        if align is not None:p.alignment=align
    return box

def base(title,kicker='论文复现',takeaway=None,dark=False):
    sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=rgb(NAVY if dark else WHITE)
    rect(sl,0,0,.16,7.5,TEAL)
    text(sl,.55,.28,12,.3,kicker,12,TEAL if not dark else '#69D3D0',True)
    text(sl,.55,.85,12,1.0,title,30,WHITE if dark else NAVY,True)
    if takeaway:
        rect(sl,.55,6.52,12.18,.48,'#EAF4F3' if not dark else '#204057',radius=True)
        text(sl,.73,6.59,11.8,.3,takeaway,14,TEAL if not dark else WHITE)
    text(sl,.6,7.16,11,.2,'arXiv:2303.15889v2  ·  原论文统计复现 + 本组扩展  ·  2026.10',9,MUTED if not dark else '#B4C5D6')
    text(sl,12.1,7.12,.6,.25,f'{len(prs.slides):02d}',11,MUTED)
    return sl

def note(sl,title,seconds,script,backup=False):
    global elapsed
    start=elapsed
    if not backup:elapsed+=seconds
    schedule=f'{start//60:02}:{start%60:02}–{elapsed//60:02}:{elapsed%60:02}' if not backup else '问答备用'
    msg=f'【{schedule}，建议 {seconds} 秒】\n{script}'
    sl.notes_slide.notes_text_frame.text=msg
    notes.append({'slide':len(prs.slides),'title':title,'seconds':seconds,'schedule':schedule,'backup':backup,'script':script})

def bullets(title,items,takeaway,seconds,script,kicker='论文复现',backup=False):
    sl=base(title,kicker,takeaway)
    gap=4.35/len(items)
    for i,(heading,body) in enumerate(items):
        y=2.0+i*gap
        rect(sl,.65,y+.06,.08,.5,TEAL)
        text(sl,.95,y,11.5,.45,heading,23,INK,True)
        text(sl,.95,y+.5,11.45,gap-.46,body,18,MUTED)
    note(sl,title,seconds,script,backup);return sl

def picture(sl,path,x,y,w,h):
    im=Image.open(path);a=im.width/im.height
    if w/h>a:ww=h*a;hh=h
    else:ww=w;hh=w/a
    sl.shapes.add_picture(str(path),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))

def chart(title,path,aside,takeaway,seconds,script,kicker='复现实验'):
    sl=base(title,kicker,takeaway);picture(sl,figdir/path,.55,1.8,8.65,4.5)
    rect(sl,9.42,1.95,3.3,4.16,LIGHT,radius=True);text(sl,9.65,2.15,2.83,3.8,aside,19)
    note(sl,title,seconds,script);return sl

def table(sl,headers,rows,x=.65,y=2,w=12.0,rowh=.6,widths=None,fs=18):
    widths=widths or [w/len(headers)]*len(headers)
    for i,row in enumerate([headers]+rows):
        xx=x
        for j,val in enumerate(row):
            rect(sl,xx,y+i*rowh,widths[j]-.025,rowh-.025,NAVY if i==0 else LIGHT if i%2 else '#E8EFF2')
            text(sl,xx+.12,y+i*rowh+.11,widths[j]-.24,rowh-.1,str(val),fs,WHITE if i==0 else INK,bold=i==0)
            xx+=widths[j]

# 20 页主汇报，计时精确 25 分钟。
sl=base('数据偏差，怎样量化？\n从论文复现到缓解探索','课程论文复现 · 25 分钟汇报',dark=True)
text(sl,.65,3.2,11.7,1.1,'Metrics for Dataset Demographic Bias\nA Case Study on Facial Expression Recognition',23,WHITE)
text(sl,.65,4.75,11.7,.7,'20 个真实公开数据集  /  17 类指标  /  容器内离线验证',23,'#69D3D0',True)
note(sl,'封面',30,'今天讨论两个问题：如何量化数据集中的人口统计偏差，以及量化之后能够做什么。我们从原论文的公开数据出发，先验证，再对不足做有限但实际可运行的改进探索。')

bullets('先核对论文身份：以 PDF 正文为准',[
('正文实际研究的是数据集人口偏差','Dominguez-Catena 等；TPAMI 2024；arXiv:2303.15889v2。'),
('文件名与链接都存在容易串题之处','文件名写 Counterfactual Fairness；连字符仓库对应另一篇 2022 年论文。'),
('正确仓库包含完整匿名统计数据','Dataset_Bias_Metrics（下划线）；固定 commit，保留作者原码与许可证。')],
'汇报对象是 PDF 实际正文；反事实实验属于本组扩展。',50,
'打开 PDF 第一页和仓库 README 可以现场核对。不要按文件名把这篇论文讲成一篇因果反事实公平算法论文。作者名字相同、仓库名字只差连接符，是本次复现最容易踩的坑。')

sl=base('验证范围：把做到的和未做到的都讲清楚',takeaway='完整复现公开元数据上的统计分析；未重跑原图标注与 ResNet50。')
table(sl,['论文部分','本次交付','验证状态'],[
['Sec.III 指标体系','17 类指标，U 双方向','独立实现 + 作者数值对照'],
['Sec.V / Fig.2–8','20 数据集、3 人口组件','全量重算与图表导出'],
['Appendix B','60 套 count / NPMI / Z','完整矩阵与 20 幅总览'],
['Appendix A','四种构造、11 个比例点','仅列联表与 (a) 指标曲线'],
['Sec.IV / A 下游模型','原图、FairFace、ResNet50','未重跑；无图像及主体信息']],rowh=.64,widths=[3.2,4.5,4.3],fs=17)
note(sl,'复现范围',60,'这里主动说明边界。公开 CSV 让我们能验证正文核心，但不能代替论文从图像开始的端到端流程。附录的 recall 曲线依赖 AffectNet 图像和网络训练，本次没有把合成实验贴上原论文标签。')

bullets('两种偏差，需要两种观察角度',[
('表征偏差：哪些群体缺失或不足？','只看人口属性边缘分布 p(g)，无需表情标签。'),
('刻板偏差：某些群体是否集中在某些表情？','看人口属性与表情标签的联合分布 p(g,y)，并与 p(g)p(y) 比较。'),
('边缘均衡，不保证联合独立','男女总体各一半，仍可能“女性多 happy、男性多 angry”。')],
'人口比例、统计依赖、模型公平性是三个不同问题。',60,
'用四格表口头举例：男女各五十人，但女性全是高兴、男性全是愤怒，总体表征均衡却具有极强刻板偏差。之后所有指标都围绕边缘分布和联合分布展开。')

sl=base('17 类指标：用少量可解释指标覆盖主要信息',takeaway='论文推荐：ENS + SEI、全局 V、局部 Z。')
table(sl,['类别','指标族','关键解释'],[
['表征：10 类','R, ENS, H, D, 1/D, 1−D, SEI, NSD, IR, BP','ENS = exp(H)，等效均衡组数'],
['全局：5 类','Cramér V, T, C, Theil U, NMI','U 计算双方向；衡量统计依赖'],
['局部：2 类','NPMI, Ducher Z','逐个群体×表情单元定位不足/超额']],rowh=.86,widths=[2.5,5.5,4],fs=16)
text(sl,.8,5.65,11.7,.5,'ENS = exp[−Σ p(g) ln p(g)]      SEI = H / ln R      V² = (χ²/n) / min(r−1,c−1)',19,TEAL,True)
note(sl,'指标体系',90,'不逐一推导十七个公式。解释三个支点：ENS 把熵换算成等效组数；SEI 关注已出现组的均匀程度；V 量化联合分布偏离独立分布的程度。Z 用来定位具体单元。所有完整公式与边界约定在 Notebook 和源码里可查。')

bullets('验证设计：避免“跑出一张图就算复现”',[
('第一层：可手算反例与边界测试','独立、完全关联、单组退化、零联合概率、尺度不变性。'),
('第二层：独立公式与固定作者原码交叉核验',f'{len(checks):,} 项标量/矩阵比较全部通过；不改作者源码。'),
('第三层：直接对照论文印刷结果','Fig.3 / Fig.5 逐项核验；保留舍入容差和聚合约定。')],
'作者代码一致，只是证据之一；还要核对公式与论文数值。',60,
'交叉验证使用两条实现路径，不能只调用作者函数再断言其返回自身。局部矩阵每个单元参与 allclose，而不是只比较均值。随机实验显式给出种子。')

sl=base('数据与环境：拿到容器即可离线重跑',takeaway='CUDA 基镜像中运行 CPU 统计任务；验证不依赖 GPU。')
for i,(num,label) in enumerate([('20','真实匿名数据集'),(f'{pr["total_rows"]:,}','完整有效记录'),('3','中文注释 Notebook')]):
    x=.7+i*4.15;rect(sl,x,2.05,3.88,1.6,LIGHT,radius=True);text(sl,x+.22,2.25,3.4,.7,num,36,TEAL,True);text(sl,x+.22,3.0,3.4,.4,label,18)
text(sl,.8,4.15,11.8,1.7,'nvidia/cuda:12.5.1-base-ubuntu20.04\nuv 创建 /workspace/.venv · Python 3.11.15 · uv.lock 固定全部依赖\n数据、Notebook、HTML、图表、PPT 与讲稿全部打包',22)
note(sl,'数据环境',60,'二十份 CSV 来自作者仓库，不是合成数据。人口标签是 FairFace 预测的表观属性。原图不在包内，所以镜像体积与所需算力可控。CPU 即可验证统计结果，老师不需要配置显卡驱动。')

chart('表征指标：一致性高，但 R 与 IR 是例外','fig03_agreement.png',
      f'ENS 与 H\nρ = 1.000\n\nSEI 与 1−NSD\nρ = {rc.loc["SEI","1-NSD"]:.3f}\n\n同向相关，不等于相同含义。',
      '复现 Fig.3：先在四组件内相关，再平均；退化相关填 0 的约定另有敏感性说明。',90,
      '指图中两类：组合多样性与均匀性。ENS 和 H 数学上单调等价，相关一并不奇怪。R 只记出现了几个组；IR 只看极端计数，都会损失大量信息。解释为何建议用 ENS 与 SEI 配合。')

chart('全局刻板偏差：六种读数的排序高度一致','fig05_agreement.png',
      '5 类指标\nU 取双方向\n\nFig.5 上三角\n21 / 21 印刷项吻合\n\n按熵分母明确 U 的方向。',
      '论文 Eq.(13) 与源码方向不同；Ux=I/H(G)，Uy=I/H(Y)。',90,
      '告诉听众 Ux、Uy 的定义，而不单靠箭头名。按正文分母映射后，图五全部印刷项吻合。高排序一致支持用一个指标做简洁比较，但数值大小不能直接当作同一个量。')

chart('局部偏差：Z 指出具体哪些组合失衡','slide_ferplus_gender.png',
      '红色：超额出现\n蓝色：不足出现\n\n这张表描述训练数据，不能代替模型错误率。\n\n其他 20 数据集完整矩阵均随包。',
      '局部相关高；完全独立表中的浮点微差会制造伪排序，需要并列处理。',90,
      f'先让听众看性别与表情的每个单元，再说明正负值相对于独立分布。全套局部结果已导出。作者原始局部相关均值为 {loc.upstream_raw_rho.mean():.4f}，将近零微差视为并列后的结果略有变化，不隐瞒这类差异。')

chart('数据来源影响偏差形态，但不是因果结论','slide_sources.png',
      'LAB：实验室\nITW-M：影视\nITW-I：互联网\n\n总体观察：互联网数据的表征多样性通常更高。',
      '更高多样性不保证较低刻板偏差；群体标签仍是自动估计的代理。',60,
      '这张图用固定 K 归一化 ENS。强调按数据集等权平均而不是把大数据集吞掉小数据集。来源与偏差有关，但集合比较不能识别来源的因果作用。')

chart('附录四种偏差：先验证受控构造本身','appendixA_curves.png',
      '1 全局比例\n2 只改变 happy\n3 happy/angry 反向\n4 happy/angry 同向\n\n11 个比例点 × 4。',
      '复现 Fig.9–12(a) 指标面板；未重跑 (b)(c) 的真实模型 recall。',60,
      '横轴都是控制性别比例。全局表征偏差不一定制造群体与类别的依赖；只在某类改比例就会引入刻板偏差。单组端点 V 公式退化，保持 NaN，不能强解释为零。')

bullets('发现的问题：定义边界比“多一个指标”更关键',[
('公式与源码的差异可以用反例核验','Pearson C 的分母印成 1−χ²/n；Theil U 的分母方向与函数命名不同。'),
('观测组均衡可能掩盖群体缺失','单组 IR=1；两个边缘均匀，但交叉群体可能只出现 2/4。'),
('统计关联不提供模型或因果公平保证','自动人口标注存在误差；无主体 ID；数据偏差与模型偏差不必线性对应。')],
'区分排版错误、代码语义差异与研究方法局限，不把它们混成“论文错误”。',90,
'用完全关联二乘二表说明 Pearson 标准值 sqrt(1/2)，印刷公式却分母为零。再举交叉群体反例。第三点论文自身已讨论部分内容，我们是在此基础上定量探索，并不声称首次提出。',kicker='不足分析')

bullets('改进方案：覆盖审计 → 支持检查 → 有约束缓解',[
('固定标签空间与交叉群体覆盖率','同时报告 R/K、ENS/K、H/log K；没有样本的单元单独标记。'),
('不确定性与人口标签噪声敏感性','300 次 bootstrap；校正 V；扰动代理标签，检查结论稳定性。'),
('保留类别分布的截断重加权','目标 q(g,y)=p(y)/K；先限权重，再按类别归一；同时报告 ESS。')],
'无支持单元不能靠加权补出来；统一比例目标也需要场景依据。',90,
'解释为什么不用简单平滑伪造缺失群体。重加权是训练分布设计策略，目标选择需要具体任务依据。本次采用均衡目标作为可检验的探索设定，保留类别先验，避免通过改变类别比例掩盖问题。',kicker='本组扩展')

chart('真实元数据探索：依赖下降，代价是有效样本量','extension_reweighting.png',
      f'FER+ / gender\n原始 V = {oldv:.4f}\n未截断 V ≈ 0\n\nESS / N = {sub.ESS_fraction:.1%}\n\n结果是加权统计量，原始记录未改变。',
      'V 降到 0 是目标分布的代数结果；不能据此宣称真实 FER 模型公平。',90,
      '先说明 cap=1 也包含按类别再归一，最终权重可能超过预截断值。观察不截断时 ESS 大约百分之九十七。所有六十个数据集组件都检查了是否有支持，缺失者如实标记不可行。',kicker='本组扩展 · 真实 CSV')

chart('合成下游：缓解选择偏差，但付出同分布精度代价','extension_synthetic.png',
      f'反转环境准确率\nERM {syn.loc[("reversed","ERM"),"accuracy"]:.1%}\n重加权 {syn.loc[("reversed","Reweighted"),"accuracy"]:.1%}\n\n同分布：\nERM {syn.loc[("same","ERM"),"accuracy"]:.1%}\n重加权 {syn.loc[("same","Reweighted"),"accuracy"]:.1%}',
      '20 个随机种子；训练集估计权重；共享测试样本；明确 SCM 的 do(A) 配对实验。',90,
      '此处是合成压力测试，不是 AffectNet。生成 X 与 Y 的稳定关系，让 A 与 Y 在训练选择中高度相关，测试反转。ERM 依赖捷径而失效；重加权更稳健，但同分布准确率下降。SCM 中 X 不受 A 影响，所以固定 X 的 do(A) 才成立。',kicker='本组扩展 · 合成机制')

bullets('哪些结论仍然不能从本次实验推出？',[
('不能声称真实模型已达到反事实公平','没有原始图像、主体划分和真实因果图；合成收益不能直接外推。'),
('不能声称重加权总是优于原始训练','同分布精度可能降低；罕见单元会放大方差，缺失单元无解。'),
('下一步实验协议是明确的','主体级划分；同款 ResNet50/20 epochs/1cycle；三次重复；报告 subgroup recall。')],
'本次交付覆盖可由公开证据支持的验证，并保留进一步原图实验的清晰接口。',60,
'重申限制不是客套。我们已经真实运行能验证的统计内容，但真实图像训练尚缺条件。向老师说明若后续拿到数据，应采用主体划分并只用训练集计算权重，避免泄漏。',kicker='结果边界')

sl=base('现场演示：从一个命令到可检查结果',kicker='课堂演示 · 2 分钟',takeaway='禁网、无 GPU、无项目挂载的新容器，独立重跑三个 Notebook。')
for i,(title,cmd) in enumerate([('① 导入随包镜像','docker load -i dist/fer-bias-repro-image.tar.gz'),('② 运行完整复现','docker run --rm --network none fer-bias-repro:1.0 reproduce'),('③ 打开交互验证','bash scripts/open_lab.sh')]):
    y=1.95+i*1.35;text(sl,.8,y,11.8,.4,title,22,TEAL,True);rect(sl,.8,y+.5,11.7,.58,LIGHT,radius=True);text(sl,1,y+.61,11.3,.35,cmd,17)
note(sl,'现场演示',120,'课前先导入镜像，避免解压占用课堂时间。现场执行复现命令，等待约一分钟。展示 ALL NOTEBOOKS PASSED，打开带输出 Notebook 的两个结果：Fig.3 和重加权表。若浏览器或 Docker 现场异常，立即用随包 HTML 与 PDF 继续，不浪费汇报时间。')

bullets('对照评分要求：每个结论都有交付证据',[
('完整性与可理解性','范围矩阵、四个研究问题、带中文注释 Notebook、论文数值对照。'),
('可运行性','离线镜像、uv.lock、哈希验证、禁网新容器重跑日志。'),
('不足、方案与探索','公式审计与交叉反例；覆盖/不确定性/重加权；真实元数据与合成模型实验。')],
'原文复现、方法审计、改进探索分开呈现，所有结果可追溯。',30,
'快速对应评议维度 a 到 g。不夸大完整性，原图训练明确未做。课堂重点是让评议组理解验证思路，而不只是展示文件数量。')

sl=base('讨论与提问','主汇报结束 · 另留 5 分钟问答',dark=True)
text(sl,.8,2.4,11.8,2.6,'数据集偏差可以被可靠地量化。\n缺失群体、支持条件和因果假设，\n决定量化结果能怎样用于缓解。',28,WHITE,True)
text(sl,.8,5.65,11.7,.5,'可现场查验：三个 Notebook / 原始 CSV / 结果表 / 容器运行记录',19,'#69D3D0')
note(sl,'进入问答',30,'主汇报结束，接下来只讨论两个问题，每个问题后单独展示答案。')

# 讲后仅保留两个问题，问题和答案各占一页。
sl=base('问题一','讲后讨论 · 1 / 2',dark=True)
text(sl,.85,2.4,11.6,2.0,'男性和女性各占 50%，\n就能说明数据集没有人口统计偏差吗？',32,WHITE,True)
text(sl,.85,5.6,11.6,.5,'先讨论，再翻页查看参考答案。',20,'#69D3D0')
note(sl,'问题一：性别均衡是否代表无偏差',45,'展示问题，等待回答；本页不展示答案。',True)

sl=base('答案一：不能，只能说明性别总体比例均衡','参考答案 · 1 / 2',takeaway='边缘比例均衡 ≠ 人口属性与任务标签独立；还需检查其他属性与交叉群体。')
table(sl,['示例 / 人数','高兴','愤怒','合计'],[['女性','50','0','50'],['男性','0','50','50']],x=.7,y=2.35,w=6.2,rowh=.75,widths=[2.3,1.3,1.3,1.3],fs=20)
text(sl,7.25,2.12,5.1,.6,'表征偏差与刻板偏差要分开看',21,TEAL,True)
text(sl,7.25,2.95,5.1,2.55,'总体男女各一半，\n性别与表情仍可完全关联。\n\n年龄、种族及交叉群体，\n也可能缺失或不足。',21)
text(sl,.85,5.55,11.6,.55,'组合使用 ENS / SEI、全局 V 与局部 Z，检查覆盖、比例和联合分布。',20,TEAL)
note(sl,'答案一：区分表征偏差与刻板偏差',90,'结合列联表解释两类偏差。',True)

sl=base('问题二','讲后讨论 · 2 / 2',dark=True)
text(sl,.85,2.35,11.6,2.2,'将重加权后的 V 降为 0，\n就能证明模型实现反事实公平吗？',32,WHITE,True)
text(sl,.85,5.6,11.6,.5,'先讨论，再翻页查看参考答案。',20,'#69D3D0')
note(sl,'问题二：V 为零是否代表反事实公平',45,'展示问题，等待回答；本页不展示答案。',True)

bullets('答案二：不能，数据分布独立不等于反事实公平',[
('V 衡量数据分布中的统计依赖','V=0 不直接约束模型预测；其他特征仍可能含敏感属性的代理。'),
('反事实公平需要明确因果假设','需定义干预敏感属性时哪些变量随之改变，再检查个体的预测变化。'),
('本次扩展的结论有明确范围','合成 SCM 的假设不能直接外推；真实模型仍需保留测试集评估。')],
'重加权还受缺失单元、有效样本量和同分布精度代价约束。',120,
'区分描述性独立、模型表现与因果反事实公平。',kicker='参考答案 · 2 / 2',backup=True)

# 给关键讲解页各增加 10 秒，形成精确 25 分钟主汇报。
elapsed=0
for item in notes:
    if item['backup']: continue
    if item['slide'] in {2,4,5,6,8,9,10,13,14,16}: item['seconds']+=10
    start=elapsed;elapsed+=item['seconds']
    item['schedule']=f'{start//60:02}:{start%60:02}–{elapsed//60:02}:{elapsed%60:02}'
    prs.slides[item['slide']-1].notes_slide.notes_text_frame.text=f"【{item['schedule']}，建议 {item['seconds']} 秒】\n{item['script']}"
assert elapsed==1500,elapsed
speech=json.loads((ROOT/'scripts/speech_content.json').read_text(encoding='utf-8'))
question_elapsed=1500
for item in notes:
    item['script']=speech[str(item['slide'])]
    if item['backup']:
        start=question_elapsed;question_elapsed+=item['seconds']
        item['schedule']=f'{start//60:02}:{start%60:02}–{question_elapsed//60:02}:{question_elapsed%60:02}'
    prs.slides[item['slide']-1].notes_slide.notes_text_frame.text=f"【{item['schedule']}，建议 {item['seconds']} 秒】\n{item['script']}"
assert question_elapsed==1800
assert len(prs.slides)==24
prs.save(ROOT/'slides/论文复现汇报.pptx')
(ROOT/'slides/slide_plan.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# PPT 逐页演讲稿（可直接照读）','',
       '对应 slides/论文复现汇报.pptx，共 24 页。第 1–20 页为 25 分钟主汇报（含 2 分钟现场演示）；第 21–24 页为 5 分钟讨论，只设两个问题，均为“问题页 → 答案页”。',
       '', '正文为口语讲稿；【】中的文字为操作或停顿提示，不需读出。计时为彩排目标，可按实际语速微调。全部讲稿已同步写入 PPT 演讲者备注。', '']
for n in notes:
    lines.extend([f'## 第 {n["slide"]} 页 · {n["title"]}',f'时间：{n["schedule"]}；建议 {n["seconds"]} 秒。','',n['script'],''])
(ROOT/'docs/演示讲稿.md').write_text('\n'.join(lines),encoding='utf-8')
import runpy
runpy.run_path(str(ROOT/'scripts/export_speech.py'))
print('Saved PPTX with',len(prs.slides),'slides; main 25:00; exactly 2 Q&A pairs in 5:00')
