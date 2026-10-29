"""从同一份逐页讲稿导出 Word，并同步精简为两个问答。只使用标准库。"""
from pathlib import Path
import json,zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
notes=json.loads((ROOT/'slides/slide_plan.json').read_text())
assert len(notes)==24
paras=[]
def paragraph(text,style='Normal'):
    return '<w:p><w:pPr><w:pStyle w:val="'+style+'"/></w:pPr><w:r><w:t xml:space="preserve">'+escape(text)+'</w:t></w:r></w:p>'
paras.append(paragraph('PPT 逐页演讲稿','Title'))
paras.append(paragraph('主汇报 25 分钟 + 两个问答 5 分钟','Subtitle'))
paras.append(paragraph('对应 24 页 PPT：第 1–20 页主汇报，第 21–24 页依次为问题一、答案一、问题二、答案二。【】中的内容为操作与停顿提示，不需读出。计时为彩排目标。'))
for n in notes:
    paras.append(paragraph(f'第 {n["slide"]} 页  {n["title"]}','Heading1'))
    paras.append(paragraph(f'{n["schedule"]}  |  建议 {n["seconds"]} 秒','Subtitle'))
    for text in n['script'].split('\n'):
        if text.strip(): paras.append(paragraph(text,'Instruction' if text.startswith('【') else 'Normal'))
ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
doc=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{ns}"><w:body>'+''.join(paras)+'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body></w:document>'
styles=f'''<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="{ns}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="24"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="140" w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:color w:val="142B43"/><w:sz w:val="40"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:rPr><w:color w:val="627487"/><w:sz w:val="21"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="320" w:after="140"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="087F8C"/><w:sz w:val="29"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Instruction"><w:name w:val="Instruction"/><w:basedOn w:val="Normal"/><w:rPr><w:color w:val="087F8C"/><w:i/><w:sz w:val="22"/></w:rPr></w:style>
</w:styles>'''
ct='''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
rels='''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
drels='''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
path=ROOT/'docs/PPT演讲稿.docx'
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
    for name,contents in [('[Content_Types].xml',ct),('_rels/.rels',rels),('word/document.xml',doc),('word/styles.xml',styles),('word/_rels/document.xml.rels',drels)]:
        ET.fromstring(contents);z.writestr(name,contents)
qa='''# 讲后讨论：只保留两个问题

PPT 共 24 页：1–20 页为 25 分钟主汇报；21–24 页为 5 分钟问答。每个问题和对应答案分别占一页，先让同学回答，再翻到答案页。

## 问题一（第 21 页）

男性和女性各占 50%，就能说明数据集没有人口统计偏差吗？

### 参考答案（第 22 页）

不能。男女各占一半，只能说明性别的总体比例均衡。若 50 个女性样本全对应高兴、50 个男性样本全对应愤怒，性别与表情仍然完全关联，存在刻板偏差。年龄、种族或交叉群体也可能缺失或不足。应结合 ENS/SEI 检查覆盖和比例，用 V 检查全局依赖，再用局部 Z 定位具体单元。

## 问题二（第 23 页）

将重加权后的 V 降为 0，就能证明模型实现反事实公平吗？

### 参考答案（第 24 页）

不能。V=0 描述加权数据分布中的统计独立，并不直接约束训练后模型的预测。模型可能使用其他敏感属性代理，或者在某些群体上有更高错误率。反事实公平还需要明确因果结构、敏感属性干预及其后代变量如何变化，再比较个体预测。本组固定 X、切换 A 的配对实验只在明确设定的合成 SCM 下成立，不能直接推广到真实人脸任务。重加权还受缺失单元、有效样本量和同分布精度代价约束，需要保留测试集上的模型评估。

问题页各留约 45 秒；答案一约 90 秒，答案二约 120 秒，合计 5 分钟。逐页完整照读稿见 `演示讲稿.md` 或 `PPT演讲稿.docx`。
'''
(ROOT/'docs/答辩问答.md').write_text(qa,encoding='utf-8')
print('Exported DOCX and exactly 2 Q&A pairs')
