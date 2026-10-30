"""从同一份逐页讲稿导出 Word，并同步精简为两个问答。只使用标准库。"""
from pathlib import Path
import json,zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
notes=json.loads((ROOT/'presentation/slide_plan.json').read_text())
assert len(notes)==25
paras=[]
def paragraph(text,style='Normal'):
    return '<w:p><w:pPr><w:pStyle w:val="'+style+'"/></w:pPr><w:r><w:t xml:space="preserve">'+escape(text)+'</w:t></w:r></w:p>'
paras.append(paragraph('PPT 逐页演讲稿','Title'))
paras.append(paragraph('主汇报 25 分钟 + 两个问答 5 分钟','Subtitle'))
paras.append(paragraph('对应 25 页 PPT：第 1–21 页主汇报，第 22–25 页依次为问题一、答案一、问题二、答案二。【】中的内容为操作与停顿提示，不需读出。'))
for n in notes:
    paras.append(paragraph(f'第 {n["slide"]} 页  {n["title"]}','Heading1'))
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
path=ROOT/'presentation/PPT演讲稿.docx'
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
    for name,contents in [('[Content_Types].xml',ct),('_rels/.rels',rels),('word/document.xml',doc),('word/styles.xml',styles),('word/_rels/document.xml.rels',drels)]:
        ET.fromstring(contents);z.writestr(name,contents)
qa=['# 讲后讨论：两个问题及参考答案','',
    '第22–25页依次为问题一、答案一、问题二、答案二。每个问题与答案分别占一页。','']
for n in notes[21:]:
    qa.extend([f'## 第{n["slide"]}页 · {n["title"]}','',n['script'],''])
(ROOT/'presentation/答辩问答.md').write_text('\n'.join(qa),encoding='utf-8')
print('已同步导出Word演讲稿及两个问题的参考答案')
