"""宿主机同源生成可编辑 PPT 与矢量 PDF，不调用 Docker 或办公服务。"""
from pathlib import Path
import subprocess
import fitz
from fontTools import subset
from fontTools.ttLib import TTFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.xmlchemy import OxmlElement
from PIL import Image

INK='#182D3B'; BLUE='#24657B'; ORANGE='#C87E43'; RED='#AE4C4C'
GRAY='#64727B'; LIGHT='#EAEFF1'; PAPER='#FAFAF7'; WHITE='#FFFFFF'

def color(s):return tuple(int(s[i:i+2],16)/255 for i in (1,3,5))

def font_path(bold=False):
    # TTC holds regional variants; index 0 is Japanese, while fc-match selects SC.
    family='Noto Sans CJK SC:style=Bold' if bold else 'Noto Sans CJK SC'
    match=subprocess.check_output(['fc-match',family,'-f','%{file}\n%{index}'],text=True).splitlines()
    source,index=match[0],int(match[1])
    folder=Path(__file__).resolve().parent/'.fonts';folder.mkdir(exist_ok=True)
    target=folder/('SimplifiedChineseBold.otf' if bold else 'SimplifiedChineseRegular.otf')
    if not target.exists() or target.stat().st_mtime < Path(source).stat().st_mtime:
        tt=TTFont(source,fontNumber=index);tt.save(target);tt.close()
    return str(target)

class Deck:
    def __init__(self,root):
        self.root=Path(root);self.prs=Presentation()
        self.prs.slide_width=Inches(13.333);self.prs.slide_height=Inches(7.5)
        self.pdf=fitz.open();self.pages=[];self.notes=[]
        # 在嵌入PDF之前用fontTools裁剪CJK字体，避免MuPDF的CFF子集化问题。
        chars=(self.root/'presentation/make_slides.py').read_text()+(self.root/'presentation/speech_content.json').read_text()
        fonts_dir=self.root/'presentation/.fonts';fonts_dir.mkdir(exist_ok=True)
        self.paths={}
        for bold in (False,True):
            target=fonts_dir/('NotoBold.otf' if bold else 'NotoRegular.otf')
            tt=TTFont(font_path(bold),fontNumber=0)
            options=subset.Options();options.desubroutinize=True;options.retain_gids=True
            sub=subset.Subsetter(options=options);sub.populate(text=chars)
            sub.subset(tt);tt.save(target);tt.close();self.paths[bold]=str(target)
        self.fonts={b:fitz.Font(fontfile=p) for b,p in self.paths.items()}

    def slide(self,title,section,seconds,script,source='',dark=False,backup=False,cover=False):
        self.sl=self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.page=self.pdf.new_page(width=960,height=540);self.pages.append(self.page)
        self.dark=dark;bg=INK if dark else PAPER;fg=WHITE if dark else INK
        self.sl.background.fill.solid();self.sl.background.fill.fore_color.rgb=RGBColor.from_string(bg[1:])
        self.rect(0,0,13.333,7.5,bg)
        if cover:
            self.notes.append({'slide':len(self.pages),'title':title,'seconds':seconds,'backup':backup,'script':script})
            return self
        self.text(.65,.24,10,.35,section,12,BLUE if not dark else '#8FC4CE',True)
        self.text(.65,.85,12,1.65 if '\n' in title else .9,title,30,fg,True)
        self.line(.65,7.03,12.68,7.03,'#D5DDE0' if not dark else '#405460',.7)
        if source:self.text(.65,7.12,11.3,.28,source,9,GRAY if not dark else '#AFC0C8')
        self.text(12.12,7.1,.55,.3,f'{len(self.pages):02}',11,GRAY if not dark else '#AFC0C8',align='right')
        self.notes.append({'slide':len(self.pages),'title':title,'seconds':seconds,'backup':backup,'script':script})
        return self

    def rect(self,x,y,w,h,fill,border=None):
        sh=self.sl.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
        sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill[1:])
        if border:sh.line.color.rgb=RGBColor.from_string(border[1:]);sh.line.width=Pt(.6)
        else:sh.line.fill.background()
        self.page.draw_rect(fitz.Rect(x*72,y*72,(x+w)*72,(y+h)*72),color=color(border) if border else None,fill=color(fill),width=.6)
        return sh

    def line(self,x1,y1,x2,y2,c=GRAY,width=1):
        sh=self.sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
        sh.line.color.rgb=RGBColor.from_string(c[1:]);sh.line.width=Pt(width)
        self.page.draw_line((x1*72,y1*72),(x2*72,y2*72),color=color(c),width=width)

    def wrap(self,s,width,fs,bold):
        font=self.fonts[bold];out=[]
        for para in str(s).split('\n'):
            line=''
            for ch in para:
                if line and font.text_length(line+ch,fontsize=fs)>width:
                    out.append(line.rstrip());line=ch.lstrip()
                else:line+=ch
            out.append(line)
        return out

    def text(self,x,y,w,h,s,fs=22,c=INK,bold=False,align='left'):
        lines=self.wrap(s,w*72-2,fs,bold)
        font=self.fonts[bold];leading=fs*1.23
        needed=fs*(font.ascender-font.descender)+(len(lines)-1)*leading
        if needed>h*72+1:
            raise ValueError(f'Slide {len(self.pages)} text height {needed:.1f}>{h*72:.1f}: {str(s)[:65]}')
        box=self.sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
        tf=box.text_frame;tf.word_wrap=False
        tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
        from pptx.enum.text import PP_ALIGN
        for i,line in enumerate(lines):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
            p.font.name='Microsoft YaHei';p.font.size=Pt(fs);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(c[1:])
            # Set East Asian fonts explicitly: latin-only names allow Office to substitute CJK.
            props=[p._p.get_or_add_pPr().get_or_add_defRPr()]
            props.extend(run._r.get_or_add_rPr() for run in p.runs)
            for prop in props:
                prop.set('lang','zh-CN')
                for tag in ('a:latin','a:ea','a:cs'):
                    element=prop.find(tag,prop.nsmap)
                    if element is None:
                        element=OxmlElement(tag);prop.append(element)
                    element.set('typeface','Microsoft YaHei')
            p.space_before=p.space_after=Pt(0);p.line_spacing=Pt(leading)
            p.alignment=PP_ALIGN.RIGHT if align=='right' else PP_ALIGN.CENTER if align=='center' else PP_ALIGN.LEFT
            span=font.text_length(line,fontsize=fs)
            dx=max(0,(w*72-span)/2) if align=='center' else max(0,w*72-span) if align=='right' else 0
            self.page.insert_text((x*72+dx,y*72+fs*font.ascender+i*leading),line,
                fontname='NotoB' if bold else 'NotoR',fontfile=self.paths[bold],fontsize=fs,color=color(c))
        return box

    def image(self,path,x,y,w,h):
        im=Image.open(path);a=im.width/im.height
        ww=min(w,h*a);hh=ww/a;x+=(w-ww)/2;y+=(h-hh)/2
        self.sl.shapes.add_picture(str(path),Inches(x),Inches(y),width=Inches(ww),height=Inches(hh))
        self.page.insert_image(fitz.Rect(x*72,y*72,(x+ww)*72,(y+hh)*72),filename=str(path))

    def formula(self,expression,x,y,w,h,fs=26,c=INK):
        """Render mathematical notation with fractions and subscripts at presentation resolution."""
        import hashlib
        from matplotlib.mathtext import math_to_image
        directory=self.root/'presentation/.formulas';directory.mkdir(exist_ok=True)
        path=directory/(hashlib.sha256((expression+str(fs)+c).encode()).hexdigest()[:16]+'.png')
        from matplotlib.font_manager import FontProperties
        import matplotlib
        with matplotlib.rc_context({'savefig.transparent':True}):
            math_to_image(expression,str(path),prop=FontProperties(size=fs),dpi=300,color=c,format='png')
        with Image.open(path) as bitmap:natural_w,natural_h=bitmap.width/300,bitmap.height/300
        scale=min(1,w/natural_w,h/natural_h)
        actual_w,actual_h=natural_w*scale,natural_h*scale
        self.image(path,x+(w-actual_w)/2,y+(h-actual_h)/2,actual_w,actual_h)

    def table(self,headers,rows,x=.7,y=2.05,widths=None,rowh=.7,fs=20):
        widths=widths or [12/len(headers)]*len(headers)
        for i,row in enumerate([headers,*rows]):
            xx=x
            for j,value in enumerate(row):
                self.rect(xx,y+i*rowh,widths[j]-.03,rowh-.025,INK if i==0 else WHITE if i%2 else LIGHT)
                self.text(xx+.14,y+i*rowh+.11,widths[j]-.3,rowh-.17,str(value),fs,WHITE if i==0 else INK,i==0)
                xx+=widths[j]

    def save(self):
        main=0;question=1500
        for n in self.notes:
            start=question if n['backup'] else main
            end=start+n['seconds']
            if n['backup']:question=end
            else:main=end
            n['schedule']=f'{start//60:02}:{start%60:02}–{end//60:02}:{end%60:02}'
            self.prs.slides[n['slide']-1].notes_slide.notes_text_frame.text=n['script']
        assert main==1500 and question==1800 and len(self.pages)==25,(main,question,len(self.pages))
        self.prs.save(self.root/'presentation/论文复现汇报.pptx')
        self.pdf.set_metadata({'title':'数据集人口统计偏差：论文阅读与实验复现','author':'课程论文复现小组'})
        # 保留完整CJK字体，避免部分CFF字体子集化造成缺字。
        self.pdf.save(self.root/'presentation/论文复现汇报.pdf',garbage=4,deflate=True)
        return self.notes
