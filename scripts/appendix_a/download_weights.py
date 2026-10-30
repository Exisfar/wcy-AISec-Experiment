#!/usr/bin/env python3
"""仅下载公开的官方预训练权重；不下载或绕过 AffectNet 原图的申请。"""
from pathlib import Path
import urllib.request,hashlib,json,argparse
ROOT=Path(__file__).resolve().parents[2]
URL='https://download.pytorch.org/models/resnet50-0676ba61.pth'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'data/models/resnet50-0676ba61.pth');args=parser.parse_args()
    dest=args.output;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        tmp=dest.with_suffix('.download');urllib.request.urlretrieve(URL,tmp);tmp.rename(dest)
    digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    if not digest.startswith('0676ba61'):raise ValueError('Official torchvision weight hash prefix failed')
    meta={'url':URL,'sha256':digest,'bytes':dest.stat().st_size,'pretrained':'ImageNet-1K V1, not AffectNet-trained'}
    dest.with_suffix('.provenance.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
