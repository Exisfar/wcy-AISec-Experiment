"""可续传公开文件下载；显式校验 Content-Range，避免错误拼接。"""
from pathlib import Path
import argparse,urllib.request,time,json,hashlib


def download(url,path,attempts=15):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.part')
    for attempt in range(attempts):
        offset=temp.stat().st_size if temp.exists() else 0
        headers={'User-Agent':'Mozilla/5.0'}
        if offset:headers['Range']=f'bytes={offset}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as r:
                content_range=r.headers.get('Content-Range','')
                if offset and r.status==206:
                    if not content_range.startswith(f'bytes {offset}-'):raise ValueError('Unexpected range')
                    mode='ab';total=int(content_range.split('/')[-1])
                else:
                    mode='wb';offset=0;total=int(r.headers.get('Content-Length',0))
                with temp.open(mode) as f:
                    while True:
                        b=r.read(1024*1024)
                        if not b:break
                        f.write(b)
                size=temp.stat().st_size
                if total and size!=total:raise IOError(f'Incomplete download {size}/{total}')
            temp.replace(path)
            h=hashlib.sha256(path.read_bytes()).hexdigest()
            info={'url':url,'bytes':path.stat().st_size,'sha256':h}
            path.with_suffix(path.suffix+'.provenance.json').write_text(json.dumps(info,indent=2))
            print(json.dumps(info),flush=True);return info
        except Exception as e:
            print(f'Download attempt {attempt+1}: {type(e).__name__}: {e}',flush=True)
            if attempt==attempts-1:raise
            time.sleep(min(5,attempt+1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('destination');a=p.parse_args();download(a.url,a.destination)
