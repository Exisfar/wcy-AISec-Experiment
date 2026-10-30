"""公开、支持 HTTP Range 的大文件续传；多个分段在容器内并行下载。"""
from pathlib import Path
import urllib.request,concurrent.futures,json,hashlib,time,argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('destination');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    dest=Path(a.destination);dest.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(a.url,headers={'Range':'bytes=0-0','User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=60) as r:
        if r.status!=206:raise RuntimeError('Server does not support byte ranges')
        total=int(r.headers['Content-Range'].split('/')[-1])
    prefix=dest.with_name(dest.name+'.part');offset=prefix.stat().st_size if prefix.exists() else 0
    assert offset<=total
    step=(total-offset+a.workers-1)//a.workers
    blocks=[(i,offset+i*step,min(total-1,offset+(i+1)*step-1)) for i in range(a.workers) if offset+i*step<total]
    def get(b):
        i,start,end=b;path=dest.with_name(dest.name+f'.chunk{i}')
        for attempt in range(15):
            have=path.stat().st_size if path.exists() else 0
            if have==end-start+1:return path
            if have>end-start+1:raise ValueError('Unexpected chunk size')
            try:
                headers={'Range':f'bytes={start+have}-{end}','User-Agent':'Mozilla/5.0'}
                with urllib.request.urlopen(urllib.request.Request(a.url,headers=headers),timeout=60) as r:
                    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start+have}-'):
                        raise ValueError('Range response mismatch')
                    with path.open('ab') as f:
                        while True:
                            data=r.read(512*1024)
                            if not data:break
                            f.write(data)
                if path.stat().st_size!=end-start+1:raise IOError('Incomplete chunk')
                print(f'Chunk {i} complete ({path.stat().st_size} bytes)',flush=True);return path
            except Exception as e:
                print(f'Chunk {i}, attempt {attempt+1}: {type(e).__name__}: {e}',flush=True)
                if attempt==14:raise
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:paths=list(ex.map(get,blocks))
    merged=dest.with_name(dest.name+'.merged')
    with merged.open('wb') as out:
        for path in ([prefix] if offset else [])+paths:
            with path.open('rb') as f:
                while data:=f.read(4*1024*1024):out.write(data)
    assert merged.stat().st_size==total
    merged.replace(dest)
    meta={'url':a.url,'bytes':total,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    dest.with_suffix(dest.suffix+'.provenance.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
