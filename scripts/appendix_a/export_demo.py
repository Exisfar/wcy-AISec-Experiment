"""导出一个已完成模型，用于老师快速重跑真实图片推理；不导出原始图片。"""
from pathlib import Path
import sys,json,shutil,argparse
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from appendix_a.data import sha256_file,CLASSES
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/appendix_a_kaggle.json');p.add_argument('--run',default='exp1_q05_rep0');a=p.parse_args()
cfg=json.loads(a.config.read_text());result=Path(cfg['output_dir'])/a.run
complete=json.loads((result/'complete.json').read_text())
checkpoint=torch.load(Path(cfg['checkpoint_dir'])/a.run/'latest.pt',map_location='cpu',weights_only=False)
assert checkpoint['config_hash']==complete['config_hash'] and checkpoint['epoch']==19
out=ROOT/'results/appendixA_demo';out.mkdir(parents=True,exist_ok=True)
weights=ROOT/'data/models/resnet50_balanced_kaggle.pth';weights.parent.mkdir(parents=True,exist_ok=True);torch.save(checkpoint['model'],weights)
source=pd.read_csv(cfg['manifest']);test=source[source.split=='test'][['sample_id','path','label','gender','split','image_sha256']]
test.to_csv(out/'test_manifest.csv',index=False)
shutil.copy2(result/'confusion.npz',out/'expected_confusion.npz')
meta={'source_run':a.run,'source_config_hash':complete['config_hash'],'trained_on_real_images':True,
      'original_paper_data_match':False,'model_sha256':sha256_file(weights),'test_manifest_sha256':sha256_file(out/'test_manifest.csv'),
      'expected_confusion_sha256':sha256_file(out/'expected_confusion.npz'),'test_rows':len(test),'expected_accuracy':complete['test_accuracy'],
      'classes':list(CLASSES),'image_size':224,'source_dataset':'fatihkgg/affectnet-yolo-format','source_version':2,
      'source_zip_sha256':'883521d22d3f80b8b3da3b221dfbf2f173f186fd9dc380c028cad1f5cbae2c66',
      'dataset_download_url':'https://www.kaggle.com/api/v1/datasets/download/fatihkgg/affectnet-yolo-format?datasetVersionNumber=2',
      'model_path':str(weights.relative_to(ROOT)),'image_root':'data/kaggle-affectnet-yolo/extracted',
      'images_included':all((ROOT/'data/kaggle-affectnet-yolo/extracted'/p).is_file() for p in test.path),'note':'The exported model is trained on the explicitly modified Kaggle subset, not the original paper AffectNet.'}
(out/'model_provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps(meta,ensure_ascii=False,indent=2))
