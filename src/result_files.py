"""Remove reproducible display exports while retaining notebook and model verification inputs."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
KEEP_TABLES = {
    'agreement_representation.csv', 'agreement_global.csv',
    'local_agreement.csv', 'upstream_crosscheck.csv', 'paper_comparison.csv',
}


def cleanup_results():
    results = ROOT / 'results'
    # These outputs are embedded in the notebooks or belong to retired pilot runs.
    for name in ['local', 'appendixA_kaggle_pilot_b128', 'appendixA']:
        path = results / name
        if path.is_dir():
            shutil.rmtree(path)
    for path in results.glob('*.html'):
        path.unlink()
    for name in ['复现报告.md', '真实人脸训练结果.md', '附录A真实训练进度.md', 'real_materials_finished.json']:
        (results / name).unlink(missing_ok=True)
    # Keep the paired-model plot: notebook 02 reads it without retraining the models.
    for path in (results / 'figures').glob('*.png'):
        if path.name != 'appendixA_real_mitigation.png':
            path.unlink()
    for path in (results / 'tables').iterdir():
        if path.is_file() and path.name not in KEEP_TABLES:
            path.unlink()
    for path in (results / 'appendixA_kaggle').glob('figure_*_recall.png'):
        path.unlink()
