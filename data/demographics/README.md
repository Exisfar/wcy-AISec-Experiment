# 作者公开的匿名人口统计数据

20份CSV源自论文作者仓库 `irisdominguez/Dataset_Bias_Metrics` 的 `example_data/`，固定版本为 `a70bdc28cc96b338f4d5fdebce5064b56d80de57`。

文件内容原样保留，共794,166条记录，字段为 `age/race/gender/label`。这些记录不含人脸图片、身份或主体索引；不应按行与下载的真实图片拼接。

每份CSV的SHA256见 `results/provenance.json`。该目录统一随离线Docker镜像交付，外层源码不重复打包数据。真实人脸图片位于 `data/kaggle-affectnet-yolo/extracted/`。
