# tasr
## 程式架構
* train.ipynb : 訓練用的jupyter notebook
* api/ : api用的程式碼
* onnxruntime/ : 將pytorch轉換為onnx格式的程式碼

## 程式環境
* 如requirements.txt

## 執行範例
* 執行train.ipynb即可完成模型訓練，並將模型導出為ONNX格式
* 透過設定train.ipynb中的TRAIN_FOLD參數，即可調整訓練使用的FOLD
* 訓練完0~3號fold之後，便可執行 api/api_onnx_ensemble.py 開啟inference api
