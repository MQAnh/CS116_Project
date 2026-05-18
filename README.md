# PyThongML Python Project

Project này được tách từ notebook `PyThongML.ipynb`.

## Cấu trúc

```txt
pythongml_project/
├── data/
│   ├── raw/              # đặt file parquet gốc ở đây
│   └── processed/        # file parquet trung gian
├── models/               # model .pkl + feature columns
├── outputs/              # feature importance
├── notebooks/            # notebook gốc
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── splits.py
│   ├── eda.py
│   ├── candidates.py
│   ├── labels.py
│   ├── features.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── main_train.py
├── main_valid.py
└── main.py
```

## Chuẩn bị dữ liệu

Copy 2 file này vào `data/raw/`:

```txt
data/raw/transaction_full_2025_final.parquet
data/raw/items.parquet
```

## Cài thư viện

```bash
pip install -r requirements.txt
```

## Chạy train

```bash
python main_train.py
```

Script này chạy lại các phần từ notebook:

1. Load + chuẩn hóa data
2. Split tháng 1-9 làm history, tháng 10 làm label
3. Generate train candidates
4. Tạo target
5. Feature engineering
6. Encode feature
7. Train LightGBM
8. Lưu model vào `models/lgbm_baseline.pkl`

## Chạy validation

```bash
python main_valid.py
```

Script này chạy lại validation:

1. History tháng 1-10
2. Label tháng 12 theo notebook gốc
3. Generate valid candidates
4. Tạo validation features
5. Predict score
6. Tính Precision@10

## Chạy end-to-end

```bash
python main.py
```

## Lưu ý quan trọng

Trong notebook gốc, validation đang dùng:

```python
valid_hist_lf = transactions_lf.filter(pl.col("month").is_between(1, 10))
valid_label_lf = transactions_lf.filter(pl.col("month") == 12)
```

Tức là dùng lịch sử tháng 1-10 để đoán tháng 12. Nếu bạn muốn đúng kiểu validate tháng 11 thì sửa trong `src/config.py`:

```python
VALID_LABEL_MONTH = 11
```

Ngoài ra, notebook có đoạn `del train_df` trước khi train lại dùng `train_df`, nên trong project mình đã sửa lại để script chạy liền mạch.
