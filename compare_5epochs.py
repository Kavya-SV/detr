import csv

def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [{k.strip(): v.strip() for k, v in row.items()} for row in reader]

v2 = load_csv(r'C:\Chokam\detr\runs\detect\RTDETR_Chokam_V2\rain\results.csv')
v1 = load_csv(r'C:\Chokam\detr\runs\detect\RTDETR_Chokam\rain\results.csv')
base = load_csv(r'C:\Chokam\detr\runs\detect\RTDETR_Baseline_Pure\rain\results.csv')

print('=== CHOKAM V2 (5 Epochs) ===')
for row in v2:
    print('Epoch ' + row['epoch'] + ': mAP50=' + row['metrics/mAP50(B)'] + ', mAP50-95=' + row['metrics/mAP50-95(B)'] + ', P=' + row['metrics/precision(B)'] + ', R=' + row['metrics/recall(B)'] + ', Sparsity=' + row['train/chokam_sparsity'] + ', Score=' + row['train/chokam_score'])

print('\n=== CHOKAM V1 (Epochs 1-5) ===')
for row in v1[:len(v2)]:
    print('Epoch ' + row['epoch'] + ': mAP50=' + row['metrics/mAP50(B)'] + ', mAP50-95=' + row['metrics/mAP50-95(B)'] + ', P=' + row['metrics/precision(B)'] + ', R=' + row['metrics/recall(B)'] + ', Sparsity=' + row['train/chokam_sparsity'] + ', Score=' + row['train/chokam_score'])

print('\n=== BASELINE PURE (Epochs 1-5) ===')
for row in base[:len(v2)]:
    print('Epoch ' + row['epoch'] + ': mAP50=' + row['metrics/mAP50(B)'] + ', mAP50-95=' + row['metrics/mAP50-95(B)'] + ', P=' + row['metrics/precision(B)'] + ', R=' + row['metrics/recall(B)'])
