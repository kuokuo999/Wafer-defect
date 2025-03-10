from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import datetime
import cv2

# 確保 GPU 設定正常
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print("Error setting memory growth:", e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])  # 限制為 2GB
    except RuntimeError as e:
        print(e)

# -------------------------
# 資料處理
# -------------------------
# 載入資料
df = pd.read_pickle("WM811K.pkl")
df['waferMap'] = df['waferMap'].apply(lambda x: np.zeros((52, 52)) if not isinstance(x, np.ndarray) else x)


def pad_or_crop(wafer_map, target_shape=(52, 52)):
    current_height, current_width = wafer_map.shape
    target_height, target_width = target_shape

    # 如果圖像尺寸小於目標尺寸，先進行等比例縮放
    if current_height < target_height or current_width < target_width:
        scale = min(target_height / current_height, target_width / current_width)
        new_height = max(1, int(current_height * scale))  # 防止 new_height = 0
        new_width = max(1, int(current_width * scale))  # 防止 new_width = 0
        wafer_map = cv2.resize(wafer_map, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # 縮放後的圖像尺寸
    scaled_height, scaled_width = wafer_map.shape

    # 確保縮放後的尺寸不超過目標大小
    if scaled_height > target_height or scaled_width > target_width:
        wafer_map = wafer_map[:target_height, :target_width]
        scaled_height, scaled_width = wafer_map.shape

    # 初始化一個目標大小的黑色背景
    padded = np.zeros(target_shape, dtype=wafer_map.dtype)
    
    # 將縮放後的圖像居中填充到目標大小
    offset_height = (target_height - scaled_height) // 2
    offset_width = (target_width - scaled_width) // 2
    padded[offset_height:offset_height + scaled_height, offset_width:offset_width + scaled_width] = wafer_map

    return padded

# 添加調試信息
df['waferMap'] = df['waferMap'].apply(lambda x: pad_or_crop(x, target_shape=(52, 52)))



# 分割訓練集和測試集
trainIdx = df[df['trainTestLabel'] == 'Training'].index
testIdx = df[df['trainTestLabel'] == 'Test'].index
if len(testIdx) == 0:  # 如果測試集為空，隨機拆分
    trainIdx = df.sample(frac=0.8, random_state=42).index
    testIdx = df.drop(trainIdx).index

train_maps = np.stack(df.loc[trainIdx, 'waferMap'].values).astype(np.float32)
test_maps = np.stack(df.loc[testIdx, 'waferMap'].values).astype(np.float32)

# 將標籤轉換為類別索引並進行 One-Hot 編碼
train_labels = df.loc[trainIdx, 'failureType']
test_labels = df.loc[testIdx, 'failureType']

unique_labels = train_labels.unique()
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
train_labels = train_labels.map(label_mapping).values
test_labels = test_labels.map(label_mapping).values

train_labels_categorical = to_categorical(train_labels, num_classes=len(unique_labels))
test_labels_categorical = to_categorical(test_labels, num_classes=len(unique_labels))

# 打印每個類別的名稱，按照資料集的順序
print("Classes in dataset (sorted by the order in dataset):")
for label in unique_labels:
    print(label)

# 確保資料形狀符合模型輸入
train_maps = train_maps[..., np.newaxis]
test_maps = test_maps[..., np.newaxis]

#print("Test Maps dtype:", test_maps.dtype)  # 應該是 np.float32
#print("Test Maps shape:", test_maps.shape)  # 應該是 (樣本數, 52, 52, 1)
#print("Train Maps shape and dtype:", train_maps.shape, train_maps.dtype)
#print("Train Labels Categorical shape and dtype:", train_labels_categorical.shape, train_labels_categorical.dtype)

# -------------------------
# 定義簡化的模型
# -------------------------
def create_simple_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)#128#64
    x = layers.Dropout(0.6)(x)#0.5#0.4
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    return models.Model(inputs, outputs, name="Simplified_Model")




#    x = layers.MaxPooling2D((2, 2))(x)
#    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)



# -------------------------
# 創建模型和編譯
# -------------------------
input_shape = (52, 52, 1)
num_classes = len(unique_labels)
simple_model = create_simple_model(input_shape, num_classes)
simple_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------
# 設定 TensorBoard 回調
# -------------------------
# 創建 TensorBoard 日誌目錄，根據當前時間命名
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# -------------------------
# 訓練模型
# -------------------------
history = simple_model.fit(
    train_maps, 
    train_labels_categorical, 
    epochs=15, #8
    batch_size=4, 
    validation_data=(test_maps, test_labels_categorical),
    callbacks=[tensorboard_callback]  # 添加 TensorBoard 回調
)

# -------------------------
# 繪製訓練過程
# -------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training and Validation')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(loc='upper right')
plt.show()

#=================================================================
# -------------------------
# 評估模型
# -------------------------
# 測試分批預測
# -------------------------
batch_size = 16
num_batches = len(test_maps) // batch_size
all_predictions = []

for i in range(num_batches + 1):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(test_maps))
    if start >= end:  # 避免空批次
        break
    batch_predictions = simple_model.predict(test_maps[start:end])
    all_predictions.append(batch_predictions)

predictions = np.concatenate(all_predictions, axis=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels_categorical, axis=1)

print("Overall Predictions shape:", predictions.shape)

# -------------------------
# 繪製混淆矩陣
# -------------------------
cm = confusion_matrix(true_classes, predicted_classes, labels=range(num_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')

# 調整 x 軸和 y 軸標籤的字體大小和旋轉角度
plt.xticks(fontsize=10, rotation=45, ha='right')  # 調整 x 軸標籤大小並旋轉 45 度
plt.yticks(fontsize=10)  # 調整 y 軸標籤大小

# 增加圖表的邊距，避免字黏在一起
plt.tight_layout()

plt.title('Confusion Matrix')
plt.show()

# -------------------------
# 顯示每個類別的辨識結果
# -------------------------
# 找到每個類別的正確預測
correct_predictions = (predicted_classes == true_classes)

fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
fig.suptitle('Sample Predictions per Class', fontsize=16)

for idx, label in enumerate(unique_labels):
    # 找到該類別且預測正確的索引
    class_indices = np.where((true_classes == idx) & correct_predictions)[0]
    if len(class_indices) > 0:
        sample_idx = class_indices[0]  # 挑選第一張正確的圖片
        axes[idx].imshow(test_maps[sample_idx].squeeze(), cmap='viridis')# 改為viridis彩映顏色
        axes[idx].set_title(f"Class: {label}")
        axes[idx].axis('off')
    else:
        axes[idx].text(0.5, 0.5, "No Image", fontsize=5, ha='center', va='center')
        axes[idx].axis('off')

plt.tight_layout()
plt.show()
