import os
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# QuantizeML-Imports
from quantizeml.models import QuantizationParams, quantize, dump_config

# Akida-Konvertierung (aus cnn2snn)
from cnn2snn import convert, set_akida_version, AkidaVersion

# -------------------------------------------------------------
# Konfiguration
# -------------------------------------------------------------
dataset_base_path = '/home/dhbw-rollerz-/Desktop/Akida/BrainAkida1000/DataSet'
batch_size = 8
epochs = 1
val_split = 0.2  # 20% der Daten für Validierung
debug_mode = 12
threshold_value = 0.05


# -------------------------------------------------------------
# 1. XML-Parsing: Bilder und Polylinien einlesen
# -------------------------------------------------------------
def parse_xml_annotations(xml_path, base_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for image in root.findall('image'):
        rel = image.attrib.get('name')
        if not rel:
            continue
        full = os.path.join(base_dir, rel)
        polys = []
        for poly in image.findall('polyline'):
            lbl = poly.attrib.get('label', 'unknown')
            pts_str = poly.attrib.get('points', '')
            pts = []
            if pts_str:
                for pair in pts_str.split(';'):
                    x_str, y_str = pair.split(',')
                    pts.append((float(x_str), float(y_str)))
            polys.append((lbl, pts))
        data.append((full, polys))
    return data


# -------------------------------------------------------------
# 2. Automatische Datensammlung aus allen Unterordnern (MIT FILTERUNG)
# -------------------------------------------------------------
def collect_all_data(base_path):
    all_data = []

    # Durch alle Unterordner im DataSet-Ordner iterieren
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # Nach annotations.xml und images-Ordner suchen
        xml_path = os.path.join(folder_path, 'annotations.xml')
        img_dir = os.path.join(folder_path, 'images')

        if os.path.exists(xml_path) and os.path.exists(img_dir):
            data = parse_xml_annotations(xml_path, img_dir)
            # Filtere Bilder ohne Polylinien
            filtered_data = [(p, polys) for p, polys in data if os.path.exists(p) and len(polys) > 0]
            all_data.extend(filtered_data)

    print(f"Insgesamt {len(all_data)} Bilder mit gültigen Labels gefunden")
    return all_data


# -------------------------------------------------------------
# 3. Segmentation-Maske erstellen (mit Skalierung & Dilation)
# -------------------------------------------------------------
def create_segmentation_mask(polylines, img_path, target_size=(256, 256)):
    # Kurze Sicherheitsprüfung
    if len(polylines) == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)

    orig = cv2.imread(img_path)
    if orig is None:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
    h0, w0 = orig.shape[:2]
    scale_x = target_size[0] / w0
    scale_y = target_size[1] / h0
    mask = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    for label, pts in polylines:
        if len(pts) < 2:
            continue
        # Punkte skalieren
        scaled_pts = []
        for x, y in pts:
            sx = int(x * scale_x)
            sy = int(y * scale_y)
            scaled_pts.append((sx, sy))
        arr = np.array(scaled_pts, dtype=np.int32).reshape(-1, 1, 2)
        # Kanal wählen
        label_low = label.lower()
        if 'left' in label_low:
            ch_idx = 0
        elif 'center' in label_low:
            ch_idx = 1
        else:
            ch_idx = 2
        tmp = mask[:, :, ch_idx].copy()
        cv2.polylines(tmp, [arr], isClosed=False, color=255, thickness=2)
        tmp = cv2.dilate(tmp, kernel, iterations=1)
        mask[:, :, ch_idx] = tmp
    return mask.astype(np.float32) / 255.0


# -------------------------------------------------------------
# 4. Daten-Generator mit Augmentation
# -------------------------------------------------------------
def load_image(image_path, target_size=(256, 256)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_size)
    return img


def augment_image(image, mask):
    factor = tf.random.uniform([], minval=0.8, maxval=1.0)
    img_aug = image * factor
    img_aug = tf.image.random_contrast(img_aug, lower=0.8, upper=1.2)
    noise = tf.random.normal(shape=tf.shape(img_aug), mean=0.0, stddev=0.05)
    img_aug = tf.clip_by_value(img_aug + noise, 0.0, 1.0)
    if tf.random.uniform([]) < 0.5:
        img_aug = tf.expand_dims(img_aug, 0)
        img_aug = tf.nn.avg_pool(img_aug, ksize=3, strides=1, padding='SAME')
        img_aug = tf.squeeze(img_aug, 0)
    return img_aug, mask


def segmentation_data_generator(data, target_img=(256, 256), target_mask=(256, 256)):
    for img_path, polys in data:
        if not os.path.exists(img_path):
            continue
        img = load_image(img_path, target_img)
        mask = create_segmentation_mask(polys, img_path, target_mask)
        # Original
        yield img, mask
        # Augmentierte Varianten
        for _ in range(3):
            img_aug, mask_aug = augment_image(img, mask)
            yield img_aug, mask_aug


# -------------------------------------------------------------
# 5. Modellaufbau: Encoder & Decoder (ausgeschrieben)
# -------------------------------------------------------------
def build_encoder(input_shape):
    inputs = keras.Input(shape=input_shape, name='encoder_input')
    x = layers.Rescaling(1. / 255)(inputs)
    # Block 1
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # Block 2
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # Block 3
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # Block 4
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # Abschließende Schicht
    x = layers.Conv2D(512, 3, padding='same', name='conv2d_8')(x)
    x = layers.ReLU(max_value=6, name='relu_9')(x)
    return keras.Model(inputs, x, name='akida_encoder')



def build_decoder(input_shape):
    inputs = keras.Input(shape=input_shape, name='decoder_input')
    # UpSampling Block 1
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(inputs)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    # UpSampling Block 2
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    # UpSampling Block 3
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    # UpSampling Block 4
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    # Output Layer
    outputs = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)
    return keras.Model(inputs, outputs, name='enhanced_decoder')


# -------------------------------------------------------------
# 6. Lane-Extraktion: Polylinien aus Vorhersage
# -------------------------------------------------------------
def predict_lane_polylines(model, image, thresh=threshold_value):
    img_resized = tf.image.resize(image, (256, 256))
    inp = tf.expand_dims(img_resized, 0)
    pred = model.predict(inp)[0]
    lanes = {}
    # Linke Spur
    mask_l = (pred[:, :, 0] > thresh).astype(np.uint8) * 255
    mask_l = cv2.morphologyEx(mask_l, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts_l, _ = cv2.findContours(mask_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys_l = []
    for cnt in cnts_l:
        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        if len(pts) > 1:
            polys_l.append(pts)
    lanes['left'] = polys_l

    # Mittlere Spur
    mask_c = (pred[:, :, 1] > thresh).astype(np.uint8) * 255
    mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts_c, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys_c = []
    for cnt in cnts_c:
        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        if len(pts) > 1:
            polys_c.append(pts)
    lanes['center'] = polys_c

    # Rechte Spur
    mask_r = (pred[:, :, 2] > thresh).astype(np.uint8) * 255
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys_r = []
    for cnt in cnts_r:
        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        if len(pts) > 1:
            polys_r.append(pts)
    lanes['right'] = polys_r

    return lanes


# -------------------------------------------------------------
# 8. Visualisierung von 9 zufälligen Beispielbildern mit Labels
# -------------------------------------------------------------
def plot_sample_images(data, num_samples=9):
    plt.figure(figsize=(12, 12))
    selected_samples = random.sample(data, num_samples)

    for i, (img_path, polys) in enumerate(selected_samples):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (256, 256))
        mask = create_segmentation_mask(polys, img_path)

        # Originalbild mit annotierten Linien
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_resized)
        for label, pts in polys:
            if len(pts) < 2:
                continue
            pts = np.array(pts)
            pts[:, 0] = pts[:, 0] * (256 / img.shape[1])  # Skalierung
            pts[:, 1] = pts[:, 1] * (256 / img.shape[0])

            color = 'red' if 'left' in label.lower() else \
                'green' if 'center' in label.lower() else 'blue'
            plt.plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=2)
            plt.scatter(pts[:, 0], pts[:, 1], s=20, color=color)

        plt.title(f"{os.path.basename(img_path)}\nLinien: {len(polys)}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# 7. Hauptprogramm
# -------------------------------------------------------------
if __name__ == '__main__':
    # A) Alle Daten sammeln und aufteilen
    all_data = collect_all_data(dataset_base_path)
    random.shuffle(all_data)  # Zufällige Mischung
    split_idx = int(len(all_data) * (1 - val_split))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    print(f'Gesamte Daten: {len(all_data)} Bilder')
    print(f'Trainingsdaten: {len(train_data)} Bilder')
    print(f'Validierungsdaten: {len(val_data)} Bilder')

    print("\nVisualisierung von 9 zufälligen Trainingsbeispielen:")
    plot_sample_images(train_data)

    # B) Trainings- und Validierungs-Datasets erstellen
    def train_gen():
        yield from segmentation_data_generator(train_data)


    def val_gen():
        yield from segmentation_data_generator(val_data)


    train_dataset = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec((256, 256, 3), tf.float32),
            tf.TensorSpec((256, 256, 3), tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec((256, 256, 3), tf.float32),
            tf.TensorSpec((256, 256, 3), tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # C) Modelle bauen
    encoder = build_encoder((256, 256, 3))
    x_enc = encoder(encoder.input)
    x_red = layers.Conv2D(256, 1, activation='relu')(x_enc)
    decoder = build_decoder((16, 16, 256))
    model = keras.Model(encoder.input, decoder(x_red), name='lane_seg_model')

    # D) Training mit Validierung
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    # E) Quantisierung & Akida-Konvertierung
    qparams = QuantizationParams(
        input_weight_bits=4,
        weight_bits=4,
        activation_bits=4,
        per_tensor_activations=True,
        output_bits=8,
        buffer_bits=32
    )
    q_enc = quantize(encoder, qparams=qparams)
    cfg = dump_config(q_enc)
    if 'relu_9' in cfg and 'output_quantizer' not in cfg['relu_9']:
        cfg['relu_9']['output_quantizer'] = {
            'bitwidth': 4,
            'signed': True,
            'axis': 'per-tensor',
            'buffer_bitwidth': 32
        }
    q_enc2 = quantize(encoder, q_config=cfg, qparams=qparams)
    with set_akida_version(AkidaVersion.v1):
        akida_enc = convert(q_enc2)
    akida_enc.save('akida_encoder.akd')
    akida_enc.save('akida_encoder.fbz')
    encoder.save('encoder.h5')
    decoder.save('decoder.h5')

    # F) Inferenz & Debug-Ausgabe
    for imgs, masks in val_dataset.take(1):  # Jetzt mit Validierungsdaten
        preds = model.predict(imgs)
        for i, img in enumerate(imgs):
            if debug_mode >= 12:
                lanes = predict_lane_polylines(model, img)
                print(f'Bild {i}: lanes = {lanes}')
            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1);
            plt.imshow(img);
            plt.title('Input');
            plt.axis('off')
            plt.subplot(1, 3, 2);
            plt.imshow(masks[i]);
            plt.title('Ground Truth');
            plt.axis('off')
            plt.subplot(1, 3, 3);
            plt.imshow(preds[i]);
            plt.title('Prediction');
            plt.axis('off')
            plt.show()