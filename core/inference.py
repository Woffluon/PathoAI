import os
import logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import cv2
from tensorflow.keras import models
from .config import Config

# --- CUSTOM OBJECTS (Segmentasyon için gerekli) ---
@tf.keras.utils.register_keras_serializable()
class SmoothTruncatedLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=0.2, name="smooth_truncated_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss_outlier = -tf.math.log(self.gamma) + 0.5 * (1 - (pt**2)/(self.gamma**2))
        loss_inlier = -tf.math.log(pt)
        return tf.reduce_mean(tf.where(pt < self.gamma, loss_outlier, loss_inlier))
    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config

def soft_dice_loss(y_true, y_pred): return 1.0
def dice_coef(y_true, y_pred): return 1.0

class InferenceEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance.classifier = None
            cls._instance.segmenter = None
        return cls._instance

    def load_models(self):
        if self.classifier is not None and self.segmenter is not None:
            return True
            
        print(f"Loading models from: {Config.MODEL_DIR}")
        
        try:
            # Sınıflandırma Modeli Yükleme
            if not os.path.exists(Config.CLS_MODEL_PATH): return False
            self.classifier = tf.keras.models.load_model(Config.CLS_MODEL_PATH, compile=False)
            
            # Segmentasyon Modeli Yükleme
            if not os.path.exists(Config.SEG_MODEL_PATH): return False
            custom_objects = {
                'SmoothTruncatedLoss': SmoothTruncatedLoss,
                'soft_dice_loss': soft_dice_loss, 
                'dice_coef': dice_coef
            }
            self.segmenter = tf.keras.models.load_model(Config.SEG_MODEL_PATH, custom_objects=custom_objects, compile=False)
            
            return True
        except Exception as e:
            print(f"Model Load Error: {e}")
            return False

    def predict_classification(self, img):
        if self.classifier is None: raise RuntimeError("Classifier not loaded!")
        
        # Görüntüyü yeniden boyutlandır
        img_resized = cv2.resize(img, Config.IMG_SIZE)
        
        # KRITİK GÜNCELLEME: EfficientNetV2 'include_preprocessing=True' ile eğitildiği için
        # manuel olarak 255'e bölmüyoruz (0-1 arası yapmıyoruz). Model 0-255 arası bekliyor.
        img_tensor = np.expand_dims(img_resized.astype(np.float32), axis=0)
        
        preds = self.classifier.predict(img_tensor, verbose=0)
        return np.argmax(preds), np.max(preds), img_tensor

    def predict_segmentation(self, img):
        if self.segmenter is None: raise RuntimeError("Segmenter not loaded!")
        
        h_orig, w_orig, _ = img.shape
        
        # Segmentasyon için 224x224
        img_resized = cv2.resize(img, (224, 224))
        # CIA-Net (U-Net türevi) genelde 0-1 arası bekler (DenseNet backbone preprocess'ine bağlı ama standart olarak normalize edelim)
        img_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
        
        preds = self.segmenter.predict(img_tensor, verbose=0)
        
        nuc_prob = preds[0][0, :, :, 0]
        con_prob = preds[1][0, :, :, 0]
        
        mask_indices = nuc_prob > 0.5
        if np.any(mask_indices):
            seg_confidence = np.mean(nuc_prob[mask_indices])
        else:
            seg_confidence = 0.0
        
        nuc_final = cv2.resize(nuc_prob, (w_orig, h_orig))
        con_final = cv2.resize(con_prob, (w_orig, h_orig))
        
        return nuc_final, con_final, seg_confidence

    def generate_gradcam(self, img_tensor, class_idx):
        """EfficientNetV2 uyumlu Grad-CAM"""
        if self.classifier is None: 
            return np.zeros((224, 224))
        
        try:
            print("Grad-CAM başlatılıyor (EfficientNetV2)...")
            
            # EfficientNetV2'nin son conv katmanını bulma mantığı
            last_conv_layer = None
            
            # 1. Strateji: 'top_activation' veya 'top_conv' ara (Standart isimlendirme)
            for layer in reversed(self.classifier.layers):
                if 'top_activation' in layer.name or 'top_conv' in layer.name:
                    last_conv_layer = layer
                    break
            
            # 2. Strateji: Bulunamazsa sondan başa doğru ilk 4D çıktı veren Conv katmanını bul
            if last_conv_layer is None:
                for layer in reversed(self.classifier.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv_layer = layer
                        break
                        
            # 3. Strateji: Base model varsa (Nested)
            if last_conv_layer is None:
                for layer in self.classifier.layers:
                    if 'efficientnet' in layer.name.lower():
                        # Base model içindeki son katmanı al
                        # Bu kısım karmaşık olabilir, activation-based fallback'e düşmesi daha güvenli
                        pass

            if last_conv_layer is None:
                print("Hedef katman bulunamadı, Activation CAM kullanılıyor.")
                return self._activation_based_cam(img_tensor)

            print(f"Hedef Katman: {last_conv_layer.name}")

            # Grad Model
            grad_model = tf.keras.Model(
                inputs=self.classifier.input,
                outputs=[last_conv_layer.output, self.classifier.output]
            )

            img_tf = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_tf, training=False)
                if class_idx < predictions.shape[-1]:
                    class_score = predictions[0, class_idx]
                else:
                    class_score = predictions[0, 0]

            grads = tape.gradient(class_score, conv_output)
            
            if grads is None:
                return self._activation_based_cam(img_tensor)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_output_np = conv_output[0].numpy()
            pooled_grads_np = pooled_grads.numpy()
            
            heatmap = np.zeros(conv_output_np.shape[:2], dtype=np.float32)
            for i in range(len(pooled_grads_np)):
                heatmap += pooled_grads_np[i] * conv_output_np[:, :, i]
            
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
                
            return heatmap

        except Exception as e:
            print(f"Grad-CAM Hatası: {e}")
            return self._activation_based_cam(img_tensor)

    def _activation_based_cam(self, img_tensor):
        """Fallback yöntem"""
        try:
            # Son conv katmanını bul
            last_conv = None
            for layer in reversed(self.classifier.layers):
                if isinstance(layer, tf.keras.layers.Conv2D) or 'top_activation' in layer.name:
                    last_conv = layer
                    break
            
            if last_conv is None: return np.zeros((224, 224))
            
            feature_model = tf.keras.Model(inputs=self.classifier.input, outputs=last_conv.output)
            features = feature_model(img_tensor, training=False)
            
            heatmap = tf.reduce_mean(features, axis=-1)[0].numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            return heatmap
        except:
            return np.zeros((224, 224))