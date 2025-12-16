import os
import logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import cv2
from tensorflow.keras import models
from .config import Config

# --- CUSTOM OBJECTS ---
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
            if not os.path.exists(Config.CLS_MODEL_PATH): return False
            self.classifier = tf.keras.models.load_model(Config.CLS_MODEL_PATH, compile=False)
            print("EfficientNetV2-S Loaded.")
            
            if not os.path.exists(Config.SEG_MODEL_PATH): return False
            custom_objects = {
                'SmoothTruncatedLoss': SmoothTruncatedLoss,
                'soft_dice_loss': soft_dice_loss, 
                'dice_coef': dice_coef
            }
            self.segmenter = tf.keras.models.load_model(Config.SEG_MODEL_PATH, custom_objects=custom_objects, compile=False)
            print("CIA-Net Loaded.")
            
            return True
        except Exception as e:
            print(f"Model Load Error: {e}")
            return False

    def _predict_structured(self, model, img_tensor):
        candidates = []
        if hasattr(model, "input_names") and isinstance(model.input_names, (list, tuple)) and len(model.input_names) == 1:
            candidates.append({model.input_names[0]: img_tensor})
        candidates.append([img_tensor])
        candidates.append(img_tensor)

        last_exc = None
        for inp in candidates:
            try:
                return model.predict(inp, verbose=0)
            except Exception as e:
                last_exc = e
        raise last_exc

    def _call_structured(self, model, img_tensor, training=False):
        candidates = []
        if hasattr(model, "input_names") and isinstance(model.input_names, (list, tuple)) and len(model.input_names) == 1:
            candidates.append({model.input_names[0]: img_tensor})
        candidates.append([img_tensor])
        candidates.append(img_tensor)

        last_exc = None
        for inp in candidates:
            try:
                return model(inp, training=training)
            except Exception as e:
                last_exc = e
        raise last_exc

    def predict_classification(self, img):
        if self.classifier is None: raise RuntimeError("Classifier not loaded!")
        
        img_resized = cv2.resize(img, Config.IMG_SIZE)
        img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) # (1, 224, 224, 3)
        
        preds = self._predict_structured(self.classifier, img_tensor)
        return np.argmax(preds), np.max(preds), img_tensor

    def predict_segmentation(self, img):
        if self.segmenter is None: raise RuntimeError("Segmenter not loaded!")
        
        h_orig, w_orig, _ = img.shape
        img_resized = cv2.resize(img, (224, 224))
        img_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
        
        preds = self._predict_structured(self.segmenter, img_tensor)
        
        nuc_prob = preds[0][0, :, :, 0]
        con_prob = preds[1][0, :, :, 0]
        
        mask_indices = nuc_prob > 0.5
        seg_confidence = np.mean(nuc_prob[mask_indices]) if np.any(mask_indices) else 0.0
        
        nuc_final = cv2.resize(nuc_prob, (w_orig, h_orig))
        con_final = cv2.resize(con_prob, (w_orig, h_orig))
        
        return nuc_final, con_final, seg_confidence

    def generate_gradcam(self, img_tensor, class_idx):
        """
        Robust CAM Implementation (Fixes Functional Graph Disconnection)
        """
        if self.classifier is None: 
            return np.zeros((224, 224))
        
        try:
            print("CAM hesaplanıyor...")
            
            # 1. Base Modeli Bul (İç içe model yapısı)
            base_model = None
            for layer in self.classifier.layers:
                # EfficientNet genelde bir katman olarak görünür
                if 'efficientnet' in layer.name.lower() or 'resnet' in layer.name.lower():
                    base_model = layer
                    break
            
            if base_model is None:
                # Eğer model sequential değilse kendisi base'dir
                base_model = self.classifier

            # 2. Son Conv Katmanını Bul
            target_layer_name = None
            candidate_layers = ['top_activation', 'top_conv', 'conv5_block3_out', 'post_swish']
            
            # Katmanları sondan başa tara
            for layer in reversed(base_model.layers):
                if layer.name in candidate_layers:
                    target_layer_name = layer.name
                    break
                # Fallback: 4 boyutlu çıktı veren son katmanı bul (Batch, H, W, Ch)
                try:
                    if len(layer.output.shape) == 4 and layer.output.shape[1] > 1: 
                        target_layer_name = layer.name
                        break
                except:
                    pass
            
            if target_layer_name is None:
                print("Hedef katman bulunamadı, fallback.")
                return self._activation_based_cam(img_tensor)

            print(f"Hedef Katman: {target_layer_name}")

            # 3. KRİTİK DÜZELTME: Modeli Base Model'den Türet
            # Classifier.input yerine base_model.input kullanıyoruz.
            # Çünkü target_layer base_model'in içinde.
            feature_model = tf.keras.Model(
                inputs=base_model.input, 
                outputs=base_model.get_layer(target_layer_name).output
            )
            
            # 4. Tahmin (Feature Extraction)
            # img_tensor'u float32'ye çevirip veriyoruz
            features = self._call_structured(feature_model, img_tensor.astype(np.float32), training=False)
            features = features.numpy() # (1, 7, 7, 1280)
            
            # 5. Ağırlıkları Al (Dense Layer)
            dense_layer = None
            for layer in reversed(self.classifier.layers):
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_layer = layer
                    break
            
            if dense_layer is None:
                return self._activation_based_cam(img_tensor)
                
            weights = dense_layer.get_weights()[0] # (1280, 3)
            target_weights = weights[:, class_idx]
            
            # 6. CAM Hesaplama
            heatmap = features[0] @ target_weights
            
            # 7. İşleme
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            print(f"CAM Başarılı. Shape: {heatmap.shape}")
            return heatmap

        except Exception as e:
            print(f"CAM Hatası: {e}")
            import traceback
            traceback.print_exc()
            return self._activation_based_cam(img_tensor)
    
    def _activation_based_cam(self, img_tensor):
        try:
            base_model = None
            for layer in self.classifier.layers:
                if 'efficientnet' in layer.name.lower():
                    base_model = layer
                    break
            if base_model is None: base_model = self.classifier
            
            feature_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
            features = self._call_structured(feature_model, img_tensor.astype(np.float32), training=False)
            
            heatmap = np.mean(features[0], axis=-1)
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
            return heatmap
        except:
            return np.zeros((224, 224))