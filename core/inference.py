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
        img_resized = cv2.resize(img, Config.IMG_SIZE)
        img_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
        preds = self.classifier.predict(img_tensor, verbose=0)
        return np.argmax(preds), np.max(preds), img_tensor

    def predict_segmentation(self, img):
        if self.segmenter is None: raise RuntimeError("Segmenter not loaded!")
        
        # --- BOYUT HATASI İÇİN DÜZELTME ---
        # Model sabit 224x224 giriş bekliyor.
        # Görüntüyü ne olursa olsun 224x224 yapıyoruz.
        h_orig, w_orig, _ = img.shape
        
        img_resized = cv2.resize(img, (224, 224))
        img_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
        
        preds = self.segmenter.predict(img_tensor, verbose=0)
        
        # Çıktıları al (Nuclei ve Contour)
        nuc_prob = preds[0][0, :, :, 0]
        con_prob = preds[1][0, :, :, 0]
        
        # Segmentasyon Güven Skoru
        # Sadece hücre olduğunu düşündüğü (>0.5) piksellerin ortalamasını al
        mask_indices = nuc_prob > 0.5
        if np.any(mask_indices):
            seg_confidence = np.mean(nuc_prob[mask_indices])
        else:
            seg_confidence = 0.0
        
        # Çıktıyı orijinal boyuta geri büyüt (Görselleştirme için)
        nuc_final = cv2.resize(nuc_prob, (w_orig, h_orig))
        con_final = cv2.resize(con_prob, (w_orig, h_orig))
        
        return nuc_final, con_final, seg_confidence

    def _find_target_layer(self):
        # ResNet50'nin bilinen son conv katmanını ara
        for layer in self.classifier.layers:
            if 'resnet50' in layer.name: # Nested model varsa
                try:
                    return layer.get_layer('conv5_block3_out').output, layer.output
                except:
                    pass
        # Düz modelse
        try:
            return self.classifier.get_layer('conv5_block3_out').output, self.classifier.output
        except:
            return None, None

    def generate_gradcam(self, img_tensor, class_idx):
        """Generate Grad-CAM heatmap - Keras 3 compatible version for nested ResNet50."""
        if self.classifier is None: 
            return np.zeros((224, 224))
        
        try:
            print("Grad-CAM baslatiliyor...")
            
            # Model yapısını analiz et
            print(f"Model katman sayisi: {len(self.classifier.layers)}")
            
            # ResNet50 base modelini bul (nested model olarak)
            base_model = None
            base_model_layer_idx = None
            for idx, layer in enumerate(self.classifier.layers):
                if 'resnet' in layer.name.lower():
                    base_model = layer
                    base_model_layer_idx = idx
                    print(f"Base model bulundu: {layer.name} (index: {idx})")
                    break
            
            if base_model is not None:
                # Nested model yapısı - ResNet50 bir Functional model olarak içeride
                print("Nested model yapisi tespit edildi")
                
                # ResNet50'nin son conv katmanını bul
                try:
                    last_conv_layer = base_model.get_layer('conv5_block3_out')
                    print(f"Son conv katmani: conv5_block3_out")
                except Exception as e:
                    print(f"conv5_block3_out bulunamadi: {e}")
                    return self._activation_based_cam(img_tensor)
                
                # Yöntem: GradientTape ile manuel hesaplama
                img_tf = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
                
                # Feature extractor model - sadece conv output için
                feature_extractor = tf.keras.Model(
                    inputs=base_model.input,
                    outputs=last_conv_layer.output
                )
                
                print("Gradient hesaplaniyor (nested model)...")
                
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(img_tf)
                    
                    # ResNet50'den feature map al
                    conv_output = feature_extractor(img_tf, training=False)
                    tape.watch(conv_output)
                    
                    # Tam model prediction
                    predictions = self.classifier(img_tf, training=False)
                    
                    # Hedef sınıf skoru
                    if class_idx < predictions.shape[-1]:
                        class_score = predictions[0, class_idx]
                    else:
                        class_score = predictions[0, 0]
                    
                    print(f"Class score: {class_score.numpy():.4f}")
                
                # Gradient hesapla
                grads = tape.gradient(class_score, conv_output)
                del tape
                
                if grads is None:
                    print("Gradient None - activation based CAM deneniyor...")
                    return self._activation_based_cam(img_tensor)
                
                print(f"Gradient shape: {grads.shape}")
                
            else:
                # Düz model yapısı
                print("Duz model yapisi - dogrudan katman aranacak")
                
                # conv5_block3_out veya son Conv2D katmanını bul
                last_conv_layer = None
                try:
                    last_conv_layer = self.classifier.get_layer('conv5_block3_out')
                except:
                    for layer in reversed(self.classifier.layers):
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            last_conv_layer = layer
                            break
                
                if last_conv_layer is None:
                    print("Conv katmani bulunamadi")
                    return self._activation_based_cam(img_tensor)
                
                print(f"Son conv katmani: {last_conv_layer.name}")
                
                # Grad model oluştur
                grad_model = tf.keras.Model(
                    inputs=self.classifier.input,
                    outputs=[last_conv_layer.output, self.classifier.output]
                )
                
                img_tf = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
                
                print("Gradient hesaplaniyor (duz model)...")
                
                with tf.GradientTape() as tape:
                    conv_output, predictions = grad_model(img_tf, training=False)
                    
                    if class_idx < predictions.shape[-1]:
                        class_score = predictions[0, class_idx]
                    else:
                        class_score = predictions[0, 0]
                
                grads = tape.gradient(class_score, conv_output)
                
                if grads is None:
                    print("Gradient None")
                    return self._activation_based_cam(img_tensor)
            
            # Grad-CAM hesaplama
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_output_np = conv_output[0].numpy()
            pooled_grads_np = pooled_grads.numpy()
            
            print(f"Conv output shape: {conv_output_np.shape}")
            print(f"Pooled grads shape: {pooled_grads_np.shape}")
            
            # Weighted sum
            heatmap = np.zeros(conv_output_np.shape[:2], dtype=np.float32)
            for i in range(len(pooled_grads_np)):
                heatmap += pooled_grads_np[i] * conv_output_np[:, :, i]
            
            # ReLU ve normalize
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            print(f"Heatmap shape: {heatmap.shape}, max: {np.max(heatmap):.4f}")
            print("Grad-CAM basariyla olusturuldu.")
            return heatmap
            
        except Exception as e:
            print(f"Grad-CAM Error: {e}")
            import traceback
            traceback.print_exc()
            return self._activation_based_cam(img_tensor)
    
    def _activation_based_cam(self, img_tensor):
        """Fallback: Activation-based Class Activation Map (gradient gerektirmez)."""
        try:
            print("Activation-based CAM kullaniliyor...")
            
            # ResNet50 base modelini bul
            base_model = None
            for layer in self.classifier.layers:
                if 'resnet' in layer.name.lower():
                    base_model = layer
                    break
            
            if base_model is not None:
                # Nested model
                try:
                    last_conv = base_model.get_layer('conv5_block3_out')
                except:
                    # Son conv katmanını bul
                    last_conv = None
                    for layer in reversed(base_model.layers):
                        if 'conv' in layer.name and hasattr(layer, 'output'):
                            last_conv = layer
                            break
                
                if last_conv is None:
                    return np.zeros((224, 224))
                
                # Feature extractor
                feature_model = tf.keras.Model(
                    inputs=base_model.input,
                    outputs=last_conv.output
                )
                features = feature_model(img_tensor, training=False)
            else:
                # Düz model - son Conv2D bul
                last_conv = None
                for layer in reversed(self.classifier.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv = layer
                        break
                
                if last_conv is None:
                    return np.zeros((224, 224))
                
                feature_model = tf.keras.Model(
                    inputs=self.classifier.input,
                    outputs=last_conv.output
                )
                features = feature_model(img_tensor, training=False)
            
            # Aktivasyonların ortalamasını al
            heatmap = tf.reduce_mean(features, axis=-1)[0].numpy()
            heatmap = np.maximum(heatmap, 0)
            
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            print(f"Activation CAM shape: {heatmap.shape}")
            return heatmap
            
        except Exception as e:
            print(f"Activation CAM Error: {e}")
            return np.zeros((224, 224))