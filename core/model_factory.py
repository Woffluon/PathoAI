import tensorflow as tf
from tensorflow.keras import layers, models
from .config import Config

class ModelFactory:
    """Factory class to reconstruct model architectures for weight loading."""
    
    @staticmethod
    def build_resnet50_classifier():
        """Reconstructs ResNet50 Classifier."""
        # ImageNet ağırlıklarıyla başlat ki katman isimleri standart olsun
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=Config.INPUT_SHAPE
        )
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(len(Config.CLASSES), activation='softmax')(x)
        return models.Model(inputs=base_model.input, outputs=output)

    @staticmethod
    def build_cia_net():
        """Reconstructs CIA-Net Segmentation Model."""
        
        def IAM_Module(nuc, con, filters):
            concat = layers.Concatenate()([nuc, con])
            smooth = layers.Conv2D(filters, 3, padding='same')(concat)
            nuc_refine = layers.Conv2D(filters, 3, padding='same', activation='relu')(smooth)
            con_refine = layers.Conv2D(filters, 3, padding='same', activation='relu')(smooth)
            return nuc_refine, con_refine

        inputs = layers.Input(shape=(None, None, 3)) 
        
        # ImageNet ağırlıklarıyla başlat
        base = tf.keras.applications.DenseNet121(
            include_top=False, 
            weights='imagenet', 
            input_tensor=inputs
        )
        
        # Feature Extraction Layers (Standard Keras Names)
        enc1 = base.get_layer('conv1_relu').output
        enc2 = base.get_layer('conv2_block6_concat').output
        enc3 = base.get_layer('conv3_block12_concat').output
        enc4 = base.get_layer('conv4_block24_concat').output
        bottleneck = base.get_layer('relu').output

        # Decoder Level 4
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(bottleneck)
        x = layers.UpSampling2D()(x)
        enc4_lat = layers.Conv2D(256, 1, padding='same')(enc4)
        
        m4 = layers.Add()([x, enc4_lat])
        nuc4, con4 = IAM_Module(m4, m4, 256)
        
        # Decoder Level 3
        nuc_up3 = layers.Conv2D(128, 1, padding='same')(layers.UpSampling2D()(nuc4))
        con_up3 = layers.Conv2D(128, 1, padding='same')(layers.UpSampling2D()(con4))
        enc3_lat = layers.Conv2D(128, 1, padding='same')(enc3)
        
        nuc_m3 = layers.Add()([nuc_up3, enc3_lat])
        con_m3 = layers.Add()([con_up3, enc3_lat])
        nuc3, con3 = IAM_Module(nuc_m3, con_m3, 128)
        
        # Decoder Level 2
        nuc_up2 = layers.Conv2D(64, 1, padding='same')(layers.UpSampling2D()(nuc3))
        con_up2 = layers.Conv2D(64, 1, padding='same')(layers.UpSampling2D()(con3))
        enc2_lat = layers.Conv2D(64, 1, padding='same')(enc2)
        
        nuc_m2 = layers.Add()([nuc_up2, enc2_lat])
        con_m2 = layers.Add()([con_up2, enc2_lat])
        nuc2, con2 = IAM_Module(nuc_m2, con_m2, 64)
        
        # Decoder Level 1
        nuc_up1 = layers.Conv2D(32, 1, padding='same')(layers.UpSampling2D()(nuc2))
        con_up1 = layers.Conv2D(32, 1, padding='same')(layers.UpSampling2D()(con2))
        enc1_lat = layers.Conv2D(32, 1, padding='same')(enc1)
        
        nuc_m1 = layers.Add()([nuc_up1, enc1_lat])
        con_m1 = layers.Add()([con_up1, enc1_lat])
        nuc1, con1 = IAM_Module(nuc_m1, con_m1, 32)
        
        # Final Output
        final_nuc = layers.UpSampling2D()(nuc1)
        final_con = layers.UpSampling2D()(con1)
        
        out_nuc = layers.Conv2D(1, 1, activation='sigmoid', name='nuclei_output')(final_nuc)
        out_con = layers.Conv2D(1, 1, activation='sigmoid', name='contour_output')(final_con)
        
        return models.Model(inputs=inputs, outputs=[out_nuc, out_con])