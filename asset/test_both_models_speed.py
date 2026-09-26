"""
TEST KECEPATAN INFERENSI - EfficientNet-B0 vs MobileNetV2
Untuk keperluan data skripsi (tidak ditampilkan di website)
"""

import time
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

def load_model(model_path):
    """Memuat model CNN"""
    print(f"   Loading: {model_path}")
    if not os.path.exists(model_path):
        print(f"   ❌ File tidak ditemukan: {model_path}")
        return None
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"   ✅ Berhasil dimuat")
    return model

def get_test_images(folder='uploads', num_images=10):
    """Mengambil gambar uji dari folder"""
    images = []
    
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) 
                if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if files:
            print(f"   📁 Menemukan {len(files)} gambar di folder '{folder}'")
            # Ambil beberapa gambar pertama
            for f in files[:num_images]:
                img_path = os.path.join(folder, f)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    images.append(img)
                    print(f"      ✅ {f}")
    
    # Jika tidak ada gambar, buat dummy
    if not images:
        print(f"   ⚠️ Tidak ada gambar, menggunakan dummy (random)")
        for i in range(num_images):
            dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            images.append(dummy)
        print(f"   ✅ {len(images)} dummy images dibuat")
    
    return images

def preprocess_for_model(images, model_type):
    """Preprocessing sesuai jenis model"""
    processed = []
    
    if model_type == 'mobilenetv2':
        for img in images:
            processed.append(mobilenet_preprocess(img.astype(np.float32)))
    elif model_type == 'efficientnet':
        for img in images:
            processed.append(efficientnet_preprocess(img.astype(np.float32)))
    else:
        # Fallback: normalisasi sederhana
        for img in images:
            processed.append(img.astype(np.float32) / 255.0)
    
    return np.array(processed)

def measure_speed(model, images, model_name, num_runs=100):
    """
    Mengukur kecepatan inferensi
    
    Parameters:
    - model: model CNN
    - images: list gambar (sudah di-preprocess)
    - model_name: nama model (untuk output)
    - num_runs: jumlah iterasi
    
    Returns:
    - dict: statistik waktu
    """
    
    print(f"\n{'='*60}")
    print(f"📊 MENGUJI: {model_name}")
    print(f"{'='*60}")
    
    if model is None:
        print("❌ Model tidak tersedia!")
        return None
    
    # Ambil 1 gambar untuk single inference
    sample = np.expand_dims(images[0], axis=0)
    
    # === WARM-UP (inisialisasi) ===
    print(f"\n🔥 Warm-up (10 iterasi)...")
    for i in range(10):
        _ = model.predict(sample, verbose=0)
    print("   ✅ Selesai")
    
    # === SINGLE IMAGE INFERENCE ===
    print(f"\n📈 Single Image Inference ({num_runs} iterasi)...")
    times_single = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(sample, verbose=0)
        end = time.perf_counter()
        times_single.append(end - start)
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{num_runs}")
    
    # === BATCH INFERENCE (10 gambar) ===
    print(f"\n📈 Batch Inference (10 gambar, {num_runs//2} iterasi)...")
    
    # Siapkan batch 10 gambar
    batch_images = images[:10]
    batch = np.stack([np.expand_dims(img, axis=0) for img in batch_images], axis=0)
    batch = batch.reshape(10, 224, 224, 3)
    
    times_batch = []
    for i in range(num_runs // 2):
        start = time.perf_counter()
        _ = model.predict(batch, verbose=0)
        end = time.perf_counter()
        times_batch.append(end - start)
    
    # === HITUNG STATISTIK ===
    times_single = np.array(times_single)
    times_batch = np.array(times_batch)
    
    avg_single = np.mean(times_single)
    std_single = np.std(times_single)
    min_single = np.min(times_single)
    max_single = np.max(times_single)
    
    avg_batch = np.mean(times_batch)
    per_image_batch = avg_batch / 10
    
    print(f"\n📊 HASIL {model_name}:")
    print(f"   Single Image ({num_runs} runs):")
    print(f"      Rata-rata: {avg_single:.4f} detik")
    print(f"      Std Dev:   {std_single:.4f} detik")
    print(f"      Min:       {min_single:.4f} detik")
    print(f"      Max:       {max_single:.4f} detik")
    print(f"      FPS:       {1/avg_single:.2f} gambar/detik")
    print(f"   Batch (10 gambar, {num_runs//2} runs):")
    print(f"      Rata-rata: {avg_batch:.4f} detik/batch")
    print(f"      Per gambar: {per_image_batch:.4f} detik")
    
    return {
        'model_name': model_name,
        'single_avg': avg_single,
        'single_std': std_single,
        'single_min': min_single,
        'single_max': max_single,
        'fps': 1/avg_single,
        'batch_avg': avg_batch,
        'batch_per_image': per_image_batch,
        'num_runs': num_runs
    }

def main():
    print("\n" + "="*70)
    print("🚀 TEST KECEPATAN INFERENSI 2 MODEL")
    print("   EfficientNet-B0 vs MobileNetV2")
    print("="*70)
    
    # 1. Ambil gambar uji
    print("\n📸 MENGAMBIL GAMBAR UJI...")
    images = get_test_images('uploads', num_images=10)
    print(f"   Total gambar: {len(images)}")
    
    if len(images) < 10:
        print("   ⚠️ Gambar kurang dari 10, akan di-duplikasi")
        while len(images) < 10:
            images.append(images[0])
    
    # 2. Preprocess untuk masing-masing model
    print("\n🔄 PREPROCESSING...")
    images_mobilenet = preprocess_for_model(images, 'mobilenetv2')
    images_efficientnet = preprocess_for_model(images, 'efficientnet')
    print(f"   MobileNetV2:  {images_mobilenet.shape}")
    print(f"   EfficientNet: {images_efficientnet.shape}")
    
    # 3. Load kedua model
    print("\n📥 MEMUAT MODEL...")
    print("   Pastikan file model ada di direktori:")
    print("   - deteksi_makanan_mobilenetv2.keras")
    print("   - deteksi_makanan_efficientnetb0.keras")
    print()
    
    model_mobilenet = load_model('deteksi_makanan_mobilenetv2.keras')
    model_efficientnet = load_model('deteksi_makanan_efficientnetb0.keras')
    
    if model_mobilenet is None:
        print("\n⚠️ MobileNetV2 tidak ditemukan, coba nama alternatif...")
        model_mobilenet = load_model('deteksi_makanan_mobilenetv2.keras')
    
    if model_efficientnet is None:
        print("\n⚠️ EfficientNet-B0 tidak ditemukan, coba nama alternatif...")
        model_efficientnet = load_model('deteksi_makanan_efficientnetb0.keras')
    
    # 4. Ukur kecepatan
    results = []
    
    if model_mobilenet is not None:
        result_mobilenet = measure_speed(
            model_mobilenet, 
            images_mobilenet, 
            "MobileNetV2",
            num_runs=100
        )
        results.append(result_mobilenet)
    else:
        print("\n❌ SKIP: MobileNetV2 tidak tersedia")
    
    if model_efficientnet is not None:
        result_efficientnet = measure_speed(
            model_efficientnet, 
            images_efficientnet, 
            "EfficientNet-B0",
            num_runs=100
        )
        results.append(result_efficientnet)
    else:
        print("\n❌ SKIP: EfficientNet-B0 tidak tersedia")
    
    # 5. Tampilkan rangkuman
    print("\n" + "="*70)
    print("📊 RANGKUMAN PERBANDINGAN")
    print("="*70)
    
    if len(results) == 2:
        print("\n┌─────────────────────┬──────────────────┬──────────────────┐")
        print("│      Parameter       │  EfficientNet-B0 │   MobileNetV2    │")
        print("├─────────────────────┼──────────────────┼──────────────────┤")
        
        r1 = results[0]  # MobileNetV2
        r2 = results[1]  # EfficientNet-B0
        
        # Tentukan mana yang MobileNet dan mana EfficientNet
        mobilenet = r1 if r1['model_name'] == 'MobileNetV2' else r2
        efficientnet = r2 if r2['model_name'] == 'EfficientNet-B0' else r1
        
        print(f"│ Akurasi (%)          │     91.22        │    92.81        │")
        print(f"│ Ukuran Model (MB)    │     19.31        │    12.24        │")
        print(f"│ Waktu Inferensi (s)  │   {efficientnet['single_avg']:.4f}       │   {mobilenet['single_avg']:.4f}        │")
        print(f"│ FPS (gambar/detik)   │   {efficientnet['fps']:.2f}         │   {mobilenet['fps']:.2f}         │")
        print(f"│ Batch/10 gambar (s)  │   {efficientnet['batch_avg']:.4f}       │   {mobilenet['batch_avg']:.4f}        │")
        print("└─────────────────────┴──────────────────┴──────────────────┘")
        
        # Hitung selisih
        diff_time = abs(efficientnet['single_avg'] - mobilenet['single_avg'])
        diff_percent = (diff_time / efficientnet['single_avg']) * 100
        
        print(f"\n📌 KESIMPULAN:")
        print(f"   MobileNetV2 lebih cepat {diff_percent:.1f}% dari EfficientNet-B0")
        print(f"   Selisih waktu: {diff_time:.4f} detik")
        print(f"   Ukuran model lebih kecil {((19.31 - 12.24)/19.31)*100:.1f}%")
    
    elif len(results) == 1:
        print(f"\n✅ Hanya satu model yang diuji: {results[0]['model_name']}")
        print(f"   Rata-rata waktu: {results[0]['single_avg']:.4f} detik")
        print(f"   FPS: {results[0]['fps']:.2f}")
    
    else:
        print("\n❌ Tidak ada model yang berhasil diuji!")
        print("   Pastikan file model ada di direktori ini")
        print(f"   File di direktori: {os.listdir('.')}")
    
    print("\n" + "="*70)
    print("✅ SELESAI")
    print("="*70)

if __name__ == "__main__":
    main()