# =========================================================
# GEREKLİ KÜTÜPHANELER
# =========================================================

# PyTorch ana kütüphanesi
# Tensor işlemleri ve model çalıştırma için kullanılır
import torch

# Derin öğrenme katmanlarını oluşturmak için kullanılır
# Conv2D, Linear, ReLU gibi katmanlar burada bulunur
import torch.nn as nn

# Görüntüleri modele uygun hale getirmek için kullanılır
# Resize, Normalize, ToTensor gibi işlemler içerir
from torchvision import transforms

# Görüntü dosyasını açmak için kullanılır
from PIL import Image

# Dosya yolları ve klasör işlemleri için
import os


# CNN modeline verilecek tüm görüntüler 128x128 boyutuna dönüştürülecek
# Böylece tüm görüntüler aynı boyutta olur
IMG_SIZE = 128

# Eğitilmiş modelin kayıtlı olduğu dosya yolu
MODEL_PATH = "models/fire_smoke_cnn.pth"


# Eğer bilgisayarda NVIDIA GPU varsa CUDA kullanılır
# Yoksa model CPU üzerinde çalışır
# Bu satır modeli otomatik optimize eder
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# CNN MODELİ
# nn.Module:
# PyTorch'ta tüm derin öğrenme modellerinin temel sınıfıdır
class FireSmokeCNN(nn.Module):

    def __init__(self, num_classes):

        # Parent class constructor çağrılır
        super(FireSmokeCNN, self).__init__()


        # Bu bölüm görüntüden özellik çıkarır
        # CNN burada:
        # - alev rengi
        # - parlaklık
        # - duman yoğunluğu
        # - şekil yapıları
        # gibi görsel örüntüleri öğrenir
        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            # ReLU aktivasyon fonksiyonu: Negatif değerleri sıfırlar. Modele doğrusal olmayan öğrenme kazandırır
            nn.ReLU(),

            # MaxPooling: Görüntü boyutunu küçültür En önemli özellikleri korur
            nn.MaxPool2d(2),

            # İlk katmandan gelen feature map'leri işler
            # Daha karmaşık yapıları öğrenmeye başlar

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Bu katmanda model artık:
            # - alev şekli
            # - duman dağılımı
            # - yoğunluk örüntüsü
            # gibi daha karmaşık özellikleri öğrenebilir

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        # CNN'den çıkan özellikleri analiz ederve görüntünün hangi sınıfa ait olduğunu belirler

        self.classifier = nn.Sequential(

            # Çok boyutlu feature map'i
            # tek boyutlu vektöre çevirir
            nn.Flatten(),

            # Bu katman CNN tarafından çıkarılan özellikleri analiz ederek daha anlamlı temsillere dönüştürmektedir.
            nn.Linear(
                64 * 16 * 16,
                128
            ),

            nn.ReLU(),

            # Dropout:Overfitting'i azaltır. Modelin ezber yapmasını engeller
            nn.Dropout(0.3),

            # Son sınıflandırma katmanı
            # Çıktı: fire / smoke / normal
            nn.Linear(
                128,
                num_classes
            )
        )

    # FORWARD FONKSİYONU
    # Veri model içinde nasıl ilerleyecek burada tanımlanır
    def forward(self, x):

        # Görüntü önce feature extraction kısmına gider
        x = self.features(x)

        # Sonra classification kısmına gider
        x = self.classifier(x)

        # Son tahmin çıktısı döndürülür
        return x


# eğitilmiş CNN modelini dosyadan okuyup ağırlıklarını yükleyerek tahmin yapmaya hazır hale getirmektedir.
# Eğitim sırasında kaydedilen model okunur
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Sınıf isimleri alınır
# Örneğin:
# ["fire", "normal", "smoke"]
classes = checkpoint["classes"]

# CNN modeli oluşturulur
model = FireSmokeCNN(
    len(classes)
).to(device)

# Eğitilmiş ağırlıklar modele yüklenir
model.load_state_dict(
    checkpoint["model_state_dict"]
)

# Model inference/test moduna geçirilir
# Dropout gibi eğitim davranışları kapanır
model.eval()


# GÖRÜNTÜ PREPROCESSING
# Kamera görüntülerini CNN modelinin anlayacağı hale getirir

transform = transforms.Compose([

    # Görüntü boyutu sabitlenir
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    # Görüntü PyTorch tensor formatına çevrilir
    transforms.ToTensor(),

    # Normalize işlemi: Piksel değerlerini dengeler Modelin daha stabil çalışmasını sağlar
    # Normalize işlemi ile görüntü değerleri belirli bir aralığa çekilmektedir. Bu sayede model daha kararlı çalışmakta ve eğitim sırasında daha hızlı öğrenebilmektedir
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# GÖRÜNTÜ TAHMİN FONKSİYONU

# Kamera görüntüsünü analiz eder ve: fire / smoke / normal tahmini üretir
def predict_image(image_path):

    # Burada kamera görüntüsü okunuyor.
    image = Image.open(image_path)

    # Model RGB formatında çalıştığı için görüntü üç kanallı RGB yapısına dönüştürülmektedir
    image = image.convert("RGB")

    # Görüntü preprocessing
    # Resize + Tensor + Normalize işlemleri uygulanır
    image = transform(image)

    # Batch boyutu eklenir
    # CNN modelleri:
    # [batch, channel, height, width]
    # formatında veri bekler
    # CNN modeli batch mantığıyla çalıştığı için tek bir görüntü kullanılsa bile batch boyutu eklenmektedi
    image = image.unsqueeze(0).to(device)

    # Tahmin işlemi
    # Gradient hesaplama kapatılır Çünkü burada eğitim değil sadece tahmin yapılıyor
    with torch.no_grad():

        # Görüntü CNN modeline gönderilir
        outputs = model(image)

        # En yüksek olasılığa sahip sınıf alınır
        _, predicted = torch.max(outputs, 1)

    # Tahmin edilen sınıf indexi gerçek sınıf adına çevrilir
    predicted_class = classes[predicted.item()]

    # Sonuç döndürülür
    return predicted_class