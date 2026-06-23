# =========================================================
# GEREKLİ KÜTÜPHANELER
# =========================================================

# PyTorch ana kütüphanesi
# Tensor işlemleri, model eğitimi ve GPU kullanımı için gerekli
import torch

# Derin öğrenme katmanlarını oluşturmak için kullanılır
# Conv2D, Linear, ReLU gibi katmanlar burada bulunur
import torch.nn as nn

# Modeli optimize etmek yani öğrenmesini sağlamak için kullanılır
# Adam optimizer burada kullanılacak
import torch.optim as optim

# Görüntü veri setini yüklemek ve görüntü dönüşümleri yapmak için
from torchvision import datasets, transforms

# Verileri batch batch modele vermek için kullanılır
from torch.utils.data import DataLoader

# Dosya ve klasör işlemleri için
import os


# =========================================================
# DOSYA YOLLARI
# =========================================================

# Veri setinin bulunduğu ana klasör
# İçinde train ve test klasörleri olacak
DATASET_DIR = "dataset"

# Eğitilmiş modelin kaydedileceği klasör
MODEL_DIR = "models"

# Modelin kaydedileceği tam dosya yolu
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fire_smoke_cnn.pth"
)



# EĞİTİM PARAMETRELERİ
# Model aynı anda kaç görüntü işleyecek
# Batch mantığı GPU kullanımını hızlandırır
BATCH_SIZE = 8

# Veri seti modele kaç kez gösterilecek
# 10 epoch = tüm veri seti 10 kez modele gösterilecek
EPOCHS = 10

# Öğrenme hızı
# Çok büyük olursa model kararsız öğrenebilir
# Çok küçük olursa çok yavaş öğrenir
LEARNING_RATE = 0.001

# Tüm görüntüler aynı boyuta çevrilecek
# CNN modelleri sabit boyut sever
IMG_SIZE = 128


# =========================================================
# MODEL KLASÖRÜNÜ OLUŞTUR
# =========================================================

# Eğer "models" klasörü yoksa oluştur
# exist_ok=True:
# klasör varsa hata verme
os.makedirs(MODEL_DIR, exist_ok=True)


# Eğer bilgisayarda NVIDIA GPU varsa CUDA kullanılır
# Yoksa CPU üzerinde çalışır
# Bu satır sistemi otomatik optimize eder
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# CNN modeli görüntüyü direkt anlayamaz
# Önce görüntüler uygun formata çevrilmeli

transform = transforms.Compose([

    # Tüm görüntüler 128x128 yapılır
    # Çünkü CNN tüm görüntülerin aynı boyutta olmasını ister
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Görüntü PyTorch tensor formatına çevrilir
    transforms.ToTensor(),
    # Piksel değerlerini normalize eder
    # Daha stabil öğrenme sağlar
    # CNN’in daha dengeli çalışmasına yardımcı olur
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])



# Eğitim görüntülerini yükle
# ImageFolder klasör isimlerini otomatik sınıf yapar
#
# Örnek yapı:
#
# dataset/
#    train/
#       fire/
#       smoke/
#       normal/

train_dataset = datasets.ImageFolder(

    # Eğitim klasörü
    root=os.path.join(DATASET_DIR, "train"),

    # Görüntü dönüşümleri uygulanacak
    transform=transform
)



# Test görüntülerini yükle
# Model burada ilk kez görmediği veriler üzerinde test edilir
test_dataset = datasets.ImageFolder(

    root=os.path.join(DATASET_DIR, "test"),

    transform=transform
)



# DataLoader:
# Verileri batch halinde modele verir
# RAM kullanımını optimize eder

train_loader = DataLoader(

    # Eğitim verisi
    train_dataset,

    # Aynı anda kaç görüntü işlenecek
    batch_size=BATCH_SIZE,

    # Eğitim sırasında verileri karıştır
    # Ezberlemeyi azaltır
    shuffle=True
)

# Test DataLoader
test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    # Test sırasında karıştırmaya gerek yok
    shuffle=False
)



# nn.Module:
# PyTorch’taki tüm derin öğrenme modellerinin temel sınıfı
class FireSmokeCNN(nn.Module):

    def __init__(self, num_classes):

        # Parent constructor çağrılır
        super(FireSmokeCNN, self).__init__()

        # CNN burada görüntüden:
        #
        # - alev parlaklığı
        # - kırmızı/turuncu yoğunluğu
        # - dumanın gri dağılımı
        # - şekil örüntüleri
        #
        # gibi özellikleri öğrenmeye çalışır

        self.features = nn.Sequential(
            # Conv2D:
            # Görüntü üzerinde filtre gezdirir
            # Kenar, renk ve temel yapıları öğrenir

            nn.Conv2d(

                # RGB görüntü = 3 kanal
                in_channels=3,

                # 16 feature map üretilecek
                out_channels=16,

                # 3x3 filtre boyutu
                kernel_size=3,

                # Padding:
                # görüntü boyutu korunur
                padding=1
            ),

            # ReLU:
            # Negatif değerleri sıfırlar
            # Doğrusal olmayan öğrenme sağlar
            nn.ReLU(),

            # MaxPooling:
            # Görüntü boyutunu küçültür
            # En önemli özellikleri korur
            nn.MaxPool2d(2),

            # =================================================
            # 2. CONVOLUTION KATMANI
            # =================================================

            # Daha karmaşık örüntüler öğrenilir
            # Örneğin:
            # duman yoğunluğu
            # alev yapısı

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),
            # Daha derin özellikler öğrenilir
            # Örneğin:
            # yoğun alev kümeleri
            # duman yayılım yapısı

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

    
        # CNN’den çıkan özellikleri analiz eder
        # ve görüntünün:
        #
        # fire
        # smoke
        # normal
        #
        # sınıflarından hangisi olduğunu belirler

        self.classifier = nn.Sequential(

            # Çok boyutlu veriyi düzleştir
            nn.Flatten(),

            # Tam bağlantılı katman
            nn.Linear(

                # Giriş boyutu
                64 * 16 * 16,

                # Çıkış boyutu
                128
            ),

            nn.ReLU(),

            # Dropout:
            # Ezberlemeyi azaltır
            # Overfitting önler
            nn.Dropout(0.3),

            # Son sınıflandırma katmanı
            nn.Linear(
                128,
                num_classes
            )
        )

    # Veri model içinde nasıl ilerleyecek burada tanımlanır
    def forward(self, x):

        # Önce feature extraction
        x = self.features(x)

        # Sonra classification
        x = self.classifier(x)

        # Sonuç döndürülür
        return x


# =========================================================
# MODEL OLUŞTUR
# =========================================================

# Veri setindeki sınıf sayısını bul
# Örneğin:
# fire / smoke / normal = 3
num_classes = len(train_dataset.classes)

# CNN modelini oluştur
model = FireSmokeCNN(
    num_classes
).to(device)

# Sınıflandırma problemleri için kullanılan loss
# Model ne kadar yanlışsa loss o kadar yükselir
criterion = nn.CrossEntropyLoss()



# Adam optimizer:
# Model ağırlıklarını günceller
optimizer = optim.Adam(

    # Öğrenilecek parametreler
    model.parameters(),

    # Öğrenme hızı
    lr=LEARNING_RATE
)


print("Sınıflar:", train_dataset.classes)
print("Eğitim başlıyor...")


# Epoch kadar tekrar et
for epoch in range(EPOCHS):

    # Model eğitim moduna geçer
    model.train()

    # Toplam loss
    running_loss = 0.0

    # Doğru tahmin sayısı
    correct = 0

    # Toplam görüntü sayısı
    total = 0

    # Batch batch veri al
    for images, labels in train_loader:

        # Verileri GPU/CPU’ya taşı
        images = images.to(device)
        labels = labels.to(device)

        # Önceki gradientleri temizle
        optimizer.zero_grad()

        # Forward pass
        # Görüntüleri modele ver
        outputs = model(images)

        # Modelin ne kadar hata yaptığını hesapla
        loss = criterion(outputs, labels)

        # Backpropagation
        # Hata geriye yayılır
        loss.backward()

        # Model ağırlıkları güncellenir
        optimizer.step()

        # Loss değeri eklenir
        running_loss += loss.item()

        # En yüksek olasılıklı sınıf seçilir
        _, predicted = torch.max(outputs, 1)

        # Toplam örnek sayısı
        total += labels.size(0)

        # Doğru tahmin sayısı artırılır
        correct += (
            predicted == labels
        ).sum().item()

    # Accuracy hesaplanır
    train_acc = 100 * correct / total

    # Epoch sonucu yazdırılır
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Train Accuracy: {train_acc:.2f}%"
    )




# Model test moduna geçer
model.eval()

correct = 0
total = 0

# Gradient hesaplama kapatılır
# Çünkü artık eğitim yapılmıyor
with torch.no_grad():

    # Test verileri dolaşılır
    for images, labels in test_loader:

        # GPU/CPU’ya taşı
        images = images.to(device)
        labels = labels.to(device)

        # Tahmin üret
        outputs = model(images)

        # En yüksek olasılıklı sınıfı seç
        _, predicted = torch.max(outputs, 1)

        # Toplam örnek sayısı
        total += labels.size(0)

        # Doğru tahminleri say
        correct += (
            predicted == labels
        ).sum().item()


# Modelin genel doğruluğu hesaplanır
test_acc = 100 * correct / total

print("Test Accuracy:", test_acc)



# Eğitilen model kaydedilir
torch.save({

    # Modelin öğrendiği ağırlıklar
    "model_state_dict": model.state_dict(),

    # Sınıf isimleri
    "classes": train_dataset.classes

}, MODEL_PATH)

print("Model kaydedildi:", MODEL_PATH)