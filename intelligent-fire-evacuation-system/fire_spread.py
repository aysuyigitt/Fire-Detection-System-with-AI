import random

# Grid dosyasındaki hücre tipleri alınır
from grid import EMPTY, WALL, ROOM, FIRE, SMOKE, EXIT, PERSON


# Hücrelerin yanma olasılıkları
# Her hücre türü yangına farklı direnç gösterir
BURN_PROBABILITY = {

    # Boş alanın yanma ihtimali
    EMPTY: 0.025,
    # Oda alanı daha kolay yanabilir
    ROOM: 0.055,
    # Dumanlı alanın aleve dönüşme ihtimali daha yüksektir
    SMOKE: 0.12,
    # Kişi hücresi doğrudan yangına çevrilmez
    PERSON: 0.0,
    # Duvar yanmaz
    WALL: 0.0,
    # Çıkış korunur
    EXIT: 0.0,
    # Zaten yangın olan hücre tekrar işlenmez
    FIRE: 0.0,
}


# Yangın çevresinde yeni duman oluşma olasılığı
SMOKE_PROBABILITY = 0.18
# Duman hücresinin zamanla aleve dönüşme olasılığı
SMOKE_TO_FIRE_PROBABILITY = 0.025

# bina içerisindeki yangın ve duman yayılımını simüle etmek için oluşturulmuştur.
class FireSpreadSimulator:

    def __init__(self, building_grid):

        # Grid referansı alınır
        # Böylece bina üzerindeki hücreler güncellenebilir
        self.bg = building_grid
    
    # yangının bir adımda nasıl yayılacağını hesaplayan ana fonksiyondur.
    def spread_fire(self):

        # Grid matrisi alınır
        g = self.bg.grid

        # Grid boyutları alınır
        rows, cols = self.bg.rows, self.bg.cols

        # Yeni oluşacak yangın hücreleri burada tutulur
        new_fires = []

        # Yeni oluşacak duman hücreleri burada tutulur
        new_smokes = []

        # Yangının yayılacağı yönler
        # Yukarı, aşağı, sağ, sol
        fire_dirs = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        # Dumanın yayılacağı yönler
        # Duman çaprazlara da yayılabilir
        smoke_dirs = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),

            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1)
        ]

        # Grid üzerindeki tüm hücreler dolaşılır
        for r in range(rows):
            for c in range(cols):

                # Eğer incelenen hücre yangın hücresi değilse
                if int(g[r, c]) != FIRE:
                    continue

                # YANGIN YAYILIMI
                # Yangının dört yönündeki komşular kontrol edilir
                for dr, dc in fire_dirs:

                    # Burada yangının etrafındaki hücrelerin koordinatları bulunuyor.
                    nr, nc = r + dr, c + dc

                    # Harita sınırları dışındaysa geç
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue

                    # Komşu hücre tipi alınır
                    neighbor = int(g[nr, nc])

                    # yangının geçemeyeceği hücreler filtreleniyor
                    if neighbor in (WALL, EXIT, FIRE, PERSON):
                        continue

                    # Hücrenin yanma ihtimali alınıyor
                    burn_prob = BURN_PROBABILITY.get(neighbor, 0.3)

                    # Rastgele sayı olasılıktan küçükse yangın yayılır
                    if random.random() < burn_prob:

                        # Yeni yangın listesine eklenir
                        new_fires.append((nr, nc))

                # DUMAN YAYILIMI
                # Dumanın 8 yönündeki komşular kontrol edilir
                for dr, dc in smoke_dirs:

                    nr, nc = r + dr, c + dc

                    # Grid dışındaki hücreler değerlendirmeye alınmıyor.
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue

                    # Komşu hücre tipi alınır
                    neighbor = int(g[nr, nc])

                    # Dumanın geçemeyeceği hücreler eleniyor
                    if neighbor in (
                        WALL,
                        EXIT,
                        FIRE,
                        SMOKE,
                        PERSON
                    ):
                        continue

                    # Rastgele sayı belirlenen olasılıktan küçükse
                    # yeni duman oluşur
                    if random.random() < SMOKE_PROBABILITY:

                        # Yeni duman listesine eklenir
                        new_smokes.append((nr, nc))


        # DUMAN -> YANGIN DÖNÜŞÜMÜ
        # Binada duman olan yerleri bulmak.
        for r in range(rows):
            for c in range(cols):

                # Duman hücreleri seçiliyor
                if int(g[r, c]) == SMOKE:

                    # Belirli olasılıkla yangına dönüşür
                    if random.random() < SMOKE_TO_FIRE_PROBABILITY:

                        new_fires.append((r, c))


        #  Hesaplanan tüm yeni yangınlar geziliyor
        for r, c in new_fires:

            # Duvar, çıkış ve kişi hücreleri korunur
            if int(g[r, c]) not in (WALL, EXIT, PERSON):

                # Hücre yangına çevrilir
                self.bg.set(r, c, FIRE)

        # Hesaplanan tüm yeni dumalar geziliyor
        for r, c in new_smokes:
            # Yangın, duvar, çıkış ve kişi üstüne duman yayılmaz
            if int(g[r, c]) not in (FIRE, WALL, EXIT, PERSON):

                # Hücre duman yapılıyor
                self.bg.set(r, c, SMOKE)

    def get_fire_danger(self):

        # Toplam hücre sayısı
        total = self.bg.rows * self.bg.cols

        # Yangın hücrelerinin sayısı
        fire_cnt = len(self.bg.find(FIRE))

        # Duman hücrelerinin sayısı
        smoke_cnt = len(self.bg.find(SMOKE))

        # Genel risk seviyesi hesaplanır
        # Yangın tam risk
        # Duman yarım risk etkisi oluşturur
        danger = (fire_cnt + smoke_cnt * 0.5) / total

        # Risk değeri maksimum 1 olabilir
        return min(1.0, danger)