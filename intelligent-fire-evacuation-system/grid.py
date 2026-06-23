
# GRID.PY — BİNA MODELİ
# Bu dosya projenin temel veri yapısını oluşturur.
# Gerçek bir binayı 2 boyutlu sayı tablosu olarak temsil eder.
# Her hücre bir sayıdır ve o sayı o hücrenin ne olduğunu söyler.
# Örnek: 0=koridor, 1=duvar, 3=yangın gibi.

import numpy as np   # Sayısal dizi işlemleri için (2D grid)
import random        # Kişileri rastgele yerleştirmek için


# HÜCRE TİPİ SABİTLERİ
# Grid içindeki her hücre aşağıdaki sayılardan birini taşır.

EMPTY  = 0   # Boş koridor — kişi serbestçe yürüyebilir
WALL   = 1   # Duvar — geçilemez, yok edilemez
ROOM   = 2   # Oda içi alan — yürünebilir ama biraz daha pahalı
FIRE   = 3   # Yangın — kesinlikle geçilemez, komşulara yayılır
SMOKE  = 4   # Duman — geçilebilir ama tehlikeli (yüksek maliyet)
EXIT   = 5   # Güvenli çıkış kapısı — hedef nokta
PERSON = 6   # Tahliye edilecek kişi — Dijkstra başlangıç noktası

# HAREKET MALİYETLERİ (COST)
# Dijkstra algoritması bir rotanın "ne kadar pahalı" olduğunu bu sözlük ile hesaplar.

# Düşük maliyet = güvenli ve tercih edilen alan
# Yüksek maliyet = tehlikeli, kaçınılması gereken alan
# None = tamamen geçilemez, rota bu hücreden geçemez
# Örnek: Dijkstra dumanlı yoldan gitmek ile 9 adım daha uzun temiz yoldan gitmek arasında tercih yaparsa,temiz yolu seçer çünkü duman maliyeti 10 kat fazla.

COST = {

    EMPTY:  1,     # Koridor — en ucuz geçiş, tercih edilir
    ROOM:   2,     # Oda içi — biraz daha pahalı (mobilya vs.)
    SMOKE:  10,    # Duman — pahalı ama çaresiz kalınırsa geçilir
    EXIT:   1,     # Çıkış — hedefe ulaşmak ucuz ve güvenli
    FIRE:   None,  # Yangın — geçilemez, Dijkstra bu yolu hiç denemez
    WALL:   None,  # Duvar — geçilemez, fiziksel engel
    PERSON: 1,     # Kişi hücresi — kişi üzerinden geçilebilir
}

# BUILDINGGRID SINIFI
# Bu sınıf tüm bina ortamını tek bir yapıda toplar.
# İçinde şunlar bulunur:
#   - Duvarlar ve odalar (statik yapı)
#   - Yangın ve duman bölgeleri (dinamik, yayılır)
#   - Çıkış kapıları (sabit hedef noktalar)
#   - Kişiler (tahliye edilecek ajanlar)
# Diğer modüller (pathfinding, fire_spread, pygame_view)
# bu sınıfı kullanarak bina hakkında bilgi alır ve günceller.

class BuildingGrid:
    # KURUCU METOT (Constructor)
    # Nesne oluşturulduğunda otomatik çalışır.Grid boyutunu alır, boş bir tablo oluşturur, ardından bina planını çizer.
    # Parametreler:
    #   rows: Kaç satır olacak (dikey boyut, default 20)
    #   cols: Kaç sütun olacak (yatay boyut, default 28)
    # --------------------------------------------------------
    def __init__(self, rows: int = 20, cols: int = 28):

        self.rows = rows   # Satır sayısı — bina yüksekliği (hücre cinsinden)
        self.cols = cols   # Sütun sayısı — bina genişliği (hücre cinsinden)
        # Sonuç: 20 satır × 28 sütun = 560 hücreli boş bir tablo
        self.grid = np.zeros((rows, cols), dtype=int)
        # Bina planını çiz: duvarlar, odalar, yangın, kişiler...
        self._build_default_layout()

    # BİNA PLANINI OLUŞTUR: Boş gridde tüm yapıyı inşa eder.
    # Sırasıyla şunları yapar:
    #   1. Çevre duvarlarını çizer
    #   2. Odaları doldurur
    #   3. İç duvarları koyar
    #   4. Kapı geçişlerini açar
    #   5. Yangın ve duman bölgelerini yerleştirir
    #   6. Çıkış kapılarını ekler
    #   7. Kişileri rastgele yerleştirir
    def _build_default_layout(self):
        g = self.grid 
        g[:, :] = EMPTY   # Önce tüm grid boşaltılır (temiz başlangıç)

        # ── 1. ÇEVRE DUVARLARI ──────────────────────────────
        # Binanın dört kenarı duvar yapılır.
        # g[0, :] → en üst satırın tüm sütunları
        # g[-1, :] → en alt satırın tüm sütunları (-1 = son indeks)
        # g[:, 0] → tüm satırların en sol sütunu
        # g[:, -1] → tüm satırların en sağ sütunu
        g[0, :]  = WALL   # Üst duvar
        g[-1, :] = WALL   # Alt duvar
        g[:, 0]  = WALL   # Sol duvar
        g[:, -1] = WALL   # Sağ duvar

        # ── 2. ODA ALANLARI ──────────────────────────────────
        # Bina 5 odaya bölünmüştür:
        # Sol üst, Sol alt, Sağ üst, Sağ alt, Büyük sağ salon
        # g[satır_başlangıç:satır_bitiş, sütun_başlangıç:sütun_bitiş]
        # Python dilimleme: bitiş değeri dahil değil (1:9 → 1,2,3,4,5,6,7,8)
        g[1:9,   1:9]  = ROOM   # Sol üst oda (satır 1-8, sütun 1-8)
        g[11:19, 1:9]  = ROOM   # Sol alt oda (satır 11-18, sütun 1-8)
        g[1:9,   10:18] = ROOM  # Sağ üst oda (satır 1-8, sütun 10-17)
        g[11:19, 10:18] = ROOM  # Sağ alt oda (satır 11-18, sütun 10-17)
        g[1:19,  19:27] = ROOM  # Büyük sağ salon (satır 1-18, sütun 19-26)

        # ── 3. İÇ DUVARLAR ───────────────────────────────────
        # Odaları birbirinden ayıran duvarlar.
        # Kapı hücreleri sonradan EMPTY yapılacak,
        # bu yüzden önce tüm bölücü satır/sütunlar WALL yapılır.

        # Sol bölücü duvar (sütun 9) — üst ve alt oda arasını ayırır
        # Ama kapı yerleri için aralıklı duvar yapılır (4:6 arası boş)
        g[1:4,   9] = WALL   # Sütun 9, satır 1-3 (üst kısım)
        g[6:9,   9] = WALL   # Sütun 9, satır 6-8 (alt kısım)
        g[11:14, 9] = WALL   # Sütun 9, satır 11-13
        g[16:19, 9] = WALL   # Sütun 9, satır 16-18

        # Yatay iç duvar (satır 9) — üst ve alt oda bloklarını ayırır
        g[9, 10:12] = WALL   # Satır 9, sütun 10-11
        g[9, 16:18] = WALL   # Satır 9, sütun 16-17

        # Sağ bölücü duvar (sütun 18) — sağ oda ile büyük salon arası
        g[1:4,   18] = WALL  # Sütun 18, satır 1-3
        g[6:9,   18] = WALL  # Sütun 18, satır 6-8
        g[11:14, 18] = WALL  # Sütun 18, satır 11-13
        g[16:19, 18] = WALL  # Sütun 18, satır 16-18

        # Büyük salondaki iç bölücüler
        g[4:6,   22] = WALL  # Satır 4-5, sütun 22
        g[13:15, 23] = WALL  # Satır 13-14, sütun 23

        # 4. KAPI GEÇİŞLERİ
        # Duvarlar içinde açılan kapı boşlukları.
        # Bu hücreler EMPTY yapılarak odalar arası geçiş sağlanır.
        # Kişiler ve algoritma bu hücrelerden geçer.
        door_cells = [
            (4, 9),  (5, 9),    # Sol duvar kapısı — üst oda girişi
            (9, 9),  (10, 9),   # Sol duvar kapısı — orta geçiş
            (14, 9), (15, 9),   # Sol duvar kapısı — alt oda girişi

            (4, 18),  (5, 18),  # Sağ duvar kapısı — üst oda girişi
            (9, 18),  (10, 18), # Sağ duvar kapısı — orta geçiş
            (14, 18), (15, 18), # Sağ duvar kapısı — alt oda girişi

            (9, 12), (9, 13),   # Yatay duvar kapısı — üst-alt geçiş
            (9, 14), (9, 15),   # Yatay duvar kapısı devamı

            (8, 22), (9, 22), (10, 22),    # Büyük salon sol kapısı
            (8, 23), (9, 23), (10, 23),    # Büyük salon sol kapısı (geniş)
        ]

        # Listedeki her kapı hücresi EMPTY yapılır → geçiş açılır
        for r, c in door_cells:
            g[r, c] = EMPTY

        # 5. YANGINN VE DUMAN 
        # Başlangıç yangın noktaları — sağ üst oda içinde
        # Bu hücreler FIRE olduğu için Dijkstra oradan geçmez
        # FireSpreadSimulator bu hücrelerden yayılmaya başlar
        g[7, 12] = FIRE   # Yangın merkezi
        g[7, 13] = FIRE   # Yangın sağa doğru
        g[8, 12] = FIRE   # Yangın aşağı doğru

        # Başlangıç duman bölgeleri — yangının hemen çevresinde
        # Duman geçilebilir ama maliyeti 10 kat fazla
        g[6, 12] = SMOKE  # Yangının üstü duman
        g[7, 14] = SMOKE  # Yangının sağı duman
        g[9, 12] = SMOKE  # Yangının altı duman

        # 6. ÇIKIŞ NOKTALARI
        # Binanın sağ duvarında iki çıkış kapısı var.
        # Dijkstra bu noktalara ulaşmaya çalışır.
        # Sütun 27 = en sağ sütun (sınır duvarı üzerinde)
        g[5,  27] = EXIT   # Üst çıkış — satır 5, en sağ kolon
        g[14, 27] = EXIT   # Alt çıkış — satır 14, en sağ kolon

        #  7. KİŞİLERİ YERLEŞTİR
        # 6 kişi bina içine rastgele ama akıllıca yerleştirilir.
        # Yangına, duvara ve çıkışa çok yakın yerlere konulmaz.
        self._spawn_random_people(count=6)

    # KİŞİLERİ AKILLI RASTGELE YERLEŞTİR: Tahliye edilecek kişilerin bina içerisine gerçekçi ve  güvenli bir şekilde yerleştirilmesini sağlamaktadır. .
    # Bu metot şu kuralları uygular:
    #   - Yangın üstüne konulma
    #   - Duvara konulma
    #   - Çıkışa çok yakın konulma (zaten güvendeler)
    #   - Yangına sıfır mesafede konulma
    #   - Kişiler birbirine çok yakın konulma
    # Önce "riskli ama mantıklı" yerlere koyar (eğitici senaryo için)
    def _spawn_random_people(self, count=6):

        g = self.grid # grid referansı alınıyor. Böylece bina haritası üzerinde işlem yapılabiliyor.

        exits      = self.find(EXIT)    # binadaki tüm çıkış kapılarının koordinatları 
        fire_cells = self.find(FIRE)    # yangının bulunduğu hücrelerin koordinatları

        candidates = [] # Uygun tüm hücreler

        # Tüm bina hücreleri tek tek geziliyor.
        # sadece binanın iç alanı kontrol ediliyor.
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):

                cell = g[r, c]

                # Sadece boş alan ve oda hücreleri aday kabul ediliyor. 
                # Duvar, yangın, duman, çıkış hücrelerine konulmaz
                if cell not in (EMPTY, ROOM):
                    continue

                # Çıkışa çok yakın yerler eleniyor.
                # Gerçek mesafe yerine kullanılır, daha hızlı hesaplanır
                too_close_to_exit = False
                for er, ec in exits:
                    if abs(er - r) + abs(ec - c) < 5: # Çıkışa çok yakın yerler eleniyor.
                        # Eğer kişi çıkışa 5 hücreden daha yakınsa o konum uygun kabul edilmiyor
                        too_close_to_exit = True
                        break
                if too_close_to_exit:
                    continue   # Bu hücreyi atla

                # Yangına çok yakın yerler eleniyor. 
                # Yangına sıfır veya bir hücre uzaklıkta başlama
                too_close_to_fire = False
                for fr, fc in fire_cells:
                    if abs(fr - r) + abs(fc - c) < 2:
                        #Eğer kişi yangına 2 hücreden daha yakınsa o hücre eleniyor. 
                        too_close_to_fire = True
                        break
                if too_close_to_fire:
                    continue   # Bu hücreyi atla

                # Her iki kontrolü de geçen hücre aday listesine girer
                candidates.append((r, c))

        # Adayları karıştır — her çalıştırmada farklı yerleşim
        random.shuffle(candidates)

        selected = []          # Seçilen kişi konumları
        risky_candidates = []  # Yangına orta mesafedeki adaylar

        # ── Riskli ama mantıklı adayları filtrele ─────────────
        # Yangına 3-9 hücre mesafedeki hücreler seçilir.
        # Çok uzaklar senaryo için ilgisiz, çok yakınlar ölümcül.
        for r, c in candidates:

            # Bu hücrenin en yakın yangına olan mesafesi
            min_fire_dist = min(
                abs(fr - r) + abs(fc - c)
                for fr, fc in fire_cells
            )

            # 3 ile 9 hücre arası = tehlikeli ama tahliye edilebilir. Yani kişi yangından ne çok uzak ne de çok yakın olacak şekilde seçilmeye çalışılıyor.
            if 3 <= min_fire_dist <= 9:
                risky_candidates.append((r, c))

        random.shuffle(risky_candidates)   # Karıştır

        # ── Önce riskli adaylardan seç ────────────────────────
        for pos in risky_candidates:
            if len(selected) >= count:
                break
            # Diğer kişilerden yeterince uzaksa seç
            if self._far_enough_from_people(pos, selected):
                selected.append(pos)

        # Eksik kalırsa diğer adaylardan tamamla
        #yeni kişinin daha önce seçilen kişilere çok yakın olup olmadığı kontrol ediliyor. 
        # Eğer kişi diğerlerinden yeterince uzaktaysa selected.append(pos) ile seçilen kişiler listesine ekleniyor.
        for pos in candidates:
            if len(selected) >= count:
                break
            if self._far_enough_from_people(pos, selected):
                selected.append(pos)

        # Seçilen hücreleri PERSON yap
        for r, c in selected:
            g[r, c] = PERSON   # Grid'e kişi yerleştir


    # KİŞİLER ARASI MİNİMUM MESAFE KONTROLÜ
    # Fonksiyonun amacı kişilerin aynı bölgede kümelenmesini engellemek ve bina içerisinde
    # daha gerçekçi bir dağılım oluşturmaktır
    # Parametreler:
    #   pos      : Kontrol edilecek hücre koordinatı (r, c)
    #   selected : Daha önce seçilmiş kişi koordinatları
    #   min_dist : Minimum gereken mesafe (Manhattan)
    def _far_enough_from_people(self, pos, selected, min_dist=2):
        r, c = pos   # Kontrol edilecek hücrenin koordinatları

        # Daha önce yerleştirilmiş her kişiyle mesafeyi kontrol et
        for sr, sc in selected:

            # yeni kişinin koordinatları ile daha önce yerleştirilen kişilerin koordinatları karşılaştırılır ve Manhattan mesafesi hesaplanır
            if abs(sr - r) + abs(sc - c) < min_dist:
                return False   # Çok yakın — uygun değil

        return True   # Tüm kişilerden yeterince uzak — uygun

    # HÜCRE GÜNCELLE
    # Fonksiyon önce verilen koordinatların bina sınırları içerisinde olup olmadığını kontrol eder.
    # Eğer koordinatlar geçerliyse ilgili hücre yeni hücre tipi ile güncellenir.
    def set(self, row: int, col: int, cell_type: int):

        # Önce grid sınırları içinde mi kontrol et
        if self._in_bounds(row, col):
            self.grid[row, col] = cell_type   # Hücreyi güncelle


    # HÜCRE TİPİNİ AL : Belirli bir hücrenin mevcut tipini döndürür.
    # Fonksiyona satır ve sütun bilgisi verildiğinde, ilgili hücrede hangi nesnenin bulunduğunu döndürür
    def get(self, row: int, col: int) -> int:
        return int(self.grid[row, col])

    # find() fonksiyonu grid üzerinde belirli bir hücre tipine ait tüm koordinatları bulmaktadır
    # Yangın, çıkış ve kişi konumlarının tespit edilmesinde kullanılmaktadır
    def find(self, cell_type: int):
        positions = np.argwhere(self.grid == cell_type)
        return [tuple(p) for p in positions]

    # HÜCREDEKİ GEÇİŞ MÜMKÜN MÜ?
    # COST sözlüğünde None olmayan hücreler geçilebilir.
    def is_passable(self, row: int, col: int) -> bool:
        return COST.get(self.get(row, col)) is not None

    # KOORDİNAT GRID İÇİNDE Mİ?
    # Verilen satır ve sütun grid sınırları içinde mi kontrol eder.
    # Komşu bulma gibi işlemlerde sınır taşmasını önler.
    def _in_bounds(self, row: int, col: int) -> bool:
        return (
            0 <= row < self.rows   # Satır 0 ile son satır arasında mı?
            and
            0 <= col < self.cols   # Sütun 0 ile son sütun arasında mı?
        )

    # GEÇİLEBİLİR KOMŞU HÜCRELERİ BUL
    # Bir hücrenin 4 komşusuna bakar (yukarı, aşağı, sol, sağ).
    # Çapraz hareket yok — gerçekçi bina simülasyonu için.
    # Dijkstra algoritması her adımda bu metodu çağırır.
    def neighbors(self, row: int, col: int):
        # 4 yön: (satır_değişimi, sütun_değişimi)
        dirs = [
            (-1,  0),   # Yukarı — satır azalır
            ( 1,  0),   # Aşağı  — satır artar
            ( 0, -1),   # Sol    — sütun azalır
            ( 0,  1),   # Sağ    — sütun artar
        ]

        result = []   # Geçilebilir komşuların listesi

        for dr, dc in dirs:

            nr = row + dr   # Komşunun satırı
            nc = col + dc   # Komşunun sütunu

            # Grid içinde mi VE geçilebilir mi?
            if self._in_bounds(nr, nc) and self.is_passable(nr, nc):
                result.append((nr, nc))   # Uygun komşu listeye eklenir

        return result   # Dijkstra bu listeyi kullanır



    # NESNE YAZDIRILDIĞINDA GÖSTER : print(bina) yazıldığında ekrana ne çıkacağını belirler.
    def __repr__(self):
        return f"BuildingGrid({self.rows}x{self.cols})"