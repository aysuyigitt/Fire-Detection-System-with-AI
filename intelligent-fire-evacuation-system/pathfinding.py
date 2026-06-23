
# PATHFINDING.PY — GÜVENLI ROTA BULMA (DİJKSTRA ALGORİTMASI)
# Bu dosya projenin en kritik AI bileşenini içerir.
# Dijkstra algoritması ile kişiden en yakın güvenli çıkışa
# en düşük maliyetli (en güvenli) rotayı hesaplar.
# Temel mantık:
#   - Yangın hücreleri tamamen atlanır (maliyet = None)
#   - Duman hücreleri pahalıdır ama geçilebilir (maliyet = 10)
#   - Normal hücreler ucuzdur (maliyet = 1 veya 2)
#   - Algoritma her zaman en ucuz yolu seçer

# Dijkstra algoritması nasıl çalışır?
#   1. Başlangıç noktasını kuyruğa ekle (maliyet=0)
#   2. Kuyruktan en düşük maliyetli hücreyi al
#   3. O hücrenin komşularını incele
#   4. Daha ucuz yol bulunursa güncelle ve kuyruğa ekle
#   5. Çıkışa ulaşınca dur, geriye doğru rotayı oluştur

import heapq   # Min-heap (öncelik kuyruğu) veri yapısı
               # heapq her zaman en küçük değeri önce verir
               # Bu sayede Dijkstra hep en ucuz yolu inceler

# grid.py'dan COST sözlüğü ve EXIT sabiti alınır
# COST: her hücre tipinin geçiş maliyeti
# EXIT: çıkış hücresi tipi (5)
from grid import COST, EXIT


# DİJKSTRA SINIFI
# Bu sınıf tek bir iş yapar: Verilen başlangıç noktasından en yakın güvenli çıkışa en düşük maliyetli rotayı döndürür.

class DijkstraPathFinder:

    # KURUCU METOT
    # BuildingGrid nesnesini alır ve saklar.
    # Tüm hücre bilgileri bu grid üzerinden sorgulanır.
    def __init__(self, building_grid):

        # Grid referansı — hücre tiplerine ve komşulara erişim için
        # self.bg.get(r,c) → hücre tipi
        # self.bg.neighbors(r,c) → geçilebilir komşular
        # self.bg.find(EXIT) → tüm çıkış noktaları
        self.bg = building_grid


    # ROTA BUL
    # Ana metot — başlangıçtan en güvenli çıkışa rota bulur.
    # Parametre:
    #   start : Başlangıç hücresi koordinatı → (satır, sütun)
    #           Örnek: (15, 3) — kişinin bulunduğu hücre
    # Döndürür:[(r,c), (r,c), ...] → başlangıçtan çıkışa koordinat listesi
    # None → güvenli rota bulunamadı (yangın kapattı)
    def find_path(self, start):

        # Grid üzerindeki TÜM çıkış noktaları bulunur
        # Örnek: [(5, 27), (14, 27)] — sağ duvardaki iki çıkış
        # Dijkstra hangisine daha ucuz ulaşırsa onu seçer
        exits = self.bg.find(EXIT)

        # ── ÖNCELIK KUYRUĞU (Priority Queue) ─────────────────
        # Python'un heapq modülü min-heap yapısıdır.
        # Her eleman (maliyet, koordinat) çiftidir.
        # heappop() her zaman en düşük maliyetli elemanı döndürür.
        # Bu sayede Dijkstra her adımda en ucuz yolu inceler.
        pq = []

        # Başlangıç noktası maliyet=0 ile kuyruğa eklenir
        # (0, start) → (maliyet=0, konum=(15,3))
        heapq.heappush(pq, (0, start))

        # Her hücreye hangi hücreden gelindiğini kaydeder.
        # Rota bulununca bu sözlük üzerinden geriye gidilir.
        # Örnek: came_from[(5,27)] = (5,26) → çıkışa sağdan gelindi
        came_from = {}

        # TOPLAM MALİYET
        # Başlangıçtan her hücreye olan en düşük toplam maliyet.
        # Başlangıç noktasının maliyeti sıfırdır.
        # Örnek: cost_so_far[(10,5)] = 14 → o hücreye 14 maliyetle ulaşıldı
        cost_so_far = {start: 0}

        # En iyi (en ucuz) çıkış noktası — başta bilinmiyor
        best_exit = None

        # ANA DİJKSTRA DÖNGÜSÜ-
        # Kuyruk boşalana kadar veya çıkış bulunana kadar devam eder.
        # Her iterasyonda en ucuz hücre incelenir.
        while pq:

            # Kuyruktan en düşük maliyetli hücreyi al
            # current_cost: o hücreye ulaşmanın toplam maliyeti
            # current: o hücrenin koordinatı (satır, sütun)
            current_cost, current = heapq.heappop(pq)

            # ÇIKIŞ KONTROLÜ 
            # Mevcut hücre bir çıkış noktası mı?
            # Dijkstra garantisi: ilk ulaşılan çıkış = en ucuz çıkış
            # Çünkü kuyruk her zaman en düşük maliyetliyi önce verir
            if current in exits:
                best_exit = current   # En iyi çıkışı kaydet
                break                 # Döngüyü sonlandır — iş bitti

            # KOMŞULARI İNCELE 
            # Mevcut hücrenin geçilebilir 4 komşusuna bak
            # neighbors() metodu duvar ve yangın hücrelerini zaten filtreler
            for neighbor in self.bg.neighbors(*current):
                # *current → (satır, sütun) tuple'ını iki ayrı argümana açar
                # neighbors(15, 3) şeklinde çağrılır

                # Komşu hücrenin tipini öğren (EMPTY, ROOM, SMOKE vs.)
                cell_type = self.bg.get(*neighbor)

                # O hücre tipinin hareket maliyetini al
                # Örnek: SMOKE → 10, EMPTY → 1, ROOM → 2
                move_cost = COST[cell_type]

                # Maliyet None ise bu hücre geçilemez
                # WALL ve FIRE maliyeti None'dır
                # neighbors() bunları zaten filtreler ama çift kontrol
                if move_cost is None:
                    continue   # Bu komşuyu atla

                # Mevcut hücrenin maliyeti + komşuya geçiş maliyeti
                # Örnek: 14 (mevcut) + 1 (boş hücre) = 15 (yeni maliyet)
                new_cost = current_cost + move_cost

                # DAHA İYİ YOL KONTROLÜ 
                # Bu komşuya daha önce hiç ulaşılmadıysa (yeni keşif)
                # VEYA şimdiki yol eskisinden daha ucuzsa (iyileştirme)
                # → güncelle ve kuyruğa ekle
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:

                    # Yeni en düşük maliyeti kaydet
                    cost_so_far[neighbor] = new_cost

                    # Komşuyu kuyruğa ekle — ileride incelenecek
                    # Daha düşük maliyetle önce incelenecek
                    heapq.heappush(pq, (new_cost, neighbor))

                    # Bu komşuya mevcut hücreden gelindi — geri iz için kaydet
                    came_from[neighbor] = current

        # SONUÇ KONTROLÜ 
        # Döngü bitti ama hiç çıkış bulunamadıysa:
        # Tüm çıkışlar yangın tarafından kesilmiş demektir
        if best_exit is None:
            return None   # Güvenli rota yok — tahliye mümkün değil

        # ── ROTAYI GERİ İZLE ──────────────────────────────────
        # came_from sözlüğü üzerinden çıkıştan başlangıca gidilir.
        # Rota tersten oluşturulur, sonra ters çevrilir.
        path = []

        # Çıkış noktasından başlayarak geriye git
        current = best_exit

        # Başlangıç noktasına ulaşana kadar devam et
        while current != start:

            path.append(current)        # Mevcut hücreyi rotaya ekle
            current = came_from[current]  # Bir önceki hücreye git
            # came_from[(5,27)] = (5,26) → bir önceki adıma git

        # Başlangıç noktasını da rotaya ekle
        # (while döngüsü start'a eşit olunca durur, onu eklemez)
        path.append(start)

        # Rota şu an [çıkış, ..., başlangıç] sırasındadır
        # reverse() ile [başlangıç, ..., çıkış] sırasına çevir
        path.reverse()

        # Tam rota döndürülür:
        # [(15,3), (15,4), ..., (5,27)] gibi
        # pygame_view.py bunu mavi çizgi olarak çizer
        return path