import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))

import pygame
from pathfinding import DijkstraPathFinder
from camera_classifier import predict_image
from fire_spread import FireSpreadSimulator
from grid import BuildingGrid, EMPTY, WALL, ROOM, FIRE, SMOKE, EXIT, PERSON

# ── Sabitler ──────────────────────────────────────────────────────
CELL    = 28      # Daha kompakt ve panel ekrana daha rahat sığar
FPS     = 30
PANEL_W = 260     # Sağ panel daha kompakt
TOP_H   = 46
BOT_H   = 54

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")

C = {
    "BG":          (12, 14, 18),
    "TOP":         (15, 18, 24),
    "TOP_BORDER":  (35, 50, 65),
    "PANEL":       (15, 18, 24),
    "CARD":        (22, 28, 38),
    "CARD_BOR":    (45, 65, 85),
    "TEXT":        (220, 228, 238),
    "MUTED":       (130, 148, 165),
    "GREEN":       (80, 220, 100),
    "GREEN_DIM":   (30, 100, 45),
    "BLUE":        (80, 210, 255),
    "BLUE_GLOW":   (35, 135, 190),
    "RED":         (230, 50, 30),
    "ORANGE":      (255, 125, 30),
    "YELLOW":      (255, 215, 40),
    # Zemin
    "FLOOR":       (46, 45, 40),
    "FLOOR_TILE":  (58, 55, 49),
    "ROOM_FLOOR":  (42, 39, 35),
    "ROOM_WALL":   (105, 102, 94),
    "WALL_C":      (92, 95, 92),
    "WALL_LT":     (185, 190, 182),
    "WALL_DK":     (18, 20, 19),
    "DOOR":        (155, 118, 58),
}

# ── Mobilya tanımları (oda bazlı) ─────────────────────────────────
FURNITURE = {
    "desk":   {"w": 52, "h": 30, "color": (72, 60, 46), "edge": (105, 88, 65)},
    "chair":  {"w": 20, "h": 20, "color": (55, 48, 38), "edge": (80, 70, 55)},
    "plant":  {"w": 16, "h": 16, "color": (35, 110, 50), "edge": (25, 80, 35)},
    "shelf":  {"w": 60, "h": 14, "color": (85, 68, 48), "edge": (115, 92, 65)},
    "sofa":   {"w": 70, "h": 28, "color": (68, 55, 45), "edge": (95, 78, 60)},
    "table":  {"w": 44, "h": 28, "color": (78, 62, 45), "edge": (108, 88, 62)},
}


class PygameView:
    def __init__(self, building_grid,
                 title="Akıllı Yangın Tahmin ve Tahliye Sistemi"):
        self.bg   = building_grid
        self.title = title
        self.tick  = 0
        self.auto_play  = True
        self.step_count = 0
        self.replan_count = 0
        self.message = "Sistem hazır. Güvenli rota hesaplandı."

        self.pathfinder     = DijkstraPathFinder(self.bg)
        self.fire_simulator = FireSpreadSimulator(self.bg)

        self.path_list    = []
        self.path         = set()
        self.fire_particles = []
        self.smoke_puffs  = []

        self.sensor_positions = [(2, 2), (12, 2), (2, 11), (12, 11), (4, 22), (15, 23)]
        self.sensor_range = 3
        self.sensor_status = []
        self.alarm_active = False
        self.alarm_message = "Sensörler normal."
        self.camera_result = "Henüz analiz yok"
        self.camera_image_path = "camera_samples/sample.jpg"
        self.camera_interval = 3
        self.camera_zones = []
        self.camera_zone_results = [
            ("Kamera-1", "NORMAL"),
            ("Kamera-2", "NORMAL"),
            ("Kamera-3", "NORMAL"),
        ]
        self.camera_temp_dir = "camera_samples"
        os.makedirs(self.camera_temp_dir, exist_ok=True)


        self.map_w = self.bg.cols * CELL
        self.map_h = self.bg.rows * CELL
        self.win_w = self.map_w + PANEL_W
        self.win_h = TOP_H + self.map_h + BOT_H

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock  = pygame.time.Clock()

        self.camera_zones = [
            {
                "name": "Kamera-1",
                "pos": (2 * CELL, TOP_H + 2 * CELL),
                "points": [
                    (3 * CELL, TOP_H + 2 * CELL),
                    (9 * CELL, TOP_H + 1 * CELL),
                    (7 * CELL, TOP_H + 6 * CELL),
                ],
                "rect": pygame.Rect(3 * CELL, TOP_H + 1 * CELL, 6 * CELL, 7 * CELL),
            },
            {
                "name": "Kamera-2",
                "pos": (10 * CELL, TOP_H + 6 * CELL),
                "points": [
                    (11 * CELL, TOP_H + 7 * CELL),
                    (18 * CELL, TOP_H + 4 * CELL),
                    (18 * CELL, TOP_H + 12 * CELL),
                ],
                "rect": pygame.Rect(11 * CELL, TOP_H + 4 * CELL, 7 * CELL, 8 * CELL),
            },
            {
                "name": "Kamera-3",
                "pos": (21 * CELL, TOP_H + 12 * CELL),
                "points": [
                    (22 * CELL, TOP_H + 13 * CELL),
                    (27 * CELL, TOP_H + 10 * CELL),
                    (27 * CELL, TOP_H + 18 * CELL),
                ],
                "rect": pygame.Rect(21 * CELL, TOP_H + 10 * CELL, 7 * CELL, 8 * CELL),
            },
        ]


        self.f_title = pygame.font.SysFont("Arial", 18, bold=True)
        self.f_head  = pygame.font.SysFont("Arial", 14, bold=True)
        self.f_body  = pygame.font.SysFont("Arial", 13)
        self.f_small = pygame.font.SysFont("Arial", 11)

        self.images = self._load_images()
        self._build_floor_cache()
        self._recalculate_path()

    # ── Görsel yükleme ────────────────────────────────────────────
    def _load_images(self):
        def load(name):
            p = os.path.join(IMAGE_DIR, name)
            try:
                img = pygame.image.load(p).convert_alpha()
                return img
            except Exception:
                return None
        return {
            "fire":   load("fire-removebg-preview.png"),
            "person": None,   # Kişi artık resim değil, ajan olarak çizilecek
            "exit":   load("exit.jpg"),
        }

    # ── Zemin dokusunu önceden oluştur ────────────────────────────
    def _build_floor_cache(self):
        """Karo desenli zemin yüzeyi — her çizimde yeniden üretmemek için."""
        self.floor_surf = pygame.Surface((self.map_w, self.map_h))
        s = self.floor_surf

        # Tüm yüzeyi temel renkle doldur
        s.fill(C["FLOOR"])

        # Karo çizgileri
        for r in range(self.bg.rows):
            for c in range(self.bg.cols):
                x, y = c * CELL, r * CELL
                cell = int(self.bg.grid[r, c])

                if cell == WALL:
                    self._draw_wall_to(s, x, y)
                else:
                    base = C["ROOM_FLOOR"] if cell == ROOM else C["FLOOR_TILE"]

                    # Daha doğal zemin: grid çizgisi çok baskın değil
                    shade = ((r * 7 + c * 5) % 10) - 5
                    floor_col = (
                        max(0, min(255, base[0] + shade)),
                        max(0, min(255, base[1] + shade)),
                        max(0, min(255, base[2] + shade))
                    )
                    pygame.draw.rect(s, floor_col, (x+1, y+1, CELL-2, CELL-2))

                    # Çok hafif karo çizgisi
                    pygame.draw.rect(s, (base[0]+6, base[1]+6, base[2]+6),
                                     (x+1, y+1, CELL-2, CELL-2), 1)

                    # Zemine küçük nokta detayı
                    if (r + c) % 3 == 0:
                        pygame.draw.circle(s, (base[0]+16, base[1]+16, base[2]+16),
                                           (x + CELL//2, y + CELL//2), 1)

        # Odaları büyük blok olarak çiz (üzerine)
        room_blocks = [
            (1,  1,  8, 8,  "ODA 1"),
            (11, 1,  8, 8,  "ODA 2"),
            (1,  10, 8, 8,  "ODA 3"),
            (11, 10, 8, 8,  "ODA 4"),
            (1,  19, 18, 8, "BÜYÜK SALON"),
        ]
        for r, c, h, w, lbl in room_blocks:
            x, y = c*CELL, r*CELL
            rct = pygame.Rect(x, y, w*CELL, h*CELL)
            pygame.draw.rect(s, C["ROOM_FLOOR"], rct)
            pygame.draw.rect(s, C["ROOM_WALL"],  rct, 3)

            # Oda içi karo desen
            for ti in range(0, w*CELL, CELL):
                for tj in range(0, h*CELL, CELL):
                    pygame.draw.rect(s, (C["ROOM_FLOOR"][0]+10,
                                        C["ROOM_FLOOR"][1]+10,
                                        C["ROOM_FLOOR"][2]+10),
                                     (x+ti, y+tj, CELL, CELL), 1)
                    if (ti // CELL + tj // CELL) % 4 == 0:
                        pygame.draw.circle(s, (C["ROOM_FLOOR"][0]+18,
                                               C["ROOM_FLOOR"][1]+18,
                                               C["ROOM_FLOOR"][2]+18),
                                           (x+ti+CELL//2, y+tj+CELL//2), 1)

            # Etiket
            txt = self.f_small.render(lbl, True, (110, 100, 88))
            s.blit(txt, (x+8, y+6))

            # Mobilyalar
            self._draw_furniture_to(s, rct, lbl)

        # Kapılar
        for (dr, dc) in [(9,4),(10,4),(9,13),(10,13),(9,22),(10,22),
                          (5,9),(14,9),(5,18),(14,18)]:
            x, y = dc*CELL, dr*CELL
            pygame.draw.rect(s, C["DOOR"],
                             (x+5, y+CELL//2-4, CELL-10, 8), border_radius=3)

    def _draw_wall_to(self, surf, x, y):
        pygame.draw.rect(surf, C["WALL_C"], (x, y, CELL, CELL))
        pygame.draw.line(surf, C["WALL_LT"], (x,y),   (x+CELL,y),   4)
        pygame.draw.line(surf, C["WALL_LT"], (x,y),   (x,y+CELL),   3)
        pygame.draw.line(surf, C["WALL_DK"], (x,y+CELL-1),(x+CELL,y+CELL-1), 4)
        pygame.draw.line(surf, C["WALL_DK"], (x+CELL-1,y),(x+CELL-1,y+CELL), 3)
        inner = pygame.Rect(x+6, y+6, CELL-12, CELL-12)
        pygame.draw.rect(surf, (112, 115, 110), inner, border_radius=2)

    def _draw_furniture_to(self, surf, room_rect, label):
        random.seed(hash(label))
        rx, ry, rw, rh = room_rect

        layouts = {
            "ODA 1": [("desk",30,30),("chair",90,35),("plant",rw-30,20),
                      ("shelf",20,rh-25),("desk",rw-80,rh-50)],
            "ODA 2": [("sofa",25,25),("table",rw//2-22,rh//2-14),
                      ("chair",rw-50,30),("plant",20,rh-30)],
            "ODA 3": [("desk",20,25),("desk",rw-80,25),("chair",rw//2-10,rh-45),
                      ("plant",rw-25,rh-28)],
            "ODA 4": [("table",rw//2-22,rh//2-14),("chair",25,25),
                      ("chair",rw-45,25),("shelf",15,rh-25)],
            "BÜYÜK SALON": [
                ("sofa",30,35),("table",rw//2-22,rh//2-14),
                ("desk",rw-90,30),("plant",rw-25,rh-30),
                ("chair",rw-50,rh-50),("shelf",20,rh-25),
            ],
        }

        for fname, ox, oy in layouts.get(label, []):
            f = FURNITURE[fname]
            fx = rx + min(ox, rw - f["w"] - 5)
            fy = ry + min(oy, rh - f["h"] - 5)
            frect = pygame.Rect(fx, fy, f["w"], f["h"])
            pygame.draw.rect(surf, f["color"], frect, border_radius=3)
            pygame.draw.rect(surf, f["edge"],  frect, 1, border_radius=3)

            # Masa üzeri detay
            if fname in ("desk", "table"):
                pygame.draw.rect(surf, (f["color"][0]+20, f["color"][1]+18,
                                        f["color"][2]+14),
                                 (fx+4, fy+4, f["w"]-8, f["h"]-8), border_radius=2)

    # ── Rota ─────────────────────────────────────────────────────
    def _recalculate_path(self):
        persons = self.bg.find(PERSON)

        if not persons:
            self.path_list = []
            self.path = set()
            return

        current_person = persons[0]

        old_path = list(self.path_list)

        danger_found = False

        for r, c in old_path:
            cell = self.bg.get(r, c)

            if cell in (FIRE,):
                danger_found = True
                break

            if cell == SMOKE:
                smoke_neighbors = 0

                for nr, nc in self.bg.neighbors(r, c):
                    if self.bg.get(nr, nc) == FIRE:
                        smoke_neighbors += 1

                if smoke_neighbors >= 2:
                    danger_found = True
                    break

        new_path = self.pathfinder.find_path(current_person)

        self.path_list = new_path if new_path else []
        self.path = set(self.path_list)

        if old_path:
            if old_path != self.path_list:
                self.replan_count += 1

                if danger_found:
                    self.message = (
                        "Tehlike algılandı. Yeni güvenli rota oluşturuldu."
                    )

    # ── Ana döngü ─────────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            self.tick += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.auto_play = not self.auto_play
                        self.message = ("Otomatik oynatma başladı."
                                        if self.auto_play else "Durakladı.")
                    elif event.key == pygame.K_a:
                        self._step()
                    elif event.key == pygame.K_r:
                        self.bg = BuildingGrid()
                        self.pathfinder     = DijkstraPathFinder(self.bg)
                        self.fire_simulator = FireSpreadSimulator(self.bg)
                        self.step_count = 0; self.replan_count = 0
                        self.fire_particles = []; self.smoke_puffs = []
                        self._build_floor_cache()
                        self._recalculate_path()
                        self.message = "Sıfırlandı."

            if self.auto_play and self.tick % 85 == 0:
                self._step()

            self._update_particles()
            self._draw()

        pygame.quit(); sys.exit()
    # Yangını yayar, sensörleri günceller, güvenli rotayı hesaplar, kişiyi rotada bir adım ilerletir ve 
    # rota değişmiş mi diye tekrar kontrol eder. 
    def _step(self):
        self.step_count += 1

        self.fire_simulator.spread_fire()
        self._update_sensors()
        self._recalculate_path()
        self._move_person_one_step()
        self._recalculate_path()

        if self.path_list:
            self.message = f"Adım {self.step_count}: Tahliye rotası güncellendi."
        else:
            self.message = f"Adım {self.step_count}: ROTA BULUNAMADI!"
    
    # “Sensör sistemi, belirlenen noktalarda çevresindeki hücreleri tarayarak yangın veya duman algılamaktadır
    def _update_sensors(self):
        self.sensor_status = []
        self.alarm_active = False
        fire_count = 0
        smoke_count = 0

        for sr, sc in self.sensor_positions:
            status = "NORMAL"

            for dr in range(-self.sensor_range, self.sensor_range + 1):
                for dc in range(-self.sensor_range, self.sensor_range + 1):
                    nr = sr + dr
                    nc = sc + dc

                    if not self.bg._in_bounds(nr, nc):
                        continue

                    cell = self.bg.get(nr, nc)

                    if cell == FIRE:
                        status = "FIRE"
                        fire_count += 1
                    elif cell == SMOKE and status != "FIRE":
                        status = "SMOKE"
                        smoke_count += 1

            if status in ("FIRE", "SMOKE"):
                self.alarm_active = True

            self.sensor_status.append(status)

        if fire_count > 0:
            self.alarm_message = "ALEV ALGILANDI"
        elif smoke_count > 0:
            self.alarm_message = "DUMAN ALGILANDI"
        else:
            self.alarm_message = "Sensörler normal."
    #  Bu alanlardan alınan görüntüler CNN modeline gönderilerek yangın, duman veya normal tahmini yapılmaktadır. A
    def _update_camera_detection(self):
        results = []

        for idx, cam in enumerate(self.camera_zones):
            name = cam["name"]
            rect = cam["rect"]

            # 1) CNN tahmini al
            try:
                safe_rect = rect.clip(pygame.Rect(0, 0, self.map_w, self.win_h))

                if safe_rect.width <= 0 or safe_rect.height <= 0:
                    cnn_result = "normal"
                else:
                    camera_view = self.screen.subsurface(safe_rect).copy()

                    image_path = os.path.join(
                        self.camera_temp_dir,
                        f"camera_{idx}.jpg"
                    )

                    pygame.image.save(camera_view, image_path)
                    cnn_result = predict_image(image_path)

            except Exception:
                cnn_result = "normal"

            # 2) Kamera alanındaki gerçek simülasyon durumunu kontrol et
            fire_count = 0
            smoke_count = 0

            for r in range(self.bg.rows):
                for c in range(self.bg.cols):
                    x, y = self._xy(r, c)
                    cell_rect = pygame.Rect(x, y, CELL, CELL)

                    if rect.colliderect(cell_rect):
                        cell = self.bg.get(r, c)

                        if cell == FIRE:
                            fire_count += 1
                        elif cell == SMOKE:
                            smoke_count += 1

            # 3) CNN + grid doğrulamalı karar
            if cnn_result == "fire" and fire_count > 0:
                label = "ALEV"
            elif cnn_result == "smoke" and smoke_count > 0:
                label = "DUMAN"
            elif fire_count > 0:
                label = "ALEV"
            elif smoke_count > 0:
                label = "DUMAN"
            else:
                label = "NORMAL"

            results.append((name, label))

        self.camera_zone_results = results

        labels = [label for _, label in results]

        if "ALEV" in labels:
            self.camera_result = "ALEV ALGILANDI"
        elif "DUMAN" in labels:
            self.camera_result = "DUMAN ALGILANDI"
        else:
            self.camera_result = "NORMAL"
    # Önce tüm kişiler bulunur. Her kişi için Dijkstra ile güvenli rota hesaplanır. 
    def _move_person_one_step(self):
        persons = self.bg.find(PERSON)
 
        if not persons:
           return

        random.shuffle(persons)

        occupied = set(persons)

        for current_pos in persons:
            if current_pos not in occupied:
                continue

            path = self.pathfinder.find_path(current_pos)

            if not path or len(path) < 2:
               continue

            next_pos = path[1]
            nr, nc = next_pos
            next_cell = self.bg.get(nr, nc)

            if next_cell == EXIT:
                pr, pc = current_pos
                self.bg.set(pr, pc, EMPTY)
                occupied.remove(current_pos)
                self.message = "Bir kişi güvenli çıkışa ulaştı."
                continue

            if next_cell in (FIRE, WALL, PERSON):
                continue

            pr, pc = current_pos
            self.bg.set(pr, pc, EMPTY)
            self.bg.set(nr, nc, PERSON)

            occupied.remove(current_pos)
            occupied.add(next_pos)

    # ── Partiküller ───────────────────────────────────────────────
    def _spawn_particles(self):
        for r, c in self.bg.find(FIRE):
            x, y = self._xy(r, c)
            cx, cy = x + CELL//2, y + CELL//2
            for _ in range(4):
                self.fire_particles.append({
                    "x": cx + random.randint(-16, 16),
                    "y": cy + random.randint(-10, 10),
                    "vx": random.uniform(-0.4, 0.4),
                    "vy": random.uniform(-1.2, -0.4),
                    "r":  random.randint(4, 11),
                    "life": random.randint(20, 45),
                    "color": random.choice([
                        (255,235,60),(255,155,30),(245,70,20),(200,35,15)
                    ])
                })
        self.fire_particles = self.fire_particles[-600:]

    def _update_particles(self):
        # Ateş partikülleri
        alive = []
        for p in self.fire_particles:
            p["x"] += p["vx"] + math.sin(self.tick*0.06 + p["y"]*0.04)*0.4
            p["y"] += p["vy"]
            p["r"]  = max(1, p["r"] - 0.07)
            p["life"] -= 1
            if p["life"] > 0:
                alive.append(p)
        self.fire_particles = alive

        # Duman bulutları - daha sinematik, yayılmış ve belirgin
        if self.tick % 4 == 0:
            for r, c in self.bg.find(SMOKE):
                x, y = self._xy(r, c)
                for _ in range(2):
                    self.smoke_puffs.append({
                        "x": x + CELL//2 + random.randint(-18, 18),
                        "y": y + CELL//2 + random.randint(-18, 18),
                        "r": random.randint(24, 44),
                        "life": random.randint(50, 90),
                        "alpha": random.randint(105, 175),
                    })
            self.smoke_puffs = self.smoke_puffs[-700:]

        alive2 = []
        for p in self.smoke_puffs:
            p["y"]    -= 0.25
            p["r"]    += 0.12
            p["alpha"] = max(0, p["alpha"] - 2)
            p["life"] -= 1
            if p["life"] > 0:
                alive2.append(p)
        self.smoke_puffs = alive2

        # Ateş partikülleri kapalı tutuldu; yangın sadece alev formuyla çiziliyor.

    def _xy(self, r, c):
        return c * CELL, TOP_H + r * CELL

    # ── Çizim ─────────────────────────────────────────────────────
    def _draw(self):
        self.screen.fill(C["BG"])
        self._draw_top()

        # Zemin
        self.screen.blit(self.floor_surf, (0, TOP_H))

        # Sensörler
        self._draw_sensors()

        # Rota
        self._draw_path()

        # Duman
        self._draw_smoke()

        # Yangın ışıması
        #self._draw_fire_glow()

        # Yangın görseli
        self._draw_fire_sprites()

        # Ateş partikülleri
        #self._draw_fire_particles()

        # Nesneler (kişi, çıkış)
        self._draw_objects()

        # Temiz sahne üzerinden kamera analizi
        self._update_camera_detection()

        # Kamera görüş alanları
        self._draw_camera_zones()

        # Vignette
        self._draw_vignette()

        # Panel
        self._draw_panel()

        # Alt bar
        self._draw_bottom()

        pygame.display.flip()

    # ── Üst bar ───────────────────────────────────────────────────
    def _draw_top(self):
        pygame.draw.rect(self.screen, C["TOP"], (0, 0, self.win_w, TOP_H))
        pygame.draw.line(self.screen, C["TOP_BORDER"],
                         (0, TOP_H-1), (self.win_w, TOP_H-1), 2)

        # Alev ikonu (kod ile)
        self._mini_flame(18, 14)

        t = self.f_title.render(
            "AKILLI YANGIN TAHMİN VE TAHLİYE SİSTEMİ", True, C["TEXT"])
        self.screen.blit(t, (46, 16))

        ctrl = self.f_small.render(
            "Kontroller:   SPACE (Oynat/Duraklat)     A (Adım)     R (Sıfırla)",
            True, C["MUTED"])
        self.screen.blit(ctrl, (self.win_w - PANEL_W - 430, 18))

    def _mini_flame(self, x, y):
        for i, col in enumerate([(200,35,15),(245,80,20),(255,155,30),(255,220,55)]):
            offset = int(math.sin(self.tick*0.15 + i)*2)
            pts = [(x+12, y+24),(x+4+i*2, y+14-i*2+offset),(x+20-i*2, y+14-i*2+offset)]
            pygame.draw.polygon(self.screen, col, pts)

    def _draw_camera_zones(self):
        if not self.camera_zones:
            return

        for idx, cam in enumerate(self.camera_zones):
            name = cam["name"]
            pos = cam["pos"]
            points = cam["points"]

            label = "NORMAL"

            if idx < len(self.camera_zone_results):
                label = self.camera_zone_results[idx][1]

            if label == "ALEV":
                color = C["RED"]
            elif label == "DUMAN":
                color = C["YELLOW"]
            else:
                color = C["BLUE"]

            cone = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)

            pygame.draw.polygon(
                cone,
                (*color, 28),
                points
            )

            pygame.draw.polygon(
                cone,
                (*color, 90),
                points,
                2
            )

            self.screen.blit(cone, (0, 0))

            cx, cy = pos

            body_rect = pygame.Rect(cx - 8, cy - 6, 20, 12)

            pygame.draw.rect(
                self.screen,
                (18, 24, 30),
                body_rect,
                border_radius=3
            )

            pygame.draw.rect(
                self.screen,
                color,
                body_rect,
                2,
                border_radius=3
            )

            lens_pos = (cx + 15, cy)

            pygame.draw.circle(
                self.screen,
                color,
                lens_pos,
                6
            )

            pygame.draw.circle(
                self.screen,
                (10, 15, 20),
                lens_pos,
                3
            )

            pygame.draw.line(
                self.screen,
                color,
                (cx - 2, cy + 8),
                (cx - 8, cy + 18),
                2
            )

            pygame.draw.line(
                self.screen,
                color,
                (cx - 8, cy + 18),
                (cx + 8, cy + 18),
                2
            )

            txt = self.f_small.render(name, True, color)
            self.screen.blit(txt, (cx - 10, cy - 22))

            status_bg = pygame.Rect(cx - 18, cy + 20, 95, 22)
            pygame.draw.rect(self.screen, (10, 14, 18), status_bg, border_radius=5)
            pygame.draw.rect(self.screen, color, status_bg, 1, border_radius=5)

            status_txt = self.f_small.render(label, True, color)
            self.screen.blit(status_txt, (cx - 6, cy + 25))

    # ── Sensörler ─────────────────────────────────────────────────
    def _draw_sensors(self):
        if not self.sensor_status:
            self._update_sensors()

        for idx, (r, c) in enumerate(self.sensor_positions):
            x, y = self._xy(r, c)
            cx, cy = x + CELL // 2, y + CELL // 2

            status = self.sensor_status[idx] if idx < len(self.sensor_status) else "NORMAL"

            if status == "FIRE":
                color = C["RED"]
            elif status == "SMOKE":
                color = C["YELLOW"]
            else:
                color = C["GREEN"]

            pygame.draw.circle(self.screen, (15, 25, 30), (cx, cy), 9)
            pygame.draw.circle(self.screen, color, (cx, cy), 4)

            pulse = int(abs(math.sin(self.tick * 0.08)) * 8) + 7
            alpha = 90 if status in ("FIRE", "SMOKE") else 45

            s = pygame.Surface((pulse * 2 + 2, pulse * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (pulse + 1, pulse + 1), pulse)
            self.screen.blit(s, (cx - pulse - 1, cy - pulse - 1))

    # ── Rota ──────────────────────────────────────────────────────
    def _draw_path(self):
        if len(self.path_list) < 2:
            return

        persons = self.bg.find(PERSON)

        if not persons:
            return

        current_pos = persons[0]

        try:
            idx = self.path_list.index(current_pos)
        except ValueError:
            idx = 0

        visible_path = self.path_list[idx:idx + 5]

        pts = []

        for r, c in visible_path:
            if self.bg.get(r, c) not in (FIRE, WALL):
                x, y = self._xy(r, c)
                pts.append((x + CELL // 2, y + CELL // 2))

        if len(pts) < 2:
            return

        glow = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)

        pygame.draw.lines(
            glow,
            (80, 220, 255, 28),
            False,
            pts,
            6
        )

        self.screen.blit(glow, (0, 0))

        pygame.draw.lines(
            self.screen,
            (55, 185, 225),
            False,
            pts,
            2
        )

        for i, p in enumerate(pts):
            alpha = max(70, 180 - i * 30)

            dot = pygame.Surface((12, 12), pygame.SRCALPHA)

            pygame.draw.circle(
                dot,
                (120, 240, 255, alpha),
                (6, 6),
                3
            )

            self.screen.blit(dot, (p[0] - 6, p[1] - 6))

    # ── Duman ─────────────────────────────────────────────────────
    def _draw_smoke(self):
        surf = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        for p in self.smoke_puffs:
            pygame.draw.circle(surf,
                               (18, 18, 18, p["alpha"]),
                               (int(p["x"]), int(p["y"])),
                               int(p["r"]))
        self.screen.blit(surf, (0, 0))

    # ── Yangın ışıması ────────────────────────────────────────────
    def _draw_fire_glow(self):
        glow = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        pulse = 1.0 + math.sin(self.tick * 0.12) * 0.15
        for r, c in self.bg.find(FIRE):
            x, y = self._xy(r, c)
            cx, cy = x+CELL//2, y+CELL//2
            r1 = int(CELL * 2.2 * pulse)
            r2 = int(CELL * 3.5 * pulse)
            pygame.draw.circle(glow, (255,  80, 20, 110), (cx,cy), r1)
            pygame.draw.circle(glow, (255, 140, 25,  55), (cx,cy), r2)
        self.screen.blit(glow, (0,0))

    # ── Yangın görseli ────────────────────────────────────────────
    def _draw_fire_sprites(self):
        fire_cells = self.bg.find(FIRE)

        if not fire_cells:
            return

        fire_surface = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        glow_surface = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)

        for r, c in fire_cells:
            x, y = self._xy(r, c)
            cx = x + CELL // 2
            cy = y + CELL // 2

            pulse = 1.0 + math.sin(self.tick * 0.12 + r * 0.7 + c * 0.5) * 0.12

            pygame.draw.circle(
                glow_surface,
                (255, 70, 10, 34),
                (cx, cy),
                int(CELL * 2.4 * pulse)
            )

            pygame.draw.circle(
                glow_surface,
                (255, 135, 20, 28),
                (cx, cy),
                int(CELL * 1.6 * pulse)
            )

            pygame.draw.ellipse(
                fire_surface,
                (135, 20, 8, 95),
                (cx - CELL, cy + CELL // 5, CELL * 2, CELL)
            )

        self.screen.blit(glow_surface, (0, 0))

        for r, c in fire_cells:
            x, y = self._xy(r, c)
            cx = x + CELL // 2
            cy = y + CELL // 2

            seed = r * 91 + c * 47
            phase = seed * 0.11
            t = self.tick * 0.16

            main_count = 3 + seed % 2

            for i in range(main_count):
                local_phase = phase + i * 1.4

                height = int((CELL * 1.4) + ((seed + i * 13) % 14))
                width = int((CELL * 0.45) + ((seed + i * 7) % 8))

                ox = math.sin(t + local_phase) * (CELL * 0.18)
                top_x = cx + int(ox)
                top_y = cy - height + int(math.cos(t + local_phase) * 4)

                base_y = cy + int(CELL * 0.55)
                left_x = cx - width + int(math.sin(t * 0.7 + local_phase) * 3)
                right_x = cx + width + int(math.cos(t * 0.8 + local_phase) * 3)

                colors = [
                    (120, 12, 5, 225),
                    (210, 45, 10, 235),
                    (255, 115, 20, 242),
                    (255, 220, 75, 250),
                ]

                for layer, color in enumerate(colors):
                    shrink = layer * 5
                    pts = [
                        (top_x, top_y + shrink),
                        (left_x + shrink, base_y - shrink // 2),
                        (cx, base_y + 4),
                        (right_x - shrink, base_y - shrink // 2),
                    ]

                    pygame.draw.polygon(fire_surface, color, pts)

            side_flames = 2 + seed % 3

            for i in range(side_flames):
                local_phase = phase + i * 2.0

                height = int(CELL * 0.8 + ((seed + i * 5) % 10))
                width = int(CELL * 0.25 + ((seed + i * 3) % 5))

                px = cx + int(math.sin(t + local_phase) * CELL * 0.55)
                py = cy + int(math.cos(t * 0.7 + local_phase) * CELL * 0.2)

                pts = [
                    (px, py - height),
                    (px - width, py + 5),
                    (px + width, py + 5),
                ]

                color = [
                    (255, 75, 15, 160),
                    (255, 135, 25, 180),
                    (255, 200, 55, 190),
                ][(seed + i) % 3]

                pygame.draw.polygon(fire_surface, color, pts)

        fire_surface.set_alpha(238)
        self.screen.blit(fire_surface, (0, 0))

    # ── Ateş partikülleri ─────────────────────────────────────────
    def _draw_fire_particles(self):
        surf = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        for p in self.fire_particles:
            a = max(0, min(255, int(p["life"] * 6)))
            pygame.draw.circle(surf, (*p["color"], a),
                               (int(p["x"]), int(p["y"])),
                               max(1, int(p["r"])))
        self.screen.blit(surf, (0, 0))

    # ── Nesneler ──────────────────────────────────────────────────
    def _draw_objects(self):
        # Çıkışlar
        for r, c in self.bg.find(EXIT):
            x, y = self._xy(r, c)
            self._draw_exit(x, y)

        # Kişiler
        for r, c in self.bg.find(PERSON):
            x, y = self._xy(r, c)
            self._draw_person(x, y)

    def _draw_person(self, x, y):
        cx, cy = x+CELL//2, y+CELL//2
        if self.images.get("person"):
            sz  = int(CELL * 1.5)
            img = pygame.transform.smoothscale(self.images["person"], (sz, sz))
            self.screen.blit(img, (cx-sz//2, cy-sz//2))
        else:
            # Detaylı insan silüeti
            col = C["BLUE"]
            # Kafa
            pygame.draw.circle(self.screen, col, (cx, cy-13), 7)
            # Gövde
            pygame.draw.line(self.screen, col, (cx, cy-6), (cx, cy+8), 4)
            # Kollar
            pygame.draw.line(self.screen, col, (cx-9, cy-1), (cx+9, cy-1), 3)
            # Bacaklar
            pygame.draw.line(self.screen, col, (cx, cy+8), (cx-6, cy+18), 3)
            pygame.draw.line(self.screen, col, (cx, cy+8), (cx+6, cy+18), 3)

    def _draw_exit(self, x, y):
        cx, cy = x+CELL//2, y+CELL//2
        if self.images.get("exit"):
            sz  = int(CELL * 1.4)
            img = pygame.transform.smoothscale(self.images["exit"], (sz, sz))
            self.screen.blit(img, (cx-sz//2, cy-sz//2))
        else:
            # Yeşil çıkış kutusu
            rect = pygame.Rect(x+3, y+3, CELL-6, CELL-6)
            pygame.draw.rect(self.screen, (18,150,65), rect, border_radius=5)
            pygame.draw.rect(self.screen, (50,220,90), rect, 2, border_radius=5)
            # ÇIKIŞ yazısı
            t = self.f_small.render("ÇIKIŞ", True, (240,255,240))
            self.screen.blit(t, (cx-t.get_width()//2, cy-6))
            # Yukarı ok
            pygame.draw.polygon(self.screen, (240,255,240),
                                [(cx,y+5),(cx-5,y+13),(cx+5,y+13)])

    # ── Vignette ──────────────────────────────────────────────────
    def _draw_vignette(self):
        ov = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        for i in range(80):
            a = int(i * 1.1)
            pygame.draw.rect(ov, (0,0,0,a), (i,i,self.map_w-2*i,self.map_h-2*i), 1)
        self.screen.blit(ov, (0, TOP_H))

    # ── Sağ panel ─────────────────────────────────────────────────
    def _card(self, x, y, w, h, title, title_color=None):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, C["CARD"],    rect, border_radius=10)
        pygame.draw.rect(self.screen, C["CARD_BOR"],rect, 1, border_radius=10)
        tc = title_color or C["GREEN"]
        self.screen.blit(self.f_head.render(title, True, tc), (x+14, y+11))
        return y + 38

    def _txt(self, text, x, y, color=None, bold=False):
        f = pygame.font.SysFont("Arial", 13, bold=bold)
        self.screen.blit(f.render(text, True, color or C["TEXT"]), (x, y))

    def _draw_panel(self):
        px = self.map_w
        pygame.draw.rect(self.screen, C["PANEL"],
                         (px, 0, PANEL_W, self.win_h))
        pygame.draw.line(self.screen, C["CARD_BOR"],
                         (px, 0), (px, self.win_h), 2)

        x0 = px + 14
        w  = PANEL_W - 28
        y  = TOP_H + 12

        # ── Simülasyon bilgileri ──────────────────────────────────
        y2 = self._card(x0, y, w, 155, "SİMÜLASYON BİLGİLERİ")
        smoke_cnt = len(self.bg.find(SMOKE))
        smoke_lbl = "Az" if smoke_cnt < 5 else ("Orta" if smoke_cnt < 15 else "Yoğun")
        rows = [
            ("Adım",           str(self.step_count)),
            ("Yangın Hücresi", str(len(self.bg.find(FIRE)))),
            ("Duman Seviyesi", smoke_lbl),
            ("Alarm", "AKTİF" if self.alarm_active else "Normal"),
            #("Kamera", self.camera_result),
            ("Güvenli Çıkış",  "Bulundu" if self.path_list else "YOK!"),
            ("Yeniden Planlama", str(self.replan_count)),
        ]
        for lbl, val in rows:
            self._txt(lbl + ":", x0+14, y2)
            vc = C["RED"] if (lbl == "Güvenli Çıkış" and not self.path_list) else C["TEXT"]
            self._txt(val, x0+w-80, y2, color=vc, bold=True)
            y2 += 20

        y2 += 8
        self._txt("Kamera Sonuçları:", x0 + 14, y2, color=C["BLUE"], bold=True)
        y2 += 20

        for name, label in self.camera_zone_results:
            if label == "ALEV":
                cc = C["RED"]
            elif label == "DUMAN":
                cc = C["YELLOW"]
            elif label == "NORMAL":
                cc = C["GREEN"]
            else:
                cc = C["MUTED"]

            self._txt(f"{name}: {label}", x0 + 14, y2, color=cc)
            y2 += 18

        y += 185

        # ── Lejant ────────────────────────────────────────────────
        y2 = self._card(x0, y, w, 200, "LEJAND")
        legend = [
            (C["WALL_C"],     "Duvar"),
            (C["FLOOR"],      "Yürüme Alanı"),
            (C["RED"],        "Yangın"),
            ((30, 30, 30),    "Duman"),
            (C["BLUE"],       "İnsan"),
            ((18,150,65),     "Çıkış"),
            (C["BLUE_GLOW"],  "Güvenli Rota"),
        ]
        for col, lbl in legend:
            pygame.draw.rect(self.screen, col, (x0+14, y2+2, 18, 14), border_radius=3)
            pygame.draw.rect(self.screen, C["CARD_BOR"], (x0+14, y2+2, 18, 14), 1, border_radius=3)
            self._txt(lbl, x0+40, y2)
            y2 += 22
        y += 210

        # ── Durum ─────────────────────────────────────────────────
        y2 = self._card(x0, y, w, 88, "DURUM")
        ok = bool(self.path_list)
        pygame.draw.circle(self.screen, C["GREEN"] if ok else C["RED"],
                           (x0+22, y2+8), 7)
        self._txt("Tahliye devam ediyor..." if ok else "Rota kapandı!",
                  x0+38, y2, color=C["TEXT"])
        y2 += 28
        msg = self.message[:36]
        self._txt(msg, x0+14, y2, color=C["BLUE"])
        y += 98

        # ── Kontroller ────────────────────────────────────────────
        y2 = self._card(x0, y, w, 100, "KONTROLLER")
        for line in ["SPACE : Oynat / Duraklat",
                     "A     : Adım ilerlet",
                     "R     : Sıfırla",
                     "ESC   : Çıkış"]:
            self._txt(line, x0+14, y2, color=C["MUTED"])
            y2 += 22

    # ── Alt bar ───────────────────────────────────────────────────
    def _draw_bottom(self):
        by = TOP_H + self.map_h
        pygame.draw.rect(self.screen, (8,12,18), (0, by, self.win_w, BOT_H))
        pygame.draw.line(self.screen, C["CARD_BOR"], (0,by), (self.win_w,by), 2)

        # Risk seviyesi
        self._txt("RİSK SEVİYESİ", 18, by+20,
                  color=(150,210,235), bold=True)

        bar_x, bar_y, bar_w, bar_h = 145, by+21, 500, 16
        pygame.draw.rect(self.screen, (35,45,52),
                         (bar_x, bar_y, bar_w, bar_h), border_radius=8)

        # Gradient bar
        for i in range(bar_w):
            t = i / bar_w
            if t < 0.5:
                r2 = min(255, int(40 + t*2*215))
                g2 = 200
                b2 = 50
            else:
                r2 = 235
                g2 = max(0, int(200 - (t-0.5)*2*185))
                b2 = 30
            pygame.draw.line(self.screen, (r2, g2, b2),
                             (bar_x+i, bar_y+2),
                             (bar_x+i, bar_y+bar_h-2))

        # Risk ibresi
        danger = self.fire_simulator.get_fire_danger()
        ind_x = bar_x + int(danger * bar_w)
        pygame.draw.polygon(self.screen, (255,255,255),
                            [(ind_x, bar_y-5),
                             (ind_x-5, bar_y-12),
                             (ind_x+5, bar_y-12)])

        self._txt("Düşük",  bar_x-52,  bar_y, color=C["GREEN"])
        self._txt("Yüksek", bar_x+bar_w+6, bar_y, color=C["RED"])

        # Otomatik oynatma butonu
        bx = self.win_w - 180
        pygame.draw.rect(self.screen, C["CARD"],
                         (bx, by+9, 155, 38), border_radius=8)
        pygame.draw.rect(self.screen, C["CARD_BOR"],
                         (bx, by+9, 155, 38), 1, border_radius=8)

        self._txt("OTOMATİK OYNATMA", bx+12, by+14,
                  color=C["MUTED"])

        # Play/pause ikonu
        if self.auto_play:
            pygame.draw.rect(self.screen, C["GREEN"], (bx+105, by+23, 5, 14))
            pygame.draw.rect(self.screen, C["GREEN"], (bx+115, by+23, 5, 14))
        else:
            pygame.draw.polygon(self.screen, C["GREEN"],
                                [(bx+105, by+21),
                                 (bx+105, by+37),
                                 (bx+123, by+29)])

        # Aktif göstergesi
        pygame.draw.circle(self.screen,
                           C["GREEN"] if self.auto_play else C["RED"],
                           (bx+143, by+29), 8)


if __name__ == "__main__":
    bina = BuildingGrid()
    PygameView(bina).run()