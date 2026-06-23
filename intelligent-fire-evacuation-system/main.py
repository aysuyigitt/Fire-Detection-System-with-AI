import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from grid import BuildingGrid, FIRE, SMOKE, EXIT, PERSON
from pygame_view import PygameView


def main():
    bina = BuildingGrid(rows=20, cols=28)

    print("=" * 45)
    print("  Akıllı Yangın Tahmin ve Tahliye Sistemi")
    print("=" * 45)
    print(f"  Bina boyutu     : {bina.rows} x {bina.cols}")
    print(f"  Yangın hücresi  : {bina.find(FIRE)}")
    print(f"  Duman hücresi   : {bina.find(SMOKE)}")
    print(f"  Çıkış kapıları  : {bina.find(EXIT)}")
    print(f"  Kişi konumu     : {bina.find(PERSON)}")
    print("=" * 45)
    print("  SPACE → Oynat / Duraklat")
    print("  A     → Tek adım")
    print("  R     → Sıfırla")
    print("  ESC   → Çık")
    print("=" * 45)

    view = PygameView(bina)
    view.run()


if __name__ == "__main__":
    main()