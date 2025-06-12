# System rozpoznawania i śledzenia twarzy

Ten projekt implementuje system rozpoznawania twarzy z opcjonalnymi możliwościami śledzenia. System wykorzystuje uczenie maszynowe do identyfikacji osób w strumieniach wideo w czasie rzeczywistym.

## Przegląd

Podstawowy system wykorzystuje kamerę do wykrywania i rozpoznawania twarzy w czasie rzeczywistym. Może być rozszerzony o zmotoryzowany uchwyt kamery do automatycznego śledzenia, utrzymujący docelowe twarze wycentrowane w kadrze.

## Funkcje

- **Wysoka dokładność wykrywania twarzy**: Wykorzystanie InsightFace do niezawodnego wykrywania twarzy
- **Wiele metod rozpoznawania**: KNN, Naive Bayes, Decision Tree, SVM, MLP
- **Interaktywne szkolenie**: Prosty interfejs szkoleniowy oparty na kamerze
- **Augmentacja danych**: Wzbogaca dane treningowe dla lepszego rozpoznawania
- **Metryki wydajności**: Wyświetlanie w czasie rzeczywistym FPS i wskaźnika rozpoznawania
- **Opcjonalne śledzenie**: Może być podłączony do zmotoryzowanego uchwytu (wymaga dodatkowego sprzętu)

## Instalacja

1. Zainstaluj wymagane pakiety Python:
```bash
pip install opencv-python numpy insightface scikit-learn requests tqdm onnxruntime
```

2. Utwórz katalog `data/known_faces` do przechowywania obrazów treningowych.

3. Dla funkcji śledzenia (opcjonalnie):
```bash
cd esp32-servo-controller
pio run -t upload
```

## Proces szkolenia

### Przegląd

Proces szkolenia obejmuje przechwytywanie obrazów twarzy, ekstrakcję cech twarzy i trenowanie wielu modeli uczenia maszynowego do porównania.

### Proces krok po kroku

1. **Zbieranie danych**:
   - Obrazy są przechwytywane za pomocą skryptu train.py z flagą `--interactive`
   - Każda osoba otrzymuje własny katalog w `data/known_faces/{person_name}`
   - Przechwytuj wiele obrazów z różnych kątów dla lepszego rozpoznawania

```bash
python train.py --interactive
```

2. **Ekstrakcja cech twarzy**:
   - InsightFace wykrywa twarze na obrazach treningowych
   - Dla każdej twarzy ekstrahowane są 512-wymiarowe embeddingi
   - Te embeddingi reprezentują unikalne cechy twarzy

3. **Augmentacja danych**:
   - Oryginalne obrazy są przekształcane w celu utworzenia dodatkowych danych treningowych:
   - Rotacje (±5 stopni)
   - Zmiany jasności (80% i 120%)
   - Rozmycie gaussowskie
   - To zwiększa rozmiar zbioru danych 5-krotnie

4. **Trenowanie modelu**:
   - Cechy są standaryzowane za pomocą `StandardScaler`
   - Trenowane są różne modele klasyfikacyjne:
     - K najbliższych sąsiadów (KNN)
     - Naive Bayes
     - Drzewo decyzyjne
     - Perceptron wielowarstwowy (MLP)
     - Maszyna wektorów nośnych (SVM)

5. **Tworzenie bazy danych**:
   - Embeddingi twarzy i nazwy osób są zapisywane do bazy danych
   - Każdy model jest zapisywany oddzielnie wraz ze skalerem cech

### Poprawa rozpoznawania nieznanych twarzy

Wraz z dodawaniem większej liczby nieznanych twarzy do zbioru treningowego:
- Progi pewności rozpoznawania stają się bardziej znaczące
- Wskaźniki fałszywych trafień maleją
- System staje się bardziej selektywny w identyfikacji

### Zapisane pliki

Po treningu tworzone są następujące pliki w katalogu `data/models/`:

- `face_recognition_database.pkl`: Podstawowa baza danych z embeddingami i nazwami
- `face_recognition_database_scaler.pkl`: StandardScaler do normalizacji cech
- Pliki modeli dla każdego klasyfikatora (KNN, Naive Bayes, Decision Tree, MLP, SVM)

## Proces rozpoznawania i śledzenia

### Główna pętla rozpoznawania

1. **Inicjalizacja**:
   - Nawiązywane jest połączenie z kamerą
   - Inicjalizowany jest detektor twarzy (InsightFace)
   - Model rozpoznawania jest ładowany na podstawie wybranej metody

2. **Przetwarzanie klatek**:
   - Klatki są przechwytywane z kamery
   - Przetwarzanie odbywa się w oddzielnym wątku dla lepszej wydajności
   - Obliczane są metryki wydajności (FPS, wskaźnik rozpoznawania)

3. **Wykrywanie twarzy**:
   - InsightFace lokalizuje twarze w każdej klatce
   - Zwraca prostokąty ograniczające i embeddingi twarzy
   - Dostarcza również punkty charakterystyczne twarzy (oczy, nos, usta)

4. **Rozpoznawanie twarzy**:
   - Embeddingi twarzy są przekazywane do wybranego modelu ML
   - Model przewiduje tożsamość z wynikiem pewności
   - Jeśli pewność jest poniżej progu, twarz jest oznaczana jako "Nieznana"

5. **Wyświetlanie**:
   - Rozpoznane twarze są oznaczane prostokątami ograniczającymi
   - Wyświetlane są nazwy i wyniki pewności
   - Punkty charakterystyczne twarzy są podświetlone

## Szczegóły techniczne

### Integracja InsightFace

InsightFace jest używany zarówno do wykrywania twarzy, jak i ekstrakcji embeddingów:

```python
# Inicjalizacja
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Wykrywanie i ekstrakcja embeddingów
faces = face_app.get(image)
for face in faces:
    embedding = face.embedding  # 512-wymiarowy wektor
    bbox = face.bbox  # Współrzędne prostokąta ograniczającego
    landmarks = face.kps  # Punkty charakterystyczne twarzy
```

## Format przechowywania danych

System rozpoznawania twarzy przechowuje dane twarzy w pliku pickle o następującej strukturze:

```python
{
    "embeddings": [array1, array2, ...],  # Lista tablic numpy (512-wymiarowe embeddingi twarzy)
    "names": ["person1", "person2", ...]  # Odpowiadające nazwy osób
}
```

Każdy wytrenowany model uczenia maszynowego jest przechowywany w oddzielnym pliku z konwencją nazewnictwa:
```
face_recognition_database_<method>.pkl
```

Dodatkowo, zapisywany jest model skalera do standaryzacji wartości cech:
```
face_recognition_database_scaler.pkl
```

## Użycie

### Trenowanie rozpoznawania twarzy

```bash
python train.py --interactive
```

Postępuj zgodnie z instrukcjami, aby przechwycić obrazy twarzy. Poruszaj głową w różnych pozycjach dla lepszych wyników treningu.

### Trenowanie na już istniejących plikach

```bash
python train.py
```

Możesz również użyć parametru `--augment` do augmentacji zdjęć (zarówno interaktywnie, jak i istniejących)

### Uruchamianie rozpoznawania twarzy

```bash
python recognize.py --recognition-method knn --target "Nazwa Osoby"
```

Opcje wiersza poleceń:
- `--recognition-method`: Wybierz spośród knn, naive_bayes, decision_tree, mlp, svm
- `--target`: Nazwa osoby do śledzenia (opcjonalnie)
- `--camera-id`: ID urządzenia kamery USB (domyślnie: 0)

### Funkcjonalność śledzenia (opcjonalnie)

Jeśli skonfigurowałeś sprzęt do śledzenia:

```bash
python recognize.py --recognition-method knn --target "Nazwa Osoby" --servo-url http://192.168.4.1
```