# Analiza Wyników Rozpoznawania Twarzy

**Konfiguracja klasyfikatorów (chyba, że określona w zestawie inaczej):**
```python
        classifiers = {
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(max_depth=5),
            'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
            'svm': SVC(kernel='linear', probability=True)
        }
```

## Zestaw 1
**Dane testowe:**
- 472 losowych twarzy
- 129 twarzy Macieja
- 30 twarzy Matta Damona
- 76 twarzy Jagody
- Augmentacja

**Wyniki klasyfikatorów:**
- **KNN:** Bardzo wysoka skuteczność rozpoznawania znanych twarzy. Sporadycznie klasyfikuje nieznane twarze jako znane.
- **Naive Bayes:** Poprawnie identyfikuje Macieja i Jagodę, nawet przy ekstremalnych pozach i częściowym zakryciu twarzy. Pozostałe twarze klasyfikuje jako nieznane.
- **Decision Tree:** Poprawnie rozpoznaje twarze Macieja i Jagody, także przy częściowym zasłonięciu. Inne twarze oznacza jako nieznane.
- **MLP:** Całkowity brak rozpoznawania jakichkolwiek twarzy.
- **SVM:** Precyzyjne rozpoznawanie Macieja i Jagody. Inne twarze tylko w szczególnych pozach.

## Zestaw 2
**Dane testowe:**
- 75 twarzy Jagody
- 127 twarzy Macieja
- 40 twarzy Matta Damona
- 236 nieznanych twarzy
- Augmentacja

**Konfiguracja klasyfikatorów:**
```python
classifiers = {
    'knn': KNeighborsClassifier(n_neighbors=min(5, min(class_counts.values())), weights='distance'),
    'naive_bayes': GaussianNB(var_smoothing=1e-8),
    'decision_tree': DecisionTreeClassifier(max_depth=8, class_weight=class_weight_dict, min_samples_split=5),
    'mlp': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True, 
                    learning_rate_init=0.001, alpha=0.0001),
    'svm': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, class_weight=class_weight_dict)
}
```

**Wyniki klasyfikatorów:**
- **KNN:** Wysoka skuteczność dla znanych twarzy, tendencja do okazjonalnego klasyfikowania nieznanych jako znane.
- **Naive Bayes:** Skuteczne rozpoznawanie tylko Macieja i Jagody (niezależnie od pozycji i częściowych zasłonięć), pozostałe jako Unknown.
- **Decision Tree:** Podobnie jak Naive Bayes - poprawnie rozpoznaje Macieja i Jagodę nawet w trudnych warunkach.
- **MLP:** Brak skuteczności w rozpoznawaniu twarzy.
- **SVM:** Wysoka odporność na zakłócenia, precyzyjne rozpoznawanie Macieja i Jagody.

## Zestaw 3
**Dane testowe:**
- 75 twarzy Jagody
- 127 twarzy Macieja
- 40 twarzy Matta Damona
- 236 nieznanych twarzy
- Augmentacja
- Brak skalowania danych

**Wyniki klasyfikatorów:**
- **KNN:** Wyjątkowo dobra skuteczność z wysoką odpornością na zakłócenia. Czasem błędnie klasyfikuje obce twarze.
- **Naive Bayes:** Dobre wyniki dla Macieja i Jagody, inne twarze nierozpoznane.
- **Decision Tree:** Skuteczne rozpoznawanie głównie Macieja, inne twarze klasyfikowane jako nieznane.
- **MLP:** Całkowity brak skuteczności.
- **SVM:** Bardzo dobre wyniki - precyzyjne rozpoznawanie wszystkich twarzy z wysoką odpornością na zakłócenia.

## Zestaw 4
**Dane testowe:**
- 75 twarzy Jagody
- 127 twarzy Macieja
- 40 twarzy Matta Damona
- 236 nieznanych twarzy
- Augmentacja

**Wyniki klasyfikatorów:**
- **KNN:** Wysoka skuteczność rozpoznawania znanych twarzy, dobra odporność na zakłócenia.
- **Naive Bayes:** Poprawne rozpoznawanie Macieja i Jagody.
- **Decision Tree:** Szczególnie skuteczny dla twarzy Macieja, wysoka odporność na zakłócenia.
- **MLP:** Doskonałe wyniki dla wszystkich twarzy z wysoką odpornością na zakłócenia.
- **SVM:** Doskonałe wyniki dla wszystkich twarzy z wysoką odpornością na zakłócenia.

## Zestaw 5
**Dane testowe:**
- 472 losowych twarzy
- 129 twarzy Macieja
- 30 twarzy Matta Damona
- 76 twarzy Jagody
- Augmentacja

**Ulepszone parametry klasyfikatorów:**
```python
'knn': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine', algorithm='auto'),
'naive_bayes': GaussianNB(var_smoothing=1e-8),
'decision_tree': DecisionTreeClassifier(max_depth=20, min_samples_split=5, criterion='entropy'),
'mlp': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, learning_rate='adaptive', 
activation='relu', solver='adam', alpha=0.0001, early_stopping=True),
'svm': SVC(kernel='rbf', C=10, gamma='scale', probability=True)
```

**Wyniki klasyfikatorów:**
- **KNN:** Doskonałe rozpoznawanie wszystkich twarzy z wysoką odpornością na zakłócenia.
- **Naive Bayes:** Skuteczne rozpoznawanie Macieja i Jagody, wysoka odporność na zakłócenia.
- **Decision Tree:** Dobre wyniki rozpoznawania, ale mniejsza odporność na zakłócenia.
- **MLP:** Brak skuteczności w rozpoznawaniu twarzy.
- **SVM:** Poprawne rozpoznawanie Macieja i Jagody.

# Podsumowanie analizy wyników rozpoznawania twarzy

1. **KNN** - konsekwentnie wykazywał wysoką skuteczność we wszystkich zestawach:
   - Bardzo dobre rozpoznawanie znanych twarzy
   - Wysoka odporność na zakłócenia
   - Tendencja do sporadycznego klasyfikowania nieznanych twarzy jako znane

2. **Naive Bayes** - dobre wyniki w specyficznym zakresie:
   - Skutecznie rozpoznawał tylko twarze Macieja i Jagody
   - Prawdopodobnie można by było ulepszyć, wyrównując ilość zdjęć każdej osoby

3. **Decision Tree**:
   - Dobra skuteczność, szczególnie dla twarzy Macieja
   - Prawdopodobnie można by było ulepszyć, wyrównując ilość zdjęć każdej osoby

4. **MLP** - najmniej spójna wydajność:
   - W większości zestawów wykazywał brak skuteczności
   - W jednym szczegolnym wypadku okazał się być bardzo dobry.

5. **SVM**: - różne wyniki
   - Wysoka precyzja rozpoznawania znanych twarzy
   - Doskonała odporność na zakłócenia

## Wnioski końcowe

- KNN i SVM były najbardziej niezawodnymi klasyfikatorami w testach
- Augmentacja danych okazała się kluczowa dla poprawy wyników
- Rozpoznawanie twarzy Macieja i Jagody było łatwiejsze niż Matta Damona w większości testów
- Wyrównanie ilości danych dla każdej klasy (np. Maciej, Jagoda, Matt Damon) skutecznie poprawiłoby wyniki klasyfikatorów