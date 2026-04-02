# Projet Enedis : Classification des Résidences (Principales vs Secondaires)

## Objectif du projet
Ce projet de Data Science vise à analyser des données de consommation électrique (Enedis) afin de classifier automatiquement les compteurs en deux catégories : Résidences Principales et Résidences Secondaires. 

L'enjeu est de réduire un dataset brut (plus de 8 millions de relevés) en profils clients exploitables grâce à des techniques de Feature Engineering et de Machine Learning (Clustering non supervisé).

## Méthodologie
1. Prétraitement (Preprocessing) : Nettoyage des données, gestion des dates (format UTC) et création de variables temporelles (jour, soir, weekend).
2. Feature Engineering : Agrégation des données par client pour créer 8 variables explicatives, notamment :
   - Consommation moyenne et maximale
   - Puissance talonnée (bruit de fond)
   - Taux d'absence (consommation < 150W)
   - Ratio de consommation Weekend / Semaine
3. Clustering : Normalisation des données (StandardScaler) et application de l'algorithme K-Means (k=2).

## Résultats et KPI Metrics
Sur un échantillon de 500 résidences analysées, l'algorithme a identifié deux clusters distincts :

- Résidences Principales (Cluster 0) : 428 foyers (85.6% du parc).
- Résidences Secondaires (Cluster 1) : 72 foyers (14.4% du parc).

Différences clés observées :
- Économie d'énergie : Les résidences secondaires consomment en moyenne 22.1% de moins que les principales.
- Taux d'absence : Il s'élève à 34.9% pour les résidences secondaires, contre seulement 21.0% pour les principales.
- Profil journalier : Les résidences principales affichent des pics de consommation marqués (matin et soir), tandis que les résidences secondaires conservent un profil beaucoup plus plat.

## Installation et Exécution
1. Cloner le repository.
2. Activer l'environnement virtuel : `venv\Scripts\activate` (Windows)
3. Installer les dépendances : `pip install -r requirements.txt`
4. Lancer le notebook Jupyter pour visualiser les étapes et les graphiques (`matplotlib`, `seaborn`).
5. Lancer le Dashboard interactif : streamlit run dashboard.py
