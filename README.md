# 🚀 Automated Trading Platform

**Une plateforme de trading automatisée avancée avec intelligence artificielle et analyse technique intégrée**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Vue d'ensemble

Cette plateforme de trading automatisée utilise des algorithmes d'intelligence artificielle avancés et l'analyse technique pour exécuter des trades automatiques sur les marchés de crypto-monnaies. Elle intègre plusieurs modèles ML, plus de 40 indicateurs techniques, et une gestion des risques sophistiquée.

### 🌟 Fonctionnalités principales

- **🤖 Moteur IA/ML Avancé**: Random Forest, LSTM, Gradient Boosting
- **📊 Analyse Technique Complète**: 40+ indicateurs (RSI, MACD, Bollinger Bands, etc.)
- **🔄 Support Multi-Exchange**: Binance, Coinbase Pro, Kraken
- **⚡ Trading en Temps Réel**: WebSocket feeds et exécution automatique
- **🛡️ Gestion des Risques**: Position sizing dynamique, corrélations, drawdown protection
- **📈 Monitoring Avancé**: Métriques de performance, dashboards Grafana
- **🐳 Containerisation Docker**: Déploiement facile avec monitoring complet
- **🔒 Sécurité Intégrée**: JWT authentication, rate limiting, HTTPS

## 🏗️ Architecture Technique

### Stack Technologique

- **Backend**: Python 3.11, FastAPI, SQLAlchemy, Redis
- **Base de Données**: PostgreSQL avec InfluxDB pour les données temporelles
- **ML/IA**: Scikit-learn, TensorFlow, TA-Lib pour l'analyse technique
- **Monitoring**: Prometheus, Grafana, ELK Stack (optionnel)
- **Containerisation**: Docker, Docker Compose
- **Proxy**: Nginx avec SSL/TLS

### Architecture des Composants

```
📦 backend/
├── 🧠 core/               # Composants principaux du trading
│   ├── data_collector.py  # Collecte multi-exchange
│   ├── ml_models.py       # Modèles ML (RF, LSTM, GB)
│   ├── signal_generator.py # Génération de signaux
│   ├── portfolio_manager.py # Gestion du portefeuille
│   ├── trading_watcher.py # Monitoring des performances
│   └── main.py           # Orchestrateur principal
├── 🌐 api/               # API REST FastAPI
│   └── app.py            # Endpoints et WebSocket
├── 🗄️ models/           # Modèles de données
│   ├── database.py       # ORM SQLAlchemy
│   └── schemas.py        # Validation Pydantic
├── ⚙️ config/           # Configuration
│   └── settings.py       # Paramètres et environnement
├── 🧪 tests/            # Tests automatisés
├── 🔧 utils/            # Utilitaires
├── 📊 migrations/       # Migrations de base de données
└── 📜 scripts/          # Scripts d'initialisation
```

## 🚀 Installation et Déploiement

### Pré-requis

- Docker et Docker Compose
- Python 3.11+ (pour développement local)
- Clés API des exchanges (Binance, Coinbase Pro, Kraken)

### 🐳 Déploiement Docker (Recommandé)

1. **Cloner le repository**:
```bash
git clone <repository-url>
cd webapp
```

2. **Configuration des variables d'environnement**:
```bash
cp .env.example .env
# Éditer .env avec vos clés API et configuration
```

3. **Démarrage en production**:
```bash
docker-compose up -d
```

4. **Démarrage en développement**:
```bash
docker-compose --profile dev up -d
```

### 🖥️ Installation Locale

1. **Installation des dépendances**:
```bash
pip install -r requirements.txt
```

2. **Configuration de la base de données**:
```bash
python -m backend.migrations.migrate
python -m backend.scripts.init_db --test-data
```

3. **Démarrage du serveur**:
```bash
uvicorn backend.api.app:app --reload --port 8000
```

## 🔧 Configuration

### Variables d'Environnement Principales

```env
# Exchanges API
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
COINBASE_API_KEY=your_coinbase_key

# Trading Settings
MAX_PORTFOLIO_RISK=0.02          # 2% risque max par trade
MAX_DAILY_DRAWDOWN=0.05          # 5% drawdown journalier max
SIGNAL_CONFIDENCE_THRESHOLD=0.7   # Seuil de confiance des signaux

# ML Model Settings
MODEL_RETRAIN_INTERVAL=24        # Réentraînement toutes les 24h
MIN_PREDICTION_CONFIDENCE=0.6    # Confiance minimum des prédictions
```

### Configuration des Stratégies

```python
# Configuration dans backend/config/settings.py
STRATEGIES = {
    "ma_crossover": {"enabled": True, "weight": 0.3},
    "rsi": {"enabled": True, "weight": 0.25},
    "macd": {"enabled": True, "weight": 0.25},
    "ml_prediction": {"enabled": True, "weight": 0.2}
}
```

## 📊 Utilisation

### API REST

La plateforme expose une API REST complète accessible à `http://localhost:8000`:

- **Documentation**: `/docs` (Swagger UI)
- **Status**: `GET /api/v1/status`
- **Portfolio**: `GET /api/v1/portfolio`
- **Trades**: `GET /api/v1/trades`
- **Signaux**: `GET /api/v1/signals`
- **Performance**: `GET /api/v1/performance`

### WebSocket en Temps Réel

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Mise à jour en temps réel:', data);
};
```

### Dashboards de Monitoring

- **Grafana**: `http://localhost:3000` (admin/admin_password)
- **Prometheus**: `http://localhost:9090`
- **Application**: `http://localhost:8000`

## 🧠 Modèles ML et Stratégies

### Modèles d'Intelligence Artificielle

1. **Random Forest Classifier**
   - Prédiction direction du prix (hausse/baisse)
   - Features: 40+ indicateurs techniques
   - Réentraînement automatique

2. **LSTM Neural Network**
   - Analyse de séquences temporelles
   - Prédiction des mouvements futurs
   - Architecture deep learning

3. **Gradient Boosting**
   - Ensemble method pour robustesse
   - Optimisation des hyperparamètres
   - Feature importance analysis

### Indicateurs Techniques (40+)

- **Momentum**: RSI, Stochastic, Williams %R
- **Trend**: SMA, EMA, MACD, ADX
- **Volatilité**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, Volume SMA, VWAP
- **Oscillateurs**: CCI, MFI, ROC

### Génération de Signaux

```python
# Exemple de signal composite
signal = {
    "symbol": "BTCUSDT",
    "action": "BUY",
    "confidence": 0.85,
    "price": 50000.0,
    "strategies": {
        "technical": 0.7,
        "ml_prediction": 0.9,
        "momentum": 0.8
    }
}
```

## 🛡️ Gestion des Risques

### Méthodes de Sizing

- **Kelly Criterion**: Optimisation mathématique
- **Volatilité Ajustée**: Basé sur l'ATR
- **Corrélation**: Évite les positions corrélées
- **Drawdown Protection**: Arrêt automatique

### Limits et Controls

```python
RISK_LIMITS = {
    "max_position_size": 0.1,      # 10% max par position
    "max_sector_exposure": 0.3,     # 30% max par secteur
    "max_correlation": 0.7,         # Corrélation max 70%
    "var_limit": 0.05,              # VaR 5%
    "expected_shortfall": 0.08      # ES 8%
}
```

## 📈 Monitoring et Performance

### Métriques Clés

- **Rendement Total**: Performance globale
- **Sharpe Ratio**: Rendement ajusté au risque
- **Drawdown Maximum**: Perte maximale
- **Taux de Réussite**: % de trades gagnants
- **Profit Factor**: Ratio gains/pertes

### Alertes Automatiques

- Drawdown excessif (-10%)
- Performance dégradée
- Erreurs API fréquentes
- Latence élevée (>1s)

## 🧪 Tests et Qualité

### Suite de Tests

```bash
# Tous les tests
pytest backend/tests/ -v

# Tests spécifiques
pytest backend/tests/test_ml_models.py -v
pytest backend/tests/test_api.py -v

# Couverture de code
pytest --cov=backend --cov-report=html
```

### Tests Inclus

- **Tests Unitaires**: Chaque composant
- **Tests d'Intégration**: API et base de données
- **Tests de Performance**: Charge et latence
- **Tests ML**: Validation des modèles
- **Mocks**: Exchanges et APIs externes

## 🔒 Sécurité

### Mesures de Sécurité

- **Authentification JWT**: Tokens sécurisés
- **Rate Limiting**: Protection contre le spam
- **Validation des Données**: Pydantic schemas
- **HTTPS/TLS**: Chiffrement des communications
- **Secrets Management**: Variables d'environnement

### Bonnes Pratiques

- Clés API en variables d'environnement
- Logs structurés (pas de secrets)
- Validation stricte des entrées
- Timeouts et retry logic
- Monitoring des erreurs

## 🐛 Dépannage

### Problèmes Courants

1. **Erreur de connexion Exchange**
   - Vérifier les clés API
   - Vérifier les permissions
   - Tester la connectivité

2. **Performance ML dégradée**
   - Réentraîner les modèles
   - Vérifier la qualité des données
   - Ajuster les hyperparamètres

3. **Latence élevée**
   - Vérifier la charge système
   - Optimiser les requêtes DB
   - Ajuster les timeouts

### Logs et Debugging

```bash
# Logs de l'application
docker-compose logs trading-app

# Logs spécifiques
docker-compose logs trading-app | grep ERROR

# Mode debug
export LOG_LEVEL=DEBUG
```

## 📚 Documentation Technique

### Structure des Données

```python
# Signal de Trading
{
    "symbol": "BTCUSDT",
    "action": "BUY|SELL|HOLD",
    "confidence": 0.85,
    "price": 50000.0,
    "timestamp": "2023-01-01T00:00:00Z",
    "indicators": {...},
    "ml_predictions": {...}
}

# Position du Portfolio
{
    "symbol": "BTCUSDT",
    "quantity": 0.1,
    "avg_price": 49500.0,
    "current_price": 50000.0,
    "pnl": 50.0,
    "pnl_percentage": 1.01
}
```

### Endpoints API Principaux

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/v1/status` | GET | Statut du système |
| `/api/v1/portfolio` | GET | Résumé du portefeuille |
| `/api/v1/trades` | GET | Historique des trades |
| `/api/v1/signals` | GET | Signaux de trading |
| `/api/v1/performance` | GET | Métriques de performance |
| `/api/v1/start` | POST | Démarrer le trading |
| `/api/v1/stop` | POST | Arrêter le trading |

## 🤝 Contribution

### Développement

1. **Fork** le repository
2. **Créer** une branche feature
3. **Développer** avec tests
4. **Tester** la suite complète
5. **Soumettre** une pull request

### Standards de Code

```bash
# Formatage
black backend/
isort backend/

# Linting
flake8 backend/
mypy backend/

# Tests
pytest backend/tests/ --cov=backend
```

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/username/trading-platform/issues)
- **Documentation**: `/docs` endpoint
- **Email**: support@tradingplatform.com

---

## ⚠️ Avertissement

Cette plateforme est destinée à des fins éducatives et de recherche. Le trading de crypto-monnaies comporte des risques importants de perte financière. Utilisez cette plateforme à vos propres risques et ne tradez jamais avec de l'argent que vous ne pouvez pas vous permettre de perdre.

**Testez toujours en mode sandbox/testnet avant d'utiliser de vrais fonds.**

---

*Développé avec ❤️ pour la communauté des traders algorithmiques*