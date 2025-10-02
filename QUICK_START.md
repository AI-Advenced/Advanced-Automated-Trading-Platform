# 🚀 Guide de Démarrage Rapide - Plateforme de Trading Automatisée

## ⚡ Démarrage en 5 Minutes

### 1. Prérequis Minimum

- **Docker** et **Docker Compose** installés
- **8 GB RAM** minimum recommandés
- **Clés API** des exchanges (mode testnet pour débuter)

### 2. Installation Express

```bash
# 1. Cloner le projet
git clone <repository-url>
cd webapp

# 2. Configuration rapide
cp .env.example .env
# Éditer .env avec vos clés API (voir section Configuration)

# 3. Lancement complet
docker-compose up -d

# 4. Attendre l'initialisation (2-3 minutes)
docker-compose logs -f trading-app
```

### 3. Vérification

```bash
# Vérifier que tous les services sont en marche
docker-compose ps

# Tester l'API
curl http://localhost:8000/health

# Accéder à l'interface
open http://localhost:8000/docs
```

## 🔧 Configuration Minimale Requise

### Variables d'Environnement Essentielles

Éditez le fichier `.env`:

```env
# 🔑 Clés API des Exchanges (MODE TESTNET RECOMMANDÉ)
BINANCE_API_KEY=your_binance_testnet_key
BINANCE_SECRET_KEY=your_binance_testnet_secret
BINANCE_TESTNET=true

COINBASE_API_KEY=your_coinbase_sandbox_key
COINBASE_SECRET_KEY=your_coinbase_sandbox_secret
COINBASE_SANDBOX=true

# 📊 Configuration Trading de Base
MAX_PORTFOLIO_RISK=0.01          # 1% risque par trade (sécurisé)
MAX_DAILY_DRAWDOWN=0.03          # 3% drawdown max par jour
SIGNAL_CONFIDENCE_THRESHOLD=0.8   # Seuil élevé pour plus de sécurité

# 🗄️ Base de Données (laisser par défaut)
DATABASE_URL=postgresql://trading_user:trading_password@postgres:5432/trading_db
REDIS_URL=redis://redis:6379/0
```

### 🔑 Obtenir vos Clés API

#### Binance Testnet (Recommandé pour débuter)
1. Aller sur [Binance Testnet](https://testnet.binance.vision/)
2. Se connecter avec GitHub
3. Générer API Key/Secret
4. **Important**: Activer le trading spot

#### Coinbase Pro Sandbox
1. Aller sur [Coinbase Pro Sandbox](https://public.sandbox.pro.coinbase.com/)
2. Créer un compte sandbox
3. Générer API credentials
4. Noter la passphrase

## 🎯 Premier Test de Trading

### 1. Démarrage du Système

```bash
# Démarrer tous les services
docker-compose up -d

# Vérifier les logs
docker-compose logs trading-app
```

### 2. Vérification via API

```bash
# Status du système
curl -X GET "http://localhost:8000/api/v1/status" | jq

# Portfolio initial
curl -X GET "http://localhost:8000/api/v1/portfolio" | jq

# Signaux récents
curl -X GET "http://localhost:8000/api/v1/signals?limit=5" | jq
```

### 3. Interface Web

Accédez à ces URLs:

- **API Documentation**: http://localhost:8000/docs
- **Grafana Monitoring**: http://localhost:3000 (admin/admin_password)
- **Application Health**: http://localhost:8000/health

### 4. Démarrage du Trading (Mode Sécurisé)

```bash
# Démarrer le trading automatique
curl -X POST "http://localhost:8000/api/v1/start"

# Surveiller les performances
curl -X GET "http://localhost:8000/api/v1/performance"

# Arrêter le trading si nécessaire
curl -X POST "http://localhost:8000/api/v1/stop"
```

## 📊 Monitoring Essentiel

### Dashboards Principaux

1. **Grafana** (http://localhost:3000):
   - Login: admin / admin_password
   - Dashboards de performance temps réel
   - Alertes automatiques

2. **API Documentation** (http://localhost:8000/docs):
   - Interface interactive Swagger
   - Test direct des endpoints
   - Schémas de données

3. **Prometheus** (http://localhost:9090):
   - Métriques système
   - Historique des performances

### Métriques Clés à Surveiller

```bash
# Performance globale
curl "http://localhost:8000/api/v1/performance" | jq '.total_return, .sharpe_ratio, .max_drawdown'

# Positions actives
curl "http://localhost:8000/api/v1/portfolio/positions" | jq '.positions | length'

# Signaux récents (dernière heure)
curl "http://localhost:8000/api/v1/signals/active" | jq 'map(select(.confidence > 0.7))'

# Status des modèles ML
curl "http://localhost:8000/api/v1/models/status" | jq
```

## 🛡️ Mode Sécurité pour Débutants

### Configuration Ultra-Conservative

```env
# Risques minimaux
MAX_PORTFOLIO_RISK=0.005         # 0.5% par trade seulement
MAX_DAILY_DRAWDOWN=0.02          # 2% drawdown max
MAX_TOTAL_DRAWDOWN=0.05          # 5% drawdown total max
SIGNAL_CONFIDENCE_THRESHOLD=0.9   # Très haut seuil de confiance

# Trading limité
MAX_POSITIONS=3                  # Maximum 3 positions
DEFAULT_LEVERAGE=1.0             # Pas de levier
MAX_SIGNALS_PER_HOUR=2           # Limite les signaux
```

### Checklist de Sécurité

- ✅ **TOUJOURS** commencer en mode testnet
- ✅ **JAMAIS** utiliser plus de 1% de capital par trade
- ✅ **SURVEILLER** les logs en continu les premiers jours
- ✅ **TESTER** l'arrêt d'urgence (`/api/v1/stop`)
- ✅ **VÉRIFIER** les métriques de drawdown régulièrement

## 🚨 Arrêt d'Urgence

Si quelque chose ne va pas:

```bash
# Arrêt immédiat du trading
curl -X POST "http://localhost:8000/api/v1/stop"

# Arrêt complet du système
docker-compose down

# Vérifier les positions ouvertes manuellement sur l'exchange
# Fermer manuellement si nécessaire
```

## 🔍 Dépannage Rapide

### Problème: Services ne démarrent pas

```bash
# Vérifier les logs
docker-compose logs

# Redémarrer proprement
docker-compose down
docker-compose up -d
```

### Problème: Erreurs API Exchange

```bash
# Vérifier la configuration
docker-compose exec trading-app python -c "
from backend.config.settings import get_settings
settings = get_settings()
print('Binance Testnet:', settings.binance_testnet)
print('API Key exists:', bool(settings.binance_api_key))
"

# Tester la connectivité
curl "http://localhost:8000/api/v1/market/BTCUSDT"
```

### Problème: Base de données

```bash
# Recréer la base de données
docker-compose down -v
docker-compose up -d
```

## 📈 Évolution Progressive

### Phase 1: Test (Semaine 1)
- Mode testnet uniquement
- Surveillance manuelle
- Paramètres ultra-conservateurs

### Phase 2: Optimisation (Semaine 2-4)
- Ajustement des paramètres
- Analyse des performances
- Backtesting des stratégies

### Phase 3: Production (Après validation)
- Passage en mode réel
- Capital limité (5-10% max)
- Surveillance continue

## 📞 Support Rapide

### Commandes Utiles

```bash
# Status complet
docker-compose ps && curl -s http://localhost:8000/health | jq

# Logs en temps réel
docker-compose logs -f trading-app

# Redémarrage propre
docker-compose restart trading-app

# Sauvegarde des données
docker-compose exec postgres pg_dump -U trading_user trading_db > backup.sql
```

### Ressources

- **Documentation complète**: [README.md](README.md)
- **API Interactive**: http://localhost:8000/docs
- **Tests automatisés**: `pytest backend/tests/ -v`

---

## ⚠️ Rappel Important

**Cette plateforme est en mode éducatif/recherche. Commencez TOUJOURS avec:**

1. **Mode testnet** des exchanges
2. **Montants minimaux** si vous passez en réel
3. **Surveillance constante** les premiers jours
4. **Compréhension** des risques inhérents au trading

**Le trading automatisé peut entraîner des pertes importantes. Utilisez uniquement de l'argent que vous pouvez vous permettre de perdre.**

---

*Bon trading ! 🚀*