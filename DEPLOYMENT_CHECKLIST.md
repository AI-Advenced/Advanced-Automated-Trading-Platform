# ✅ Checklist de Déploiement - Plateforme de Trading Automatisée

## 🎯 Pre-Déploiement (Obligatoire)

### 1. Vérifications Système
- [ ] **Docker installé** (version 20.10+)
- [ ] **Docker Compose installé** (version 2.0+)
- [ ] **8 GB RAM minimum** disponibles
- [ ] **20 GB d'espace disque** libre
- [ ] **Connexion Internet stable** (pour APIs exchanges)

### 2. Configuration des Clés API

#### Binance (Recommandé: commencer en testnet)
- [ ] Compte Binance créé
- [ ] API Key générée avec permissions:
  - [ ] ✅ Spot Trading
  - [ ] ✅ Read Info
  - [ ] ❌ Futures Trading (désactivé pour sécurité)
  - [ ] ❌ Withdrawals (désactivé pour sécurité)
- [ ] API Secret sauvegardée de manière sécurisée
- [ ] Mode testnet activé pour les tests (`BINANCE_TESTNET=true`)

#### Coinbase Pro (Optionnel)
- [ ] Compte Coinbase Pro créé
- [ ] API Key/Secret/Passphrase générés
- [ ] Mode sandbox activé pour les tests (`COINBASE_SANDBOX=true`)

#### Kraken (Optionnel)
- [ ] Compte Kraken créé
- [ ] API Key/Secret générés avec permissions limitées

### 3. Configuration Environnement
- [ ] Fichier `.env` créé à partir de `.env.example`
- [ ] Clés API configurées
- [ ] Paramètres de risque ajustés (conservateurs pour débuter)
- [ ] Vérification: aucune clé API dans les logs ou le code

## 🚀 Déploiement Initial

### 1. Installation Automatique
```bash
# Option 1: Script automatique (recommandé)
./scripts/setup.sh

# Option 2: Makefile
make quick-start

# Option 3: Manuel
cp .env.example .env
# Éditer .env avec vos clés API
docker-compose up -d
```

### 2. Vérifications Post-Déploiement
- [ ] **Tous les conteneurs démarrés**: `docker-compose ps`
- [ ] **API répond**: `curl http://localhost:8000/health`
- [ ] **Base de données connectée**: Vérifier les logs
- [ ] **Redis fonctionnel**: Tests de cache
- [ ] **Grafana accessible**: http://localhost:3000

### 3. Tests de Fonctionnement
```bash
# Status système
make status
# ou
curl -s http://localhost:8000/api/v1/status | jq

# Portfolio initial
curl -s http://localhost:8000/api/v1/portfolio | jq

# Test des exchanges (ne doit pas générer d'erreurs)
curl -s http://localhost:8000/api/v1/market/BTCUSDT | jq
```

## 🛡️ Configuration Sécurisée

### 1. Paramètres de Risque Ultra-Conservateurs
```env
# Dans le fichier .env
MAX_PORTFOLIO_RISK=0.005        # 0.5% max par trade
MAX_DAILY_DRAWDOWN=0.02         # 2% drawdown max par jour  
MAX_TOTAL_DRAWDOWN=0.05         # 5% drawdown total max
SIGNAL_CONFIDENCE_THRESHOLD=0.9  # Très haut seuil de confiance
MAX_POSITIONS=3                 # Maximum 3 positions
MAX_SIGNALS_PER_HOUR=2          # Limite les signaux
```

### 2. Mode Test Obligatoire
- [ ] `BINANCE_TESTNET=true` ✅
- [ ] `COINBASE_SANDBOX=true` ✅  
- [ ] `ENVIRONMENT=development` ✅
- [ ] `DEBUG=true` pour les premiers tests ✅

### 3. Surveillance Active
- [ ] **Grafana configuré**: http://localhost:3000
- [ ] **Alertes activées** pour drawdown > 3%
- [ ] **Logs en temps réel**: `make logs-app`
- [ ] **Métriques surveillées**: Performance, erreurs, latence

## 📊 Tests de Validation

### 1. Tests Automatisés
```bash
# Suite complète de tests
make test

# Tests spécifiques
make test-api      # Tests API REST
make test-ml       # Tests modèles ML
make test-unit     # Tests unitaires
```

### 2. Tests Manuels Obligatoires

#### Test 1: Collecte de Données
```bash
# Vérifier la collecte de données
curl "http://localhost:8000/api/v1/market/BTCUSDT?limit=10" | jq '.data | length'
# Doit retourner 10 points de données
```

#### Test 2: Génération de Signaux
```bash
# Vérifier la génération de signaux
curl "http://localhost:8000/api/v1/signals/active" | jq 'map(select(.confidence > 0.7)) | length'
# Doit retourner des signaux avec confiance > 70%
```

#### Test 3: Arrêt d'Urgence
```bash
# Test d'arrêt immédiat
curl -X POST "http://localhost:8000/api/v1/stop"
# Doit arrêter le trading immédiatement
```

### 3. Test de Charge API
```bash
# Test de charge basique (optionnel)
make load-test
# ou
ab -n 100 -c 5 http://localhost:8000/api/v1/status
```

## 🎛️ Mise en Production

### 1. Migration vers Comptes Réels (ATTENTION!)

⚠️ **SEULEMENT après validation complète en testnet**

```env
# Passage en mode production (avec extrême prudence)
BINANCE_TESTNET=false
COINBASE_SANDBOX=false
ENVIRONMENT=production
DEBUG=false

# Paramètres encore plus conservateurs pour débuter
MAX_PORTFOLIO_RISK=0.003        # 0.3% max par trade
MAX_DAILY_DRAWDOWN=0.01         # 1% drawdown max par jour
```

### 2. Configuration Production
- [ ] **SSL/HTTPS activé** pour l'accès distant
- [ ] **Firewall configuré** (ports 8000, 3000 uniquement)
- [ ] **Sauvegardes automatiques** configurées
- [ ] **Monitoring 24/7** en place
- [ ] **Alertes SMS/Email** configurées

### 3. Capital Initial Limité
- [ ] **Maximum 5-10%** du capital total au début
- [ ] **Positions minimales** pour tester
- [ ] **Surveillance manuelle** 24h les premiers jours
- [ ] **Plan d'arrêt d'urgence** préparé

## 📋 Monitoring et Maintenance

### 1. Surveillance Quotidienne
- [ ] **Performance dashboard**: Vérifier Grafana
- [ ] **Logs d'erreurs**: `make logs-app | grep ERROR`
- [ ] **Métriques clés**: Drawdown, win rate, latence
- [ ] **Santé des exchanges**: Connectivité APIs

### 2. Maintenance Hebdomadaire
- [ ] **Sauvegarde base de données**: `make backup`
- [ ] **Mise à jour dépendances**: `make update`
- [ ] **Analyse des performances**: Rapports Grafana
- [ ] **Réentraînement modèles ML**: Si performance dégradée

### 3. Commandes de Maintenance
```bash
# Vérification santé complète
make health

# Sauvegarde
make backup

# Redémarrage propre
make restart

# Nettoyage système
make clean
```

## 🚨 Procédures d'Urgence

### 1. Arrêt Immédiat du Trading
```bash
# Arrêt via API
curl -X POST "http://localhost:8000/api/v1/stop"

# Arrêt complet du système
make down

# Vérification manuelle des positions sur exchanges
```

### 2. En Cas de Problème Majeur
1. **ARRÊTER immédiatement** le système
2. **VÉRIFIER** les positions ouvertes sur les exchanges
3. **FERMER manuellement** les positions si nécessaire
4. **ANALYSER** les logs pour comprendre le problème
5. **NE PAS REDÉMARRER** avant résolution complète

### 3. Contacts d'Urgence
- Documentation: `/docs` endpoint
- Logs système: `make logs`
- Support communauté: GitHub Issues

## ✅ Checklist de Go-Live

### Phase 1: Tests (Semaine 1-2)
- [ ] ✅ Installation et configuration réussies
- [ ] ✅ Tests automatisés passés à 100%
- [ ] ✅ Tests manuels validés
- [ ] ✅ Mode testnet fonctionnel 24h sans erreur
- [ ] ✅ Monitoring opérationnel
- [ ] ✅ Procédures d'urgence testées

### Phase 2: Production Limitée (Semaine 3-4)
- [ ] ✅ Configuration ultra-conservative validée
- [ ] ✅ Capital limité (5% max)
- [ ] ✅ Surveillance manuelle continue
- [ ] ✅ Performance positive sur 7 jours minimum
- [ ] ✅ Aucun incident majeur

### Phase 3: Montée en Charge (Mois 2+)
- [ ] ✅ Performance validée sur 30 jours
- [ ] ✅ Optimisation des paramètres
- [ ] ✅ Augmentation progressive du capital
- [ ] ✅ Automatisation complète

## ⚠️ Avertissements Critiques

1. **JAMAIS** utiliser plus de 10% de votre capital total
2. **TOUJOURS** commencer en mode testnet
3. **SURVEILLER** activement les premiers jours
4. **COMPRENDRE** que le trading automatisé comporte des risques
5. **AVOIR** un plan d'arrêt d'urgence
6. **NE JAMAIS** trader avec de l'argent que vous ne pouvez pas perdre

---

## 📞 Support et Ressources

- **Documentation API**: http://localhost:8000/docs
- **Guide de démarrage**: [QUICK_START.md](QUICK_START.md)
- **Documentation complète**: [README.md](README.md)
- **Structure du projet**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

**Bon trading responsable ! 🚀**

*La sécurité avant la performance - Always.*