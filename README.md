# jev-tactics

Perception par vision d'un jeu isometrique au tour par tour, et **decision confiee a
Jev** — le modele « System One » de TypeSafe AI, qui rend des sorties typees (choix,
score, probabilite) plutot que du texte.

> **Cadre : recherche.** Vision par ordinateur sur rendu isometrique, extraction d'etat
> de jeu, et planification tactique sous contraintes. Evaluation en local et en
> simulation. **Non destine aux serveurs officiels**, dont les conditions d'utilisation
> interdisent l'automatisation.

La perception, la legalite des coups et l'execution viennent d'un projet anterieur et
fonctionnent. Ce qui est neuf ici, c'est **qui decide**.

## Le partage des roles

```
CombatState ──▶ legal_actions ──▶ top_sequences ──▶ [plans] ──▶ Jev ──▶ indice ──▶ Plan
                  (legalite)        (elagage)                  (choix)
```

- **le code enumere** — un plan illegal n'est pas rejete, il n'est pas construit ;
- **le code elague** — on ne soumet pas 10⁸ sequences, on en soumet huit ;
- **Jev tranche** — et rend un *identifiant*, borne-verifie, jamais un coup.

Jev ne remplacera jamais le solveur. L'espace d'actions vaut `1 + cases × (1 + sorts)`,
soit **5 233** indices en configuration reelle (218 cases, 23 sorts) ; une question
`choice` en accepte 255, et un tour n'est pas une action mais une **sequence**. Ce que
Jev remplace, c'est `scoring.py` : 14 ko de poids regles a la main pour departager des
plans — soit exactement une tache de jugement typee.

## Etat

| Etage | Etat |
|---|---|
| Perception (ATH, plateau, entites, timeline, barre de sorts) | ✅ porte |
| Solveur 1 tour, recherche elaguee | ✅ porte |
| Simulateur + arene de mesure | ✅ porte |
| Couche de decision, decideur de reference, repli | ✅ |
| Decideur Jev, projection d'etat, transport a cassettes | ✅ |
| Plans candidats **reellement distincts** | ❌ **bloquant**, cf. `docs/PLAN.md` |
| Deplacement : graphe de cartes, trajets, boucle de marche | ✅ hors-ligne |
| Engagement d'un combat | ⬜ a extraire |
| Quetes et PNJ | ⬜ **rien n'existe** |

**792 tests passent, 220 sautent, 0 echec.** Aucun ne demande le jeu, ni le reseau, ni une
cle API.

### Ce que le chiffre ne dit pas

Les 220 tests sautes demandent des **captures non versionnees**. Sur un clone neuf, la
suite couvre donc un quart de moins que sur la machine d'origine. Commiter un corpus cure
est au plan (`docs/PLAN.md`, P4) ; d'ici la, ne pas lire « vert » comme « couvert ».

## Le faire jouer

```bash
# 1. la perception voit-elle quelque chose ? (jeu au premier plan)
python scripts/observe.py --live
python scripts/inspect_board.py --live

# 2. se deplacer -- DRY-RUN par defaut, rien ne bouge
python scripts/play.py --goto 7,3
python scripts/play.py --goto 7,3 --execute

# 3. combattre (en combat, avec la barre de sorts decrite)
python scripts/play.py --fight --keys Ravage=1,Hostilite=2 --execute
```

`--execute` demande l'extra `action` : `pip install -e ".[action]"`.

Les scripts `--live` attendent que la fenetre du jeu soit au premier plan avant de
capturer. Ceux qui prennent un fichier (`read_ui.py`, `bench.py --image`) travaillent sur
une capture stockee, pas sur l'ecran.

**Dry-run par defaut.** Le script clique et appuie sur des touches dans une fenetre de
jeu ; le defaut inverse transformerait une faute de frappe en session pilotee. Il attend
que le jeu soit au premier plan avant de capturer, et refuse de piloter si le moniteur
capture ne commence pas a (0, 0) -- sinon chaque clic partirait decale d'une constante,
souvent sur l'autre ecran, avec une detection parfaite et zero combat.

Les quetes n'y sont pas : le directeur sait quoi faire, il lui manque des yeux.

## Mesure

Le chiffre a battre depend du **regime**, et c'est la nuance la plus facile a rater :

| Regime | Taux de victoire (40 combats, graine 0) |
|---|---|
| PV ennemis **connus** | 80 % |
| PV ennemis **inconnus** | **60 %** |

Sur 32 captures reelles : **72 ennemis detectes, zero avec des PV lisibles.** Ni l'OCR
(reserve au joueur) ni la timeline (un ratio sans maximum) ne les donnent. Le regime que
le bot joue reellement est celui a **60 %** — et tous les poids de `scoring.py` ont ete
regles dans l'autre.

## Demarrage

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[dev]"                              # suffit pour tout tester
pytest -q
```

Extras, au besoin : `jev` (appels reels, `typesafe-sdk`), `ocr` (fabrique les gabarits de
chiffres), `action` (pilotage souris/clavier).

### La cle Jev

```bash
cp .env.example .env                  # puis renseigner TYPESAFE_API_KEY
python scripts/jev_ping.py --check    # verifie cle + SDK, SANS appeler
python scripts/jev_ping.py            # un appel reel, quelques millimes
```

L'environnement reel a **priorite** sur `.env` : une variable deja exportee n'est jamais
ecrasee, pour que la CI et les essais ponctuels gardent la main. `.env` est gitignore.

`jev_ping.py` verifie trois choses qui cassent independamment : que la cle est lue, que la
forme de la reponse correspond a ce que `transport._from_sdk_result` attend — ecrit
d'apres la documentation, **sans jamais voir le SDK** — et la latence reelle, qui n'est
pas mesuree alors que les budgets de tour sont poses par analogie.

**Les tests n'ont jamais besoin de cle** : `ReplayTransport` rejoue des cassettes, dont la
cle est un hachage de la requete — un refactor de la projection invalide la cassette au
lieu de rejouer une reponse perimee.

## Securite

Une sortie typee **n'est pas une frontiere de securite**. Check Point a montre
(septembre 2026) qu'un modele de decision se manipule par son *entree*, avec ~59 % de
verdicts retournes.

La defense ici n'est pas la validation apres coup, c'est la construction : Jev choisit un
**indice** dans une liste que le code a produite. Et rien de controle par un tiers n'entre
dans l'etat envoye — les `entity_id` sont synthetiques (`"me"`, `"enemy_42"`), les noms de
sorts viennent de la configuration de l'operateur. Aucun pseudonyme, aucun message de
joueur. `projection.assert_untrusted_free` en fait une verification executable, qui **leve**
plutot que de filtrer.

## Documents

- [`CLAUDE.md`](CLAUDE.md) — les invariants, la discipline de mesure, les pieges connus
- [`docs/PLAN.md`](docs/PLAN.md) — l'etat audite et la suite du travail

## Licence

MIT, cf. [`LICENSE`](LICENSE).
