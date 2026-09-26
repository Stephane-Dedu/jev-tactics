# Plan

Revision du 26 septembre 2026. Le plan initial a ete ecrit avant que quoi que ce soit
n'ait tourne ; celui-ci part d'un audit execute. Chaque chiffre cite ci-dessous a ete
mesure sur ce depot, pas repris de l'amont.

---

## Etat au 26 septembre 2026

La suite tourne **verte** : **667 passes, 217 sautes, 0 echec** en 73 s.

Le portage « combat seul » depuis le depot d'origine est clos : `state`, `calibration`,
`perception`, `capture`, `action`, `planner`, `rules`, `rl`, `sim`, `pipeline`, plus 21
scripts compatibles. Exclus : `bot`, `farming`, `market`, `ui`, `diagnose`, `overlay`.

La couche de decision existe et est cablee a l'arene :

```
CombatState ──▶ legal_actions ──▶ top_sequences ──▶ [plans] ──▶ Jev ──▶ indice ──▶ Plan
                  (legalite)        (elagage)                  (choix)
```

### Corrige pendant l'audit

| Defaut | Preuve | Etat |
|---|---|---|
| `scripts/` jamais copie | 2 tests non collectables | copie, 21 scripts |
| `test_endurance`, `test_play_turn_guards` retenus a tort | leurs scripts importent `bot`/`farming`/`overlay` | retires |
| `ap_left` annoncait les PA intacts | plan depensant 6 PA -> `ap_left=6`, attendu `0` | corrige, `RUF021` etait deja rouge |
| `top_sequences` sans repli d'approche | 1/60 scenarios : `candidates[0]` ne lance rien la ou le solveur engage | corrige, **60/60** identiques |
| Tables d'audit visant `farming`/`bot` | 2 tests en echec sur un depot qui ne porte que le combat | frontiere de portage explicite, les cibles restees en amont sont tues, les autres echouent toujours |

### Ouvert

| # | Defaut | Preuve |
|---|---|---|
| **D1** | **Les plans candidats ne sont pas distincts** | `sacrieur.json`, 20 sorts : **16 options sur 24 redondantes**, degats `93-93`, memes sorts sur les memes cibles. Seule la case d'arrivee differe |
| D2 | Aucun test sur `decision/` ni `candidates.py` | — |
| D3 | 217 tests sautent sur un clone neuf | captures non versionnees ; la CI serait verte en couvrant un quart de moins |
| D4 | `doctor.py` et `spells.py` absents | bloques par `diagnose` -> `farming` |
| D5 | `_from_sdk_result` devine la forme du SDK | jamais confronte a `typesafe-sdk` reel |
| D6 | 29 erreurs ruff, 31 mypy | l'amont en compte 89 ; aucune CI ne l'a jamais vu |

**D1 est bloquant.** Demander a Jev de trancher entre quatre descriptions identiques est
un appel paye pour rien. Et le choix compte : forcer `plan_0` donne **62 %** de victoires,
forcer le dernier **35 %** -- vingt-sept points separent deux options que rien ne
distingue dans l'enonce.

---

## P1 — Des postures, pas des variantes  *(bloquant)*

Remplacer le top-K par **un plan par intention tactique**. Le solveur continue d'optimiser ;
ce qu'on demande a Jev, c'est **sous quel objectif** optimiser.

Postures visees, chacune produite par un `evaluate` pondere differemment :

| Posture | Objectif |
|---|---|
| `degats_max` | maximiser les degats totaux |
| `achever` | maximiser les mises a mort (ennemi passant a 0) |
| `sur` | maximiser la distance finale au plus proche ennemi |
| `position` | se placer pour le tour suivant, degats secondaires |
| `econome` | conserver PA/PM, ne rien gaspiller |

Critere d'acceptation, mesurable :

- sur 60 scenarios `sacrieur.json`, **au moins 3 postures distinctes** par tour en
  mediane, ou l'appel est saute ;
- deux postures ne se ressemblent jamais : `(cast-set, degats, distance finale)` differe ;
- **`postures[0]` reste identique a `best_sequence`** sur 60/60 scenarios -- c'est la
  garantie que le repli vaut toujours la reference.

Ne pas toucher a `scoring.py` ni a `search.py` : la regle ne change pas pendant qu'on
change le joueur.

## P2 — Le banc de comparaison

Un script qui fait jouer plusieurs decideurs **sur les memes graines**, dans **les deux
regimes**, et imprime l'ecart.

```bash
python scripts/compare_deciders.py --fights 200 --spells configs/spells/sacrieur.json
```

Sortie attendue : par decideur, taux de victoire a PV connus **et** inconnus, tours
medians, PV restants, profil des defaites, part de replis, et **part des tours ou le
decideur s'ecarte du rang 0** -- un decideur qui suit toujours l'heuristique ne justifie
pas son appel.

Reference a battre : **60 %**, `enemy_hp_known=False`.

## P3 — Cassettes et premier appel reel

1. Confronter `_from_sdk_result` au vrai `typesafe-sdk` (**D5**) et corriger la forme.
2. Enregistrer des cassettes sur ~50 tours varies via `RecordingTransport`.
3. Rejouer en CI. Aucun test ne touche le reseau.
4. Mesurer **latence et cout reels** : la documentation annonce 120 req/min et
   0,042 $/M tokens en entree. A ~2 000 tokens d'etat et ~30 decisions par combat, cela
   fait de l'ordre de **0,25 centime par combat**. Le cout n'est pas le sujet ; la
   latence par tour l'est.

## P4 — Corpus de captures et CI  *(decide : sous-ensemble cure)*

Commiter un jeu **reduit et recadre** de captures couvrant les 217 tests qui sautent, puis
brancher la CI. Recadrer aux ROI utiles plutot que verser des captures plein ecran : plus
leger, et cela limite ce qui est redistribue.

CI : `ruff`, `mypy`, `pytest` sur Python 3.11, sans `typesafe-sdk`.
Cliquet qualite (**D6**) : `docs/QUALITY_BASELINE.txt` enregistre les comptes du jour ;
la CI echoue sur toute erreur **nouvelle**, et la base ne fait que decroitre.

Critere : **moins de 20 tests sautes** sur un clone neuf.

## P5 — Diagnostics de combat  *(decide : scinder `diagnose`)*

Extraire de `diagnose.py` (47 ko) la moitie combat/perception, laisser la moitie
recolte/chasse. Cela debloque `doctor.py` et `spells.py`.

`spells.py --live` est le plus urgent des deux : un `slot` errone retire un sort sans rien
lever, le bot cesse de le lancer, et l'on ne voit qu'un bot prudent.

## P6 — Publication

`LICENSE` (MIT, deja declare dans `pyproject.toml`), `README.md`, `git init`, premier
commit, depot public.

A trancher avant de pousser : le nom. `jev-tactics` est neutre ; le depot cible un jeu
dont les conditions d'utilisation interdisent l'automatisation, et les depots publics de
bots pour ce jeu ont deja fait l'objet de retraits. Le cadrage recherche du README
d'origine est conserve.

---

## Objectif — un bot qui joue

Le perimetre « combat seul » est leve. La cible est un bot qui **joue**, dans cet ordre
d'importance :

1. **se deplacer sur la carte**
2. **engager un combat**
3. **gagner le combat avec les bons sorts**
4. puis, une fois les trois acquis : **bout en bout** — prendre une quete, la faire, la
   rendre au bon PNJ.

### G1 — Se deplacer  *(en cours)*

| Brique | Etat |
|---|---|
| Graphe de cartes, trajets, circuits (`world/navigation.py`) | **porte, 93 tests verts** |
| Ou cliquer pour sortir (`world/travel.edge_point`) | **extrait, teste** |
| Juger le deplacement (`world/travel.judge_move`) | **extrait, teste** |
| Lecture des coordonnees (`perception/coordinates`) | deja la |
| Boucle : cliquer, attendre, confirmer, rembobiner | **a ecrire** |

`navigation.py` vivait sous `farming/` sans rien en importer : le ranger la rendait le
deplacement inaccessible a tout ce qui n'etait pas de la recolte. Il est maintenant sous
`world/`.

L'extraction de `travel` separe ce qui etait mele dans `farming/runner.py` : **decider ou
cliquer** et **juger ce qui s'est passe** d'un cote, la comptabilite de la recolte de
l'autre. `judge_move` rend trois issues — arrive / derive / immobile — et non un booleen,
parce que la **derive** est le mode de panne qui ne se voit pas : cliquer un bord fait
marcher le personnage, un obstacle le fait sortir ailleurs, la comparaison d'images dit
« reussi », et le circuit repart d'une carte decalee qu'il ne rattrapera jamais.

Reste : la boucle qui enchaine clic -> attente -> confirmation -> rembobinage, sans la
session de recolte. Puis un `Walker` qui suit un trajet du graphe jusqu'a une carte cible.

### G2 — Engager un combat

`perception/monsters.py` (72 ko) et `perception/entities.py` sont deja portes. Manque la
logique d'engagement, aujourd'hui dans `farming/runner.py` et `bot/orchestrator.py` :
reperer un groupe, juger s'il est engageable, cliquer, confirmer l'entree en combat.

A extraire de la meme facon que `travel` — la decision d'engager est un bon candidat pour
une question `noul` a Jev (« ce groupe vaut-il d'etre engage ? »), une fois la perception
branchee.

### G3 — Gagner le combat  *(bloque par D1)*

C'est P1 ci-dessus : les plans candidats ne sont pas distincts, donc Jev n'a rien a
trancher. Rien d'autre ne bloque — la boucle percevoir -> decider -> agir tourne.

### G4 — Quetes et PNJ  *(rien n'existe)*

**Aucune brique n'existe, dans aucun des deux depots.** Les seules mentions de « quetes »
en amont concernent le panneau de quetes comme **source de bruit visuel a masquer** — soit
exactement l'inverse de le lire.

A construire de zero :

- lecture du journal de quetes (OCR sur panneau, objectifs, etat) ;
- detection des PNJ sur la carte — distincte de celle des monstres, qui suppose un groupe
  engageable ;
- fenetre de dialogue : la reconnaitre, lire les repliques, choisir la bonne option ;
- machine a etats de quete : objectif courant, carte cible, condition d'achevement.

C'est le plus gros poste du projet et le moins entame. Ne pas le commencer avant que
G1–G3 tiennent : une quete est une SEQUENCE de deplacements, de combats et de dialogues,
et elle ne peut pas etre plus fiable que ses termes.

## Ce que Jev ne peut pas faire

L'espace d'actions vaut `1 + cases x (1 + sorts)`, soit **5 233** indices en configuration
reelle (218 cases, 23 sorts). Une question `choice` en accepte 255. L'ecart est un ordre
de grandeur -- et un tour n'est pas une action mais une **sequence**, que la primitive
n'exprime pas. Jev ne remplacera donc jamais le solveur : il remplace `scoring.py`,
14 ko de poids regles a la main pour departager des plans.
