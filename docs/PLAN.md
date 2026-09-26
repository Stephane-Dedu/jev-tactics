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

## P1 — Des postures, pas des variantes  *(fait)*

`planner/postures.py` : un plan par INTENTION, chacun etant l'optimum du solveur sous une
ponderation differente de `evaluate`. `scoring.Weights` rend la ponderation parametrable ;
son defaut reproduit l'ancien comportement au bit pres, donc la reference reste comparable.

Postures : `reference` (l'heuristique, et le repli), `degats`, `achever`, `abri`,
`position`.

**Une seule exploration.** `Candidate` porte desormais son `TurnState` final, donc les
plans sont RENOTES sous chaque ponderation au lieu d'etre recherches cinq fois. Cout
mesure : **+10 %** sur le temps de decision (2919 ms contre 2652 ms pour `best_sequence`
seul, a 20 sorts / 10 PA / plateau 9x9).

### Le defaut est corrige, et mesure

| | avant | apres |
|---|---|---|
| descriptions factuelles identiques | **16 / 24** | **0 / 99** |
| plage de degats entre options | 93–93 | reelle |

### Le critere « >= 3 postures » etait mal pose

Je l'avais ecrit sans nommer le REGIME -- exactement l'erreur relevee plus haut a propos
des « 79 % ». Le nombre d'intentions distinctes depend de la puissance du personnage :

| regime | 1 option | 2 | 3 | 4 | mediane |
|---|---|---|---|---|---|
| 2 ennemis, 200 PV, 10 PA | 17/40 | 21 | 1 | 1 | 2 |
| 4 ennemis, 300 PV, 10 PA | 10/40 | 20 | 9 | 1 | 2 |
| **2 ennemis, 60 PV, 6 PA** | **0/40** | 12 | 22 | 6 | **3** |

Un personnage surpuissant n'a pas de dilemme : le meilleur plan l'est sous toutes les
intentions. Le dilemme apparait quand les ressources manquent -- c'est-a-dire quand la
decision compte. Une seule option n'est donc pas un echec : `distinct_enough` fait sauter
l'appel, et c'est l'information la plus honnete que la position puisse donner.

### Le choix change le resultat

120 combats, memes graines, PV ennemis inconnus, regime contraint (2 ennemis, 60 PV, 6 PA) :

| politique | victoires |
|---|---|
| toujours `degats` | **50,0 % ± 8,9** |
| toujours `reference` | 40,8 % ± 8,8 |
| regle « abri si PV < 35 % » | 25,8 % ± 7,8 |
| toujours `abri` | 24,2 % ± 7,7 |

Trois enseignements, et deux changent la suite du travail :

1. **Vingt-six points separent la meilleure intention de la pire.** Il y a donc bien
   quelque chose a decider -- c'etait la question ouverte que P1 devait trancher.
2. **La barre pour Jev est 50 %, pas 40,8 %.** Battre la reference ne suffit plus : il
   faut battre « toujours frapper fort ».
3. **Une regle simple echoue** (25,8 %, a peine mieux que fuir toujours). Choisir MAL est
   pire que ne pas choisir. Ce n'est pas un argument pour Jev, c'est un avertissement :
   un mauvais selecteur coute plus qu'il ne rapporte.

### Ce que cela revele sur `scoring.py`

`degats` -- qui neutralise prudence, prime de mise a mort et penalites -- bat de neuf
points la ponderation reglee a la main. Les intervalles se chevauchent, donc ce n'est pas
conclu ; mais la direction est la meme aux deux tailles d'echantillon testees (40 et 120),
et elle est coherente avec ce que l'arene documente deja : **tous les poids ont ete regles
a PV ennemis CONNUS**, regime que le bot ne rencontre jamais. A mesurer plus largement
avant d'y toucher.

## P2 — Le banc de comparaison  *(fait)*

`scripts/compare_deciders.py` : plusieurs politiques, les MEMES graines, et **les deux
regimes de PV cote a cote**.

```bash
python scripts/compare_deciders.py --fights 60 --spells configs/spells/sacrieur.json     --regime contraint --postures
```

### Le regime inverse le classement

40 combats, `sacrieur.json`, regime contraint :

| politique | PV connus (l'arene) | PV inconnus (le jeu) | ecart |
|---|---|---|---|
| toujours `reference` | **62,5 %** | 37,5 % | **−25,0** |
| toujours `position` | 62,5 % | 37,5 % | −25,0 |
| toujours `achever` | 60,0 % | 37,5 % | −22,5 |
| solveur seul | 55,0 % | 30,0 % | −25,0 |
| toujours `degats` | 45,0 % | **42,5 %** | **−2,5** |
| toujours `abri` | 0,0 % | 22,5 % | +22,5 |

A 40 combats l'intervalle vaut ±15 points : aucun ecart ENTRE politiques n'est concluant.
Ce qui l'est davantage, c'est la comparaison D'UNE politique a elle-meme d'un regime a
l'autre -- meme code, memes graines, une seule variable.

Et la, le resultat est net : **la ponderation reglee a la main perd 25 points en passant
au regime reel, la ponderation naive en perd 2,5.** `reference` est la meilleure dans
l'arene et mediocre dans le jeu ; `degats` est la pire dans l'arene et la meilleure dans
le jeu.

C'est la confirmation la plus directe de ce que l'arene documentait deja sans en tirer la
consequence : les poids de `scoring.py` sont optimaux pour un regime que le bot ne
rencontre jamais. Les retuner a PV inconnus est desormais un chantier identifie -- mais il
demande plus de combats que 40, l'intervalle actuel etant trop large pour regler quoi que
ce soit.

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

### G1 — Se deplacer  *(hors-ligne : fait ; reste la validation en jeu)*

| Brique | Etat |
|---|---|
| Graphe de cartes, trajets, circuits (`world/navigation.py`) | **porte, 93 tests verts** |
| Ou cliquer pour sortir (`world/travel.edge_point`) | **extrait, teste** |
| Juger le deplacement (`world/travel.judge_move`) | **extrait, teste** |
| Lecture des coordonnees (`perception/coordinates`) | deja la |
| Boucle : cliquer, attendre, confirmer, rembobiner (`world/walker.py`) | **ecrit, 16 tests** |
| Trajet vers une carte cible (`Walker.travel_to`) | **ecrit, teste** |

`navigation.py` vivait sous `farming/` sans rien en importer : le ranger la rendait le
deplacement inaccessible a tout ce qui n'etait pas de la recolte. Il est maintenant sous
`world/`.

L'extraction de `travel` separe ce qui etait mele dans `farming/runner.py` : **decider ou
cliquer** et **juger ce qui s'est passe** d'un cote, la comptabilite de la recolte de
l'autre. `judge_move` rend trois issues — arrive / derive / immobile — et non un booleen,
parce que la **derive** est le mode de panne qui ne se voit pas : cliquer un bord fait
marcher le personnage, un obstacle le fait sortir ailleurs, la comparaison d'images dit
« reussi », et le circuit repart d'une carte decalee qu'il ne rattrapera jamais.

`Walker` scrute au lieu d'attendre. Un delai FIXE est le pire mode de panne du circuit :
il couvre deux choses de duree tres variable -- la MARCHE jusqu'au bord, puis le
chargement -- et quand il ne suffit pas, la verification tombe sur l'ancienne carte,
conclut « sans effet », rembobine la route, et la carte charge quand meme. Le circuit
reste decale d'un cran, definitivement. Le signal d'arret ne coute rien : ce sont les
coordonnees, qu'il faut lire de toute facon pour juger le pas.

Verifie par MUTATION : remplacer l'attente adaptative par un retour immediat tue
`test_a_slow_map_is_still_a_success`, et lui seul.

`travel_to` recalcule le trajet a chaque pas plutot que de le suivre en aveugle -- c'est
ce qui rend une derive rattrapable, la sortie trompeuse venant d'etre enregistree dans le
graphe. C'est aussi la seule reponse a un SAUT de plusieurs cartes (zaap, rappel).

Reste pour clore G1 : brancher sur le vrai `grab`/backend et valider en jeu.

### G2 — Engager un combat  *(hors-ligne : fait ; reste la validation en jeu)*

| Brique | Etat |
|---|---|
| Detection de groupes (`perception/monsters.py`, 72 ko) | deja la |
| Cliquer et confirmer (`world/engage.Engager`) | **ecrit, 16 tests** |
| Verdict differe (`Engager.settle`) | **ecrit, teste** |
| Liste noire plafonnee (`world/engage.FailedSpots`) | **ecrit, teste** |
| Choisir QUEL groupe engager | ⬜ candidat `noul` pour Jev |

La verification n'est pas un confort : la detection par mouvement attrape aussi **les
autres joueurs**, que rien de visuel ne distingue d'un groupe. Sans controle, le bot
clique un joueur qui passe, indefiniment, en croyant engager. La timeline tranche — elle
n'existe qu'en combat.

Le defaut garde ici n'est pas « le clic rate », c'est le **verdict trop tot**. Un
depassement d'echeance rend `PENDING`, jamais `MISSED` : un groupe lointain demande une
longue marche plus l'ecran de placement, et le combat peut demarrer juste apres. Conclure
tout de suite ecrivait un faux negatif au journal *et* condamnait l'endroit pour toute la
session — sur les groupes LOINTAINS et GROS, soit exactement ceux qu'il fallait apprendre
a preferer. Au cycle suivant, `settle(in_combat=True)` rend `LATE` et non `STARTED` : un
groupe qui passe peut agresser, rien ne prouve que le combat vienne du clic.

Verifie par MUTATION : conclure a l'echeance tue les trois tests du verdict differe.

`EngageResult.waited` rend le delai reel, qui etait jete. `ENGAGE_TIMEOUT` vaut 8 s par
ANALOGIE, jamais par mesure ; ce chiffre est ce qui permettra de le poser sur des donnees.

Reste : **choisir** quel groupe engager. C'est la premiere question `noul` naturelle pour
Jev (« ce groupe vaut-il d'etre engage ? »), avec `monsters.best_group` comme reference.

### G3 — Gagner le combat  *(bloque par D1)*

C'est P1 ci-dessus : les plans candidats ne sont pas distincts, donc Jev n'a rien a
trancher. Rien d'autre ne bloque — la boucle percevoir -> decider -> agir tourne.

### G4 — Quetes et PNJ  *(logique faite ; perception bloquee sur des captures)*

**Rien n'existait**, dans aucun des deux depots. Les seules mentions de « quetes » en
amont visent a MASQUER le panneau comme source de bruit visuel.

| Brique | Etat |
|---|---|
| Modele : quete, etapes, avancement (`quest/model.py`) | **ecrit, teste** |
| Directeur : quelle action ensuite (`quest/director.py`) | **ecrit, 25 tests** |
| Deroule bout en bout (`plan_ahead`) | **ecrit, teste** |
| Executeur : jouer la quete, gerer les pannes (`quest/runner.py`) | **ecrit, 11 tests** |
| Boucle de combat vivante (`bot/fight.py`) | **ecrit, 12 tests** |
| Gestionnaires reels (`quest/handlers.py`) | **ecrit, 11 tests** |
| Lecture du journal de quetes | ⬜ **bloque : aucune capture** |
| Detection des PNJ sur la carte | ⬜ **bloque : aucune capture** |
| Fenetre de dialogue, choix de replique | ⬜ **bloque : aucune capture** |

`director.next_intent` est une fonction PURE : une quete entiere se joue en quelques
millisecondes, sans le jeu, et une erreur d'ordre se voit dans un test au lieu de se
decouvrir apres quarante minutes de session. C'est aussi la couche qui relie enfin les
trois briques faites -- une quete n'est qu'une SEQUENCE de deplacements, de combats et de
dialogues.

Deroule obtenu sur une quete a trois etapes, depart (0,0) :

```
 1. travel en (5, 5)
 2. talk Aventurier en (5, 5)  [non executable : perception manquante]
 3. travel en (6, 4)
 4. step_done en (6, 4)
 5. travel en (7, 3)
 6. fight Bouftou x3 en (7, 3)
 7. harvest Ble x5 en (7, 3)
 8. travel en (2, 8)
 9. talk Ancien en (2, 8)  [non executable : perception manquante]
10. done
```

**Le manque est declare dans le CODE, pas ici.** `Intent.implemented` vaut faux pour tout
ce qui demande la perception des PNJ, et `test_the_executable_set_matches_the_layers_that_exist`
echouera le jour ou cette perception arrivera sans que le directeur soit mis a jour --
exactement le moment ou il faut y penser.

Un defaut d'API corrige au passage : `Act.DONE` signifiait a la fois « cette etape est
satisfaite, continuer » et « la quete est finie, s'arreter ». Un appelant ecrivant
`if intent.act is Act.DONE: stop()` arretait la quete au milieu, sur un simple
deplacement accompli. Scinde en `STEP_DONE` / `DONE`.

#### L'executeur, et le PIETINEMENT

`quest/runner.py` joue les intentions. La partie difficile n'est pas d'enchainer des
succes -- un bot ne passe pas sa vie a reussir -- mais de reconnaitre **une intention qui
reussit sans rien faire avancer** : on se rend sur la bonne carte, on y tue le mauvais
monstre. Rien n'« echoue », donc un compteur d'echecs laisse la boucle tourner en la
declarant saine. C'est la meme forme que « 30 tours joues » pour un combat ou le bot
n'avait lance aucun sort. Le compteur porte donc sur l'AVANCEMENT, pas sur le succes.

Un defaut trouve par un test, pas par relecture : `skip()` fait avancer l'index
d'OBJECTIF, ce qui ne veut rien dire tant que la quete n'est pas acceptee. Un deplacement
impossible vers le donneur faisait sauter des objectifs qu'on n'avait pas le droit de
commencer, et le directeur redemandait le meme trajet a chaque tour. La reponse depend
desormais de la PHASE : avant l'acceptation et au rendu rien n'est sautable -- la quete
est abandonnee en le disant -- et seuls les objectifs peuvent etre passes.
Verifie par mutation.

#### La boucle de combat, et les tours MUETS

`bot/fight.py` relie enfin percevoir / decider / executer face au client. La boucle est
simple ; savoir si elle a REELLEMENT joue ne l'est pas.

Un plan reduit a « fin de tour » s'execute **sans erreur** et marque le tour comme joue.
« 30 tours joues » se lisait donc exactement comme un combat mene au corps a corps, alors
que le bot n'avait lance aucun sort -- et la cause la plus frequente n'est pas tactique :
un `slot` errone retire le sort en silence. `FightReport` compte donc les tours muets a
part, remonte les sorts sans raccourci, et affiche une PROPORTION : un tour muet isole est
normal, la moitie est un avertissement, la totalite nomme la barre de sorts.
Verifie par mutation -- retirer le compte tue trois tests.

Une frame illisible ne termine pas le combat : une frame de transition l'est sans que le
combat soit fini, et la declarer terminee renvoyait le bot sur son circuit en pleine
bagarre. Trois lectures vides d'affilee valent une fin, une seule vaut un hoquet. Zero
tour joue est rapporte comme une panne de perception (`BLIND`) et non comme une victoire
rapide.

`quest/handlers.py` branche le directeur sur `Walker` et `Engager`. Deux refus y sont
gardes par des tests, parce que tous deux feraient progresser une quete sur une panne :
un deplacement reussi n'avance aucun objectif, et **un combat entierement muet n'est pas
un combat gagne**, quoi qu'en dise la timeline.

Le comptage des morts est volontairement SOUS-ESTIME (un par combat) : rien ne sait
combien d'ennemis un groupe contenait ni lesquels sont morts, les PV ennemis n'etant
jamais lisibles. Sous-estimer refait un combat de trop ; surestimer envoie rendre une
quete qui ne l'est pas.

#### Ce qu'il faut pour debloquer la suite

Des captures 1920x1080 du client, hors combat :

1. **le journal de quetes ouvert**, avec au moins une quete en cours et ses objectifs ;
2. **une carte portant un PNJ**, et la meme carte sans lui si possible -- c'est l'ecart
   qui apprend le detecteur ;
3. **une fenetre de dialogue ouverte**, avec plusieurs repliques proposees ;
4. la meme fenetre a une autre etape, pour voir ce qui bouge.

Sans elles, la perception des PNJ ne peut pas etre ecrite -- ni, surtout, verifiee. Le
depot a douze defauts trouves par confrontation a de vraies captures, chacun invisible en
test : en inventer serait en fabriquer un treizieme.

## Ce que Jev ne peut pas faire

L'espace d'actions vaut `1 + cases x (1 + sorts)`, soit **5 233** indices en configuration
reelle (218 cases, 23 sorts). Une question `choice` en accepte 255. L'ecart est un ordre
de grandeur -- et un tour n'est pas une action mais une **sequence**, que la primitive
n'exprime pas. Jev ne remplacera donc jamais le solveur : il remplace `scoring.py`,
14 ko de poids regles a la main pour departager des plans.
