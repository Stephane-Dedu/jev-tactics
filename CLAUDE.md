# Directives de travail

Perception par vision d'un combat au tour par tour, et **decision confiee a Jev**
(TypeSafe AI, modele « System One » a sortie typee). La perception, la legalite et
l'execution viennent d'un depot anterieur et fonctionnent ; ce qui est neuf ici, c'est
uniquement **qui decide**.

Cadre : recherche, evaluation en local / serveur prive. Non destine aux serveurs officiels.

---

## 1. Les invariants

Ils ne se negocient pas. Chacun a coute un defaut pour etre ecrit.

**Une seule interface de decision.** `decision.Decider` est le seul point ou l'on choisit
quoi jouer. `best_sequence` ne s'appelle **que** depuis `decision/search.py`. Un second
appelant, et l'on ne peut plus comparer deux decideurs -- or la comparaison est tout
l'objet du depot.

**Jev choisit, il ne formule jamais.** Le code enumere (`legal_actions`), elague
(`top_sequences`), et soumet un eventail. Jev rend un **identifiant**, borne-verifie par
`_index_of`. Un coup illegal n'est pas rejete apres coup : il n'est pas construit. C'est la
seule defense qui tienne -- Check Point a montre (septembre 2026) qu'une sortie typee
n'est pas une frontiere de securite et qu'un modele de decision se manipule par son entree.

**Rien de controle par un tiers n'entre dans `state`.** Les `entity_id` sont synthetiques
(`"me"`, `"enemy_42"`, `"e3"`), les noms de sorts viennent de la configuration de
l'operateur. Aucun pseudonyme, aucun message de joueur.
`projection.assert_untrusted_free` en fait une verification executable ; elle **leve**
plutot que de filtrer, parce qu'une perception qui se met a produire des noms venus du jeu
doit arreter la boucle, pas la faire continuer autrement.

**Un repli est bruyant.** Toute panne -- delai, quota, schema, confiance basse -- retombe
sur le solveur et s'inscrit dans `Decision.source`. Jamais sur la fin de tour.
Le mode de panne de ce projet est documente : *un bot qui n'attaque pas ressemble a un bot
prudent*. Un repli muet se lit exactement comme une decision deliberee.

**Aucun test ne sort sur le reseau.** `ReplayTransport` rejoue des cassettes ; la cle est
un hachage de la requete, donc un refactor de la projection invalide la cassette au lieu
de rejouer une reponse perimee. `typesafe-sdk` est un extra optionnel, la CI ne l'installe pas.

---

## 2. La mesure

**Toujours citer le regime avec un taux de victoire.** C'est la correction la plus
importante apportee au plan initial, qui repetait « 79 % » comme s'il s'agissait d'un
chiffre unique.

| Regime | Taux mesure (40 combats, graine 0, plateau 9x9, `example.json`) |
|---|---|
| PV ennemis **connus** | **80 %** |
| PV ennemis **inconnus** | **60 %** |

Sur 32 captures reelles : **72 ennemis detectes, zero avec des PV lisibles.** Ni l'OCR
(reserve au joueur) ni la timeline (un ratio sans maximum) ne les donnent. Le regime que
le bot joue reellement est donc celui a **60 %**, et tous les poids de `scoring.py` ont
ete regles dans l'autre.

> Le chiffre a battre est **60 %**, `enemy_hp_known=False`. Un gain mesure a PV connus ne
> prouve rien sur le jeu.

Deux decideurs se comparent **sur la meme graine** : sinon l'ecart mesure est celui des
scenarios. `benchmark(..., seed=0)` existe pour cela.

---

## 3. Qualite

**Le cliquet.** Le depot ne passe pas `mypy --strict` ni son propre jeu ruff -- il en
heritait 29 et 31 a l'import, et l'amont en compte 89. On ne bloque pas le travail
dessus, mais **le compte ne remonte jamais** : la CI compare a une base enregistree et
echoue sur toute erreur nouvelle. La base ne fait que decroitre.

Avant de committer :

```bash
ruff check .                      # zero erreur NOUVELLE vs docs/QUALITY_BASELINE.txt
mypy --strict src/                # idem
pytest -q                         # 667 passes, 217 sautes, 0 echec
```

**Le lint est un temoin, pas une formalite.** Le defaut le plus couteux trouve dans la
couche de decision -- chaque plan s'annoncant avec ses PA intacts alors qu'il les
depensait tous -- etait signale par `RUF021`, une regle deja activee dans `pyproject.toml`.
Personne n'avait lance l'outil.

---

## 4. Le style du depot

Les commentaires sont en francais et expliquent **pourquoi**, pas quoi. Ils portent des
**mesures** quand il y en a (`« 0/33 decisions tronquees a 60 000 noeuds »`,
`« 12 cases calculees contre 10 surlignees »`). Un commentaire qui paraphrase le code
n'a pas sa place ; un commentaire qui explique une contrainte non evidente, ou qui
consigne le defaut qu'une ligne empeche de revenir, en a.

Quand une hypothese est refutee par la mesure, **on retire aussi la contrainte qu'elle
justifiait**. Une borne survivant a son motif a deja fait tomber tout le pipeline ici.

---

## 5. Pieges connus

- **`tests/` insere `scripts/` dans `sys.path` lui-meme.** Un test peut donc dependre de
  `farming` ou `diagnose` sans qu'aucun `import` ne l'indique dans le fichier de test.
  Filtrer sur les tests seuls ne suffit pas : verifier aussi le script charge.
- **217 tests sautent sur un clone neuf** : ils demandent des captures non versionnees.
  La CI est verte en n'executant qu'un quart de moins que la suite locale. Tant que le
  corpus cure n'est pas commite (cf. `docs/PLAN.md`), ne pas lire « vert » comme « couvert ».
- **`configs/spells/example.json` est fictif et ne compte que 3 sorts.** Toute mesure de
  diversite ou de saturation faite dessus est trompeuse : `sacrieur.json` en porte 20, et
  c'est la que les plans candidats se revelent redondants.
- **Un `slot` errone retire un sort en silence** ; il ne leve rien. C'est le defaut le
  plus difficile a diagnostiquer du projet.
