Execucao:
1 - instalar requirments
2 - trainar agente com train.py (evolutivo) eou train_rl.py (reforco)
3 - selecionar pesos treinados no main.py. Por defeito tenta carregar pesos treinado com train.py - best_weights.json, para utilizar pesos treinados por train_rl.py, mudar  best_weights.json para  best_weights_rl.json em:
                      *Aqui*
    if os.path.exists("best_weights.json"):
        try:           *Aqui* 
            with open("best_weights.json", "r") as f:
                weights = json.load(f)
            for a in la:
                a.set_weights(weights)
            print("Loaded best weights!")
        except Exception as e:
            print(f"Could not load weights: {e}")

4 - selecionar senario de ENV_SCENARIOS, por defeito seleciona o segundo senario (ENV_SCENARIOS[1]).
5 - executar main.py
