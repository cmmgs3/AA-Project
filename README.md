Projeto de Agentes Autónomos\
Carlos Gonçalves - 112966\
Álvaro Arenas - 136707\
Nuno Zhu - 122684

Execução:\
1 - Instalar requirements (numpy e mathplotlib)\
2 - Treinar o agente com train.py (evolutivo) ou train_rl.py (reforço)\
3 - Selecionar os pesos treinados no main.py. Por defeito tenta carregar pesos treinados com train.py - "best_weights.json"\
Para utilizar pesos treinados por train_rl.py, mudar "best_weights.json" para "best_weights_rl.json" em:
***
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
***
4 - Selecionar cenário de ENV_SCENARIOS, por defeito vem selecionado o segundo cenario ( ENV_SCENARIOS[1] )\
5 - Executar main.py
