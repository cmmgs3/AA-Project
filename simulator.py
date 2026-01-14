from world import Objective, Farol, Ambiente

class Simulador:
    def __init__(self, listaAgente: list, ambiente: Ambiente):
        self.listaAgentes = listaAgente
        self.ambiente = ambiente
        for agent in self.listaAgentes:
            self.ambiente.add_object(agent)

    def listarAgentes(self):
        for agent in self.listaAgentes:
            print(agent)

    def executa(self):
        objectives = []
        for obj in self.ambiente.objects.values():
            if isinstance(obj, Objective) or isinstance(obj, Farol):
                objectives.append(obj)
        self.ambiente.atualizacao()
        for agent in self.listaAgentes:
            agent.executar(self.ambiente)

        win = False
        for obj in objectives:
            if obj.checkWin(self.ambiente):
                win = True
        return win