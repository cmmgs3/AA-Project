from world import Objective, Farol, Ambiente

class Simulador:
    def __init__(self, listaAgente: list, ambiente: Ambiente):
        self.ambiente = ambiente
        self.listaAgentes = listaAgente
        for agent in self.listaAgentes:
            self.ambiente.add_object(agent)
        self.objectives = []
        for obj in ambiente.objects.values():
            if isinstance(obj, Objective) or isinstance(obj, Farol):
                self.objectives.append(obj)

    def listarAgentes(self):
        for agent in self.listaAgentes:
            print(agent)

    def executa(self):
        self.ambiente.atualizacao()
        for agent in self.listaAgentes:
            agent.executar(self.ambiente)
        win = False
        for obj in self.objectives:
            if obj.checkWin(self.ambiente):
                win = True
        return win