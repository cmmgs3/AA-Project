import random
import time
import os

class Agente:
    def __init__(self, x, y, designacao):
        self.x = x
        self.y = y
        self.designacao = designacao

    def mover(self, tamanho):
        # Movimenta aleatoriamente para cima, baixo, esquerda ou direita,
        # mas sem sair do mundo
        direcoes = ['cima', 'baixo', 'esquerda', 'direita']
        movimento = random.choice(direcoes)

        if movimento == 'cima' and self.x > 0:
            self.x -= 1
        elif movimento == 'baixo' and self.x < tamanho - 1:
            self.x += 1
        elif movimento == 'esquerda' and self.y > 0:
            self.y -= 1
        elif movimento == 'direita' and self.y < tamanho - 1:
            self.y += 1

class Mundo:
    def __init__(self, tamanho):
        self.tamanho = tamanho
        self.agentes = []

    def adicionar_agente(self, agente):
        self.agentes.append(agente)

    def mover_agentes(self):
        for agente in self.agentes:
            agente.mover(self.tamanho)

    def representar(self):
        # Cria uma matriz vazia
        matriz = [['.' for _ in range(self.tamanho)] for _ in range(self.tamanho)]

        # Marca os agentes na matriz pela primeira letra da designação
        for agente in self.agentes:
            matriz[agente.x][agente.y] = agente.designacao[0].upper()

        # Imprime a matriz
        for linha in matriz:
            print(' '.join(linha))

if __name__ == '__main__':
    mundo = Mundo(10)
    mundo.adicionar_agente(Agente(0, 0, 'Alice'))
    mundo.adicionar_agente(Agente(9, 9, 'Bob'))

    for passo in range(10):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Passo {passo + 1}")
        mundo.representar()
        mundo.mover_agentes()
        time.sleep(1)
