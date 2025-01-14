

<div style="display:flex;flex-direction:row">
    <img src="https://raw.githubusercontent.com/gabriel-ferreira-da-silva/thoth/f0c7f066c924120c134915c4589f2b7f756d78ef/docs/thoth.png" alt="Description" style="margin: 20px; width:140px; border-radius: 10px ;height: 70px; box-shadow: 5px 5px 10px \#888;">
    <img src="https://raw.githubusercontent.com/gabriel-ferreira-da-silva/thoth/f0c7f066c924120c134915c4589f2b7f756d78ef/docs/logo.png" alt="Description" style="margin: 20px; width:70px; border-radius: 10px ;height: 70px; box-shadow: 5px 5px 10px \#888;">
</div>

### __implementation of ml algorithms__

Thoth is module for the Implementation of artificial neural networks algorithms for the Introduction to Deep Learning Class.

The classes of ANN and MLP are imported from the directory. Some codes to train over datasets like iris, breast cancer or wine quality are in examples folder and can be executed as python module as bellow: 

```
cd thoth
python3 -m examples.mlp-breastcancer
```

these examples serves as guides to import MLP's and other ANN

![image-20250114100939466](https://github.com/gabriel-ferreira-da-silva/thoth/blob/test-cases/docs/Screenshot%20from%202025-01-14%2010-08-00.png?raw=true)

and how to set atributes of the MLP

![image-20250114101100555](https://github.com/gabriel-ferreira-da-silva/thoth/blob/test-cases/docs/Screenshot%20from%202025-01-14%2010-10-54.png?raw=true)

# IF867 - Introdução à Aprendizagem Profunda

### 1ª atividade prática

Discente(s): Gabriel Ferreira, Williams Andrade

Período:

## Instruções e Requisitos
- Objetivo: Implementar e treinar um Multilayer Perceptron (MLP), inteiramente em [NumPy](https://numpy.org/doc/stable/) ou [Numba](https://numba.readthedocs.io/en/stable/index.html), sem o uso de bibliotecas de aprendizado profundo.
- A atividade pode ser feita em dupla.

### Tarefas

### Implementação (50%):

- [x] Implemente pelo menos duas funções de ativação diferentes para as camadas ocultas; use Sigmoid e Linear para a camada de saída.

- [x] Implemente forward e backpropagation.

- [x] Implemente um otimizador de sua escolha, adequado ao problema abordado.

- [x] Implemente as funções de treinamento e avaliação.

__Aplicação (30%):__

  Teste se os seus modelos estão funcionando bem com as seguintes tarefas:
- [x] Regressão

- [x] Classificação binária

__Experimentação (20%):__

  Teste os seus modelos com variações na arquitetura, no pré-processamento, etc. Escolha pelo menos uma das seguintes opções:
- [x] Variações na inicialização de pesos

- [x] Variações na arquitetura

- [x] Implementação de técnicas de regularização

- [x] Visualização das ativações e gradientes

***Bônus:*** Implemente o MLP utilizando uma biblioteca de machine learning (ex.: [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/?hl=pt-br), [tinygrad](https://docs.tinygrad.org/), [Jax](https://jax.readthedocs.io/en/latest/quickstart.html)) e teste-o em uma das aplicações e em um dos experimentos propostos. O bônus pode substituir um dos desafios de aplicação ou experimentos feitos em NumPy, ou simplesmente somar pontos para a pontuação geral.

### Datasets recomendados:
Aqui estão alguns datasets recomendados, mas fica a cargo do aluno escolher os datasets que utilizará na atividade, podendo escolher um dataset não listado abaixo.
- Classificação

  - [X] [Iris](https://archive.ics.uci.edu/dataset/53/iris)

  - [X] [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

  - [ ] [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

- Regressão

  - [X] [Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality)
  
  - [ ] [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
  
  - [X] [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

### Requisitos para Entrega

Um notebook Jupyter (de preferência, o link do colab) ou script Python contendo:

- Código: Implementação completa da MLP.
- Gráficos e Análises: Gráficos da curva de perda, ativações, gradientes e insights do treinamento, resultantes dos experimentos com parada antecipada e diferentes técnicas de regularização.
- Relatório: Um breve relatório detalhando o impacto de várias configurações de hiperparâmetros(ex.: inicialização de pesos, número de camadas ocultas e neurônios) e métodos de regularização no desempenho do modelo.



