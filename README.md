# artificial-intelligence

Dois projetos acadêmicos de inteligência artificial implementados em Java sem bibliotecas de aprendizado de máquina:

- **`neural_networks/`** — perceptron simples, perceptron multicamadas e uma rede neural genérica com retropropagação.
- **`genetic_algorithms/`** — motor de algoritmo genético aplicado a N-Rainhas e a três funções clássicas de otimização contínua.
---

## Sumário

**Parte I — Redes neurais artificiais**

1. [O neurônio como unidade de cálculo](#1-o-neurônio-como-unidade-de-cálculo)
2. [O perceptron de Rosenblatt](#2-o-perceptron-de-rosenblatt)
3. [O limite: separabilidade linear](#3-o-limite-separabilidade-linear)
4. [1969 e o inverno](#4-1969-e-o-inverno)
5. [Por que mais de uma camada resolve](#5-por-que-mais-de-uma-camada-resolve)
6. [O problema da atribuição de crédito e a retropropagação](#6-o-problema-da-atribuição-de-crédito-e-a-retropropagação)
7. [Aproximação universal — e o que o teorema não diz](#7-aproximação-universal--e-o-que-o-teorema-não-diz)
8. [O que quebrou depois, e como foi consertado](#8-o-que-quebrou-depois-e-como-foi-consertado)

**Parte II — O projeto de redes neurais**

9. [As três implementações](#9-as-três-implementações)
10. [Estrutura do código](#10-estrutura-do-código)
11. [Verificação numérica do gradiente](#11-verificação-numérica-do-gradiente)
12. [Validação e parada antecipada](#12-validação-e-parada-antecipada)
13. [Reprodutibilidade](#13-reprodutibilidade)
14. [Formato das bases](#14-formato-das-bases)

**Parte III — Algoritmos genéticos**

15. [A ideia da computação evolutiva](#15-a-ideia-da-computação-evolutiva)
16. [O motor e os problemas](#16-o-motor-e-os-problemas)

**Parte IV — Operação**

17. [Como compilar e executar](#17-como-compilar-e-executar)
18. [Limitações conhecidas e preservadas](#18-limitações-conhecidas-e-preservadas)

---

# Parte I — Redes neurais artificiais

## 1. O neurônio como unidade de cálculo

Em 1943, o neurofisiologista **Warren McCulloch** e o lógico **Walter Pitts** publicaram *A Logical Calculus of the Ideas Immanent in Nervous Activity*. A proposta era desconcertantemente simples: tratar o neurônio como um dispositivo binário. Ele recebe sinais de entrada, soma-os, e dispara se a soma ultrapassa um limiar.

```
        x1 ──w1──┐
        x2 ──w2──┼──► [ Σ ] ──► [ limiar ] ──► saída (0 ou 1)
        x3 ──w3──┘
```

O artigo mostrou que redes desses neurônios podiam computar qualquer função lógica — AND, OR, NOT e, por composição, qualquer coisa que uma máquina de estados finitos computa. Era um resultado sobre *poder de representação*.

Faltava tudo o mais. Os pesos eram fixados à mão. Não havia mecanismo de aprendizado: o modelo dizia o que uma rede *podia* representar, não como ela chegaria lá sozinha.

A peça que faltava apareceu em 1949, com **Donald Hebb** e *The Organization of Behavior*. A hipótese de Hebb — que se tornou o slogan "neurônios que disparam juntos se conectam" — propunha que a força de uma conexão aumenta quando os dois neurônios que ela liga se ativam ao mesmo tempo. Pela primeira vez, uma regra de mudança dos pesos derivada de experiência, e não de projeto.

## 2. O perceptron de Rosenblatt

Em 1958, **Frank Rosenblatt**, psicólogo do Cornell Aeronautical Laboratory, publicou *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*. Ele juntou o neurônio de McCulloch-Pitts com uma regra de aprendizado que fazia algo que a de Hebb não fazia: **corrigir erros**.

A regra é quase trivial:

```
para cada amostra:
    saída = limiar( Σ wi·xi + viés )
    erro  = alvo − saída
    wi   ← wi + η · erro · xi
```

Se a rede acerta, o erro é zero e nada muda. Se erra, cada peso é empurrado na direção que teria produzido a resposta certa, proporcionalmente à entrada que contribuiu para o engano. O `η` — a taxa de aprendizado — controla o tamanho do empurrão.

E Rosenblatt provou algo forte sobre ela, o **teorema da convergência do perceptron**: se existe um conjunto de pesos que classifica corretamente todos os exemplos, a regra o encontra em um número finito de passos. Não "provavelmente"; não "aproximadamente". Encontra.

O perceptron não era só teoria. O **Mark I Perceptron**, construído em Cornell entre 1958 e 1960, era uma máquina física: um arranjo de 400 fotocélulas fazia as vezes de retina, os pesos eram potenciômetros, e motores elétricos giravam os potenciômetros para aplicar a regra de aprendizado. Ela aprendia a reconhecer formas simples ajustando resistências.

O entusiasmo foi proporcional. O *New York Times* noticiou, em 1958, que a Marinha americana esperava do perceptron "o embrião de um computador eletrônico que será capaz de andar, falar, ver, escrever, reproduzir-se e ter consciência de sua própria existência".

No mesmo período, **Bernard Widrow** e **Ted Hoff** desenvolveram em Stanford o ADALINE e a regra delta (ou LMS, *least mean squares*), que substituía o degrau por uma saída linear e minimizava o erro quadrático. É a mesma ideia de descida do gradiente que sustenta o treinamento moderno, e a regra delta continua sendo a base do que este projeto implementa.

## 3. O limite: separabilidade linear

O perceptron traça uma reta. Em mais dimensões, um hiperplano. Ele classifica pelo lado em que o ponto cai:

```
Σ wi·xi + b  >  0   →  classe A
Σ wi·xi + b  ≤  0   →  classe B
```

Isso resolve qualquer problema **linearmente separável**. O AND é separável — há retas que isolam o ponto (1,1) dos outros três. O OR também. O NAND também.

O **XOR** não é.

```
   x2
    │
  1 │  ●(0,1)=1        ○(1,1)=0
    │
  0 │  ○(0,0)=0        ●(1,0)=1
    └─────────────────────────── x1
       0                1
```

As duas saídas 1 estão em cantos opostos, e as duas saídas 0 nos outros dois cantos opostos. Nenhuma reta separa `{(0,1),(1,0)}` de `{(0,0),(1,1)}`. Não é uma questão de treinar mais, nem de escolher melhor o `η`: **a solução não existe no espaço de funções que um perceptron consegue representar**. O teorema da convergência continua verdadeiro — sua hipótese é que a solução exista, e aqui ela não existe.

E o XOR não é um caso de laboratório. É a estrutura de "um ou outro, mas não os dois", que aparece em qualquer problema real com interação entre atributos.

## 4. 1969 e o inverno

Em 1969, **Marvin Minsky** e **Seymour Papert** publicaram *Perceptrons: An Introduction to Computational Geometry*. O livro é uma análise matemática rigorosa — e devastadora — do que perceptrons de camada única podem e não podem fazer. O XOR é o exemplo mais citado; os autores foram além, mostrando limitações para paridade e para a detecção de conectividade em imagens.

Minsky e Papert conheciam a saída: acrescentar camadas intermediárias. O ceticismo deles não era sobre o poder de representação de redes multicamadas, e sim sobre a possibilidade de **treiná-las** — não se conhecia regra de aprendizado que funcionasse além da primeira camada.

O efeito sobre o campo foi profundo. O financiamento para pesquisa em redes neurais praticamente desapareceu, e a área entrou em um hiato de cerca de quinze anos que ficou conhecido como o primeiro inverno das redes neurais. Rosenblatt morreu em 1971, aos 43 anos, num acidente de barco, sem ver a reabilitação de sua ideia.

## 5. Por que mais de uma camada resolve

A saída é acrescentar uma **camada oculta** — neurônios que não são nem entrada nem saída, e cuja função é construir uma representação intermediária do problema.

Volte ao XOR. Ele tem uma decomposição clássica:

```
XOR(a, b) = AND( OR(a, b), NAND(a, b) )
```

Tanto o OR quanto o NAND são linearmente separáveis — cada um deles cabe num perceptron. E o AND, que combina os dois, também é. O que é impossível numa camada torna-se trivial em duas:

```
         ┌──────────┐
   a ────┤          ├── h1 = OR(a,b)   ──┐
         │  camada  │                    ├── AND ──► XOR(a,b)
   b ────┤  oculta  ├── h2 = NAND(a,b) ──┘
         └──────────┘
```

O que a camada oculta faz, geometricamente, é **redesenhar o espaço**. Os quatro pontos do XOR são reposicionados no espaço `(h1, h2)` de um jeito em que uma reta passa a separá-los. A camada de saída continua fazendo a única coisa que sabe fazer — traçar um hiperplano — mas agora num espaço onde isso basta.

Essa é a ideia geral, e ela não é sobre o XOR. Cada camada oculta aplica uma transformação não-linear; a composição delas constrói representações progressivamente mais úteis, até que o problema vire linearmente separável na última.

Repare na palavra **não-linear**. É indispensável. Se as camadas fossem lineares, a composição de duas seria uma transformação linear — e uma rede de dez camadas lineares representaria exatamente o mesmo que um único neurônio. É a função de ativação que impede o colapso, e é por isso que ela não é um detalhe de implementação: é o que torna a profundidade significativa.

## 6. O problema da atribuição de crédito e a retropropagação

Acrescentar a camada resolve a representação e cria um problema novo.

A regra do perceptron precisa do erro do neurônio: `erro = alvo − saída`. Para a camada de saída, isso é conhecido — o alvo está no conjunto de dados. Para um neurônio oculto, **não existe alvo**. Ninguém sabe qual deveria ter sido a saída correta de `h1`. Então qual é o erro dele? E de quanto os pesos que chegam nele devem mudar?

Isso é o **problema da atribuição de crédito**: distribuir a responsabilidade por um erro observado na saída entre parâmetros que não estão na saída. Foi exatamente esse impasse que fundamentou o ceticismo de 1969.

A solução é a regra da cadeia do cálculo. Se o erro `E` é uma função diferenciável da saída, e a saída é uma função diferenciável dos pesos, então `∂E/∂w` existe e pode ser calculada — camada por camada, de trás para a frente:

```
    propagação para frente  ──────────────────────────────►
    entrada → [camada 1] → [camada 2] → [camada 3] → saída → E
    ◄────────────────────────────────────  retropropagação
```

Para cada neurônio se calcula um **delta**: o erro que chega do lado da saída, multiplicado pela derivada da própria ativação. O delta multiplicado pela entrada dá o gradiente do peso; o delta multiplicado pelo peso dá o erro que a camada anterior deve receber. Um neurônio oculto recebe, portanto, a soma dos erros dos neurônios que ele alimenta, ponderada pela força com que os alimenta. Quem mais contribuiu para o erro, mais responsabilidade recebe.

Isso exige que a ativação seja **diferenciável** — e o degrau de Rosenblatt não é. Daí a sigmoide: `f(u) = 1/(1+e^(−u))`, uma versão suave do degrau, com derivada em forma fechada e barata, `f'(u) = f(u)·(1−f(u))`.

A história do algoritmo é a de uma redescoberta múltipla. **Paul Werbos** o descreveu em sua tese de doutorado em Harvard, em 1974, e o trabalho passou quase despercebido. **David Parker** e **Yann LeCun** chegaram a formulações equivalentes por conta própria em meados dos anos 1980. Mas o marco que reacendeu o campo é de 1986: **David Rumelhart, Geoffrey Hinton e Ronald Williams** publicaram na *Nature* o artigo *Learning representations by back-propagating errors*, que apresentou o método de forma clara, com experimentos convincentes, e mostrou que as camadas ocultas aprendiam representações interpretáveis do problema.

A objeção de Minsky e Papert tinha resposta. As redes multicamadas eram treináveis.

## 7. Aproximação universal — e o que o teorema não diz

Em 1989, **George Cybenko** provou o **teorema da aproximação universal**: uma rede com uma única camada oculta, com ativações sigmoides e neurônios suficientes, aproxima qualquer função contínua num domínio limitado, com a precisão que se quiser. **Kurt Hornik, Maxwell Stinchcombe e Halbert White** provaram no mesmo ano uma versão mais geral, mostrando que a propriedade vem da arquitetura, não da sigmoide em particular.

O resultado é frequentemente citado como a justificativa teórica das redes neurais. Vale entender exatamente o que ele afirma — e o que não afirma.

Ele afirma que **existe** um conjunto de pesos que aproxima a função. Ele não diz:

- **quantos neurônios** são necessários — pode ser um número absurdamente grande;
- **como encontrar** esses pesos — a existência não é um algoritmo, e a descida do gradiente encontra mínimos locais;
- que a rede vai **generalizar** — aproximar bem os dados de treino e acertar dados novos são coisas diferentes.

Na prática, é justamente por isso que redes profundas venceram: uma camada oculta larguíssima é teoricamente suficiente e praticamente inviável, enquanto várias camadas estreitas representam a mesma função com muito menos parâmetros.

## 8. O que quebrou depois, e como foi consertado

Com a retropropagação funcionando, o campo esbarrou no limite seguinte: redes com muitas camadas continuavam difíceis de treinar.

**O gradiente que desaparece.** A derivada da sigmoide vale no máximo 0,25 e tende a zero nas saturações. Como a retropropagação multiplica derivadas ao atravessar cada camada, o gradiente encolhe pelo menos quatro vezes por camada. Após cinco ou seis, o que chega às primeiras camadas é numericamente indistinguível de zero, e elas simplesmente não aprendem. **Sepp Hochreiter** analisou o fenômeno em detalhe em sua tese de 1991.

Três correções mudaram o quadro:

- **ReLU** (`max(0, u)`) tem derivada exatamente 1 na região ativa. O gradiente atravessa muitas camadas sem se dissolver. O preço é o *neurônio morto*: se a entrada líquida fica sempre negativa, a saída é sempre zero, a derivada é sempre zero, e o neurônio nunca mais se recupera.

- **Inicialização dimensionada pelo fan-in.** A entrada líquida é a soma de `n` produtos, então a escala dos pesos precisa depender de `n`. **Xavier Glorot e Yoshua Bengio** (2010) propuseram desvio `√(1/n)` para ativações simétricas; **Kaiming He** e coautores (2015) corrigiram para `√(2/n)` na ReLU, compensando o fato de ela zerar metade das ativações. Uma faixa fixa, independente do fan-in, é o que costuma matar uma rede profunda nas primeiras épocas.

- **Momento e otimizadores adaptativos**, que estabilizam a descida em superfícies mal-condicionadas.

Este projeto implementa a sigmoide, a ReLU e as três estratégias de inicialização, e a `Main` usa He justamente porque a arquitetura de exemplo tem duas camadas ReLU — com inicialização uniforme, ela satura e para de aprender nas primeiras épocas.

---

# Parte II — O projeto de redes neurais

## 9. As três implementações

As três classes em `core/` não são alternativas: são a mesma história contada em ordem. Vale ler nesta sequência.

| Classe | Arquitetura | O que ela mostra |
|---|---|---|
| **`Perceptron`** | Uma camada, sigmoide | O modelo de Rosenblatt. Resolve AND, OR, NAND. Não resolve XOR — e o motivo não é o treinamento. |
| **`MultilayerPerceptron`** | Uma camada oculta fixa, sigmoide fixa | A retropropagação escrita de forma direta, com duas matrizes de pesos e laços explícitos. Resolve XOR. Toda a mecânica está visível. |
| **`NeuralNetwork`** | Arbitrária, via vetor de arquitetura | A mesma coisa, generalizada. Camadas viram objetos (`Layer`), ativações viram objetos (`ActivationFunction`), e a topologia vira um parâmetro. |

Passar de `MultilayerPerceptron` para `NeuralNetwork` é a passagem de "eu implementei uma rede" para "eu implementei redes". A arquitetura vira dado:

```java
int[] arquitetura = { 10, 19, 19, 29 };   // 10 entradas, duas ocultas de 19, 29 saídas
```

## 10. Estrutura do código

```
neural_networks/
├── Main.java                     ponto de entrada: prepara a base, monta a rede, treina
├── activations/
│   ├── ActivationFunction.java   contrato: forward + derivada
│   ├── Sigmoid.java              a suavização do degrau que tornou o backprop possível
│   └── ReLU.java                 a ativação que destravou as redes profundas
├── core/
│   ├── Perceptron.java           camada única (Rosenblatt)
│   ├── MultilayerPerceptron.java uma camada oculta, escrita à mão
│   ├── NeuralNetwork.java        rede genérica, com validação e parada antecipada
│   ├── Layer.java                camada densa: forward, backward, update
│   └── Inicializacao.java        UNIFORME | XAVIER | HE
├── utils/
│   ├── Base.java                 carga, embaralhamento, divisão de validação, normalização
│   ├── Statistics.java           métricas por época, relatório .log e .csv
│   ├── GradientCheck.java        verificação numérica do gradiente
│   ├── FileUtils.java            preparação das bases (uso pontual)
│   └── plotCharts.py             gráficos a partir dos CSVs
├── bases/datasets/               bases de treino e teste
└── data/                         logs e CSVs gerados (criado automaticamente)
```

## 11. Verificação numérica do gradiente

A retropropagação é um algoritmo silencioso quando está errado. Um sinal trocado, uma derivada aplicada à camada errada, um índice deslocado — nada disso lança exceção. A rede treina menos, ou não treina, e o sintoma é idêntico ao de uma taxa de aprendizado mal escolhida.

A defesa é comparar o gradiente calculado pelo algoritmo com a definição de derivada, que não depende de implementação nenhuma:

```
∂E/∂w  ≈  [ E(w + ε) − E(w − ε) ] / (2ε)
```

Duas propagações para frente por peso. Lentíssimo — inútil para treinar, perfeito como referência. Se os dois valores concordam até várias casas decimais, a implementação está correta; se não concordam, o índice do peso divergente aponta a camada onde está o defeito.

```bash
java -cp out neural_networks.utils.GradientCheck
```

```
1 camada, sigmoide           (3 -> 2)        [OK] maior erro relativo = 2,602e-11
2 camadas, sigmoide          (4 -> 5 -> 3)   [OK] maior erro relativo = 8,427e-10
3 camadas, sigmoide          (6 -> 8 -> 8 -> 4) [OK] maior erro relativo = 4,591e-08
4 camadas, sigmoide          (5 -> 7 -> 7 -> 7 -> 2) [OK] maior erro relativo = 3,645e-07
3 camadas, ReLU + sigmoide   (6 -> 9 -> 9 -> 4) [OK] maior erro relativo = 4,733e-08

Controle negativo — a mesma rede com a heurística |1 − saída| ligada.
3 camadas, sigmoide + heurística  [FALHOU] maior erro relativo = 1,000e+00
```

Como ler os números: abaixo de `1e-7` o gradiente está correto; entre `1e-7` e `1e-4` é suspeito, e só aceitável com ReLU, cujo joelho em zero produz divergência legítima; acima de `1e-4` há defeito.

O último caso é um **controle negativo** deliberado. A implementação original multiplicava o delta da camada de saída por `|1 − saída|`, um fator que não é o gradiente de nenhuma função de erro padrão — ele desliga o aprendizado justamente onde a rede está acertando. A verificação o reprova com erro relativo 1,0. O fator continua disponível, desligado por padrão, via `setHeuristicaAceleracao(true)`, para quem quiser reproduzir o comportamento antigo.

Rodar a verificação leva menos de um segundo. `Main` a executa antes de qualquer treinamento.

## 12. Validação e parada antecipada

Um treinamento tem três conjuntos, com papéis distintos:

| Conjunto | Papel |
|---|---|
| **Treino** | Ajusta os pesos. |
| **Validação** | Decide *quando parar* e compara configurações. |
| **Teste** | Estima generalização. Tocado uma única vez, no fim. |

A distinção entre os dois últimos é metodológica, não burocrática. Escolher a época de parada olhando o conjunto de teste faz o teste participar da construção do modelo — e ele deixa de medir o que se propõe a medir. Um relatório que diz "melhor época de teste: 78" está, sem perceber, informando um resultado otimista.

A divisão é feita **em memória**, sem tocar em nenhum arquivo:

```java
Base base = new Base("abalone", 28, semente);
base.shuffle();                 // as amostras vêm agrupadas por classe
base.separarValidacao(0.2);     // 20% do treino viram validação
base.normalize();               // escala estimada só com o treino restante
```

A ordem importa nas três linhas. Embaralhar antes evita que a validação seja composta só pelas últimas classes carregadas. Normalizar depois garante que os parâmetros de escala venham apenas do treino — calcular mínimo e máximo com dados de validação já é vazamento.

A parada antecipada acompanha o erro de validação, guarda os pesos da melhor época e interrompe quando não há mais progresso:

```java
rna.setEarlyStopping(50, 1e-5);   // 50 épocas de paciência, melhora mínima de 1e-5
```

Ao final, **os pesos da melhor época são restaurados** — o modelo devolvido é o melhor visto, não o último. Numa execução de exemplo com o abalone, o teto de 10.000 épocas nunca é alcançado: o treino para na época 128 e restaura os pesos da 78.

## 13. Reprodutibilidade

Sem semente controlada, nenhum número de um relatório é verificável. Cinco execuções idênticas de uma mesma configuração podem dar de 16% a 0% de acurácia, incluindo colapsos totais — e nada no código indica qual das cinco você está vendo.

Toda aleatoriedade dos dois projetos passa por uma semente explícita:

```java
// redes neurais — a semente percorre base, divisão e inicialização de pesos
long semente = 42L;
Base base = new Base("abalone", 28, semente);
NeuralNetwork rna = new NeuralNetwork(base, arq, fs, epochs, lr, range, Inicializacao.HE, semente);

// algoritmos genéticos — um gerador único para todo o projeto
Rng.setSeed(42L);
long usada = Rng.setSeedAleatoria();   // execução independente, mas registrável
```

Para uma bateria de execuções independentes e ainda assim reproduzíveis em conjunto, derive a semente do índice: `Rng.setSeed(sementeBase + i)`. Registre sempre a semente ao lado do resultado.

## 14. Formato das bases

**Arquivo de teste** — `bases/datasets/<nome>/<nome>.test`:

```
1048 10 29          ← amostras, atributos de entrada, neurônios de saída
1,0 0,0 0,0 ...     ← entradas, seguidas da saída esperada em um-de-N
```

**Arquivos de classe** — `bases/datasets/<nome>/classes/classN`:

```
516                 ← número de linhas
1,0 0,0 0,7 ... 1   ← entradas, seguidas do rótulo bruto
```

Os dados usam **vírgula decimal**. A leitura fixa a localidade `pt-BR` explicitamente, de modo que o projeto funciona em qualquer máquina, independentemente da configuração regional do sistema.

Bases disponíveis: `abalone`, `tic-tac-toe`, `horse-colic`, `inflamatory`, `xor`, `and`.

---

# Parte III — Algoritmos genéticos

## 15. A ideia da computação evolutiva

A evolução biológica é um algoritmo de otimização. Ela não tem objetivo, não calcula derivadas e não sabe nada sobre o espaço em que opera — e ainda assim produz soluções extremamente boas para problemas difíceis, usando três mecanismos: **variação** (mutação e recombinação), **seleção** (quem se reproduz) e **hereditariedade** (o que passa adiante).

**John Holland**, na Universidade de Michigan, formalizou essa intuição como método de busca em *Adaptation in Natural and Artificial Systems* (1975). Ideias próximas surgiram de forma independente na mesma época: as *estratégias de evolução* de Ingo Rechenberg e Hans-Paul Schwefel, na Alemanha, e a *programação evolutiva* de Lawrence Fogel. O livro de David Goldberg, de 1989, consolidou os algoritmos genéticos como ferramenta de engenharia.

O que torna um AG útil é o que ele **não** exige. Ele não precisa que a função objetivo seja diferenciável, contínua, ou sequer expressável em forma fechada — basta conseguir avaliar candidatos e compará-los. Isso o coloca exatamente onde os métodos baseados em gradiente não chegam: espaços combinatórios, funções com muitos ótimos locais, problemas em que "avaliar" é rodar uma simulação.

O ciclo, a cada geração:

```
população  ──recombinação──►  filhos
     │                          │
     └──── mutação ──► mutantes │
                    │           │
                    ▼           ▼
              candidatos = pais + mutantes + filhos
                    │
              elitismo + seleção
                    │
                    ▼
             nova população
```

## 16. O motor e os problemas

`Ag` é agnóstico ao problema: conhece apenas as interfaces `Ind` (um indivíduo sabe se recombinar, se mutar e se avaliar) e `IndFactory` (sabe criar indivíduos aleatórios). Trocar de problema custa duas classes pequenas, sem tocar no motor.

```
genetic_algorithms/
├── AgMain.java              ponto de entrada
├── Ag.java                  o motor evolutivo
├── Ind.java                 contrato de um indivíduo
├── IndFactory.java          contrato de uma fábrica
├── Rng.java                 gerador único, para reprodutibilidade
├── RoletaViciada.java       seleção proporcional à aptidão
├── NRainhasInd.java         + fábrica — problema combinatório
├── DixonInd.java            + fábrica — otimização contínua unimodal
├── LangermannInd.java       + fábrica — otimização contínua multimodal
└── PermInd.java             + fábrica — otimização contínua mal-condicionada
```

**N-Rainhas** é o problema combinatório. A codificação é `genes[coluna] = linha`, o que já elimina por construção os ataques verticais; se o vetor for uma permutação, os horizontais também desaparecem, restando apenas as diagonais. É o único problema do conjunto com ótimo conhecido e igual a zero, e portanto o único em que o critério de parada por otimalidade de `Ag` funciona.

> **Correção registrada.** O construtor sorteava linhas no intervalo `0..N` *inclusive* — N+1 linhas num tabuleiro de N. Como a avaliação só conta colisões entre rainhas, um tabuleiro com uma linha a mais é *mais fácil* de resolver, e o algoritmo declarava ótimas soluções impossíveis. Em 100 execuções com N=6, nenhuma solução era válida. Com `i < n`, a taxa de soluções válidas passa a 100% para N ≤ 8, 95% para N=10 e 81% para N=12 — que são os números reais de convergência do algoritmo. `NRainhasInd.isValido()` verifica a propriedade e serve de teste de regressão.

**As três funções contínuas** foram escolhidas por atacarem capacidades diferentes:

| Função | Ótimo | O que ela testa |
|---|---|---|
| **Dixon-Price** | 0 | Um vale longo, estreito e curvo. Testa refinamento fino — passo grande demais sai do vale. |
| **Langermann** | −9,25625 (nesta implementação, em [0,10]²) | Superfície cheia de picos irregulares. Testa exploração: não ficar preso no primeiro vale. |
| **Perm β** | 0 | Termos com potências de ordem crescente, escalas radicalmente diferentes. Testa robustez a mau condicionamento. |

> **Cuidado com o ótimo de referência da Langermann.** O valor `−9,25625` é o mínimo *desta* parametrização em `[0,10]²`, confirmado por busca exaustiva em grade. O número `−4,1558`, que aparece na maioria dos livros e sites, corresponde a outra parametrização da mesma família. Comparar um resultado correto deste código com aquele valor faz o resultado parecer errado. Os ótimos estão declarados como constantes `OTIMO_CONHECIDO` em cada classe.

---

# Parte IV — Operação

## 17. Como compilar e executar

**Requisitos:** JDK 17 ou superior. Para os gráficos: Python 3 com `pandas` e `matplotlib`.

### Redes neurais

Os caminhos das bases são relativos, então os comandos precisam rodar de dentro de `neural_networks/`.

```bash
cd neural_networks
javac -d out $(find . -name "*.java")

java -cp out neural_networks.utils.GradientCheck   # verifica a retropropagação
java -cp out neural_networks.Main                  # treina
```

No Windows, com PowerShell:

```powershell
cd neural_networks
javac -d out (Get-ChildItem -Recurse -Filter *.java | ForEach-Object { $_.FullName })
java -cp out neural_networks.Main
```

Cada execução escreve em `data/` um relatório `.log` e um `.csv` com o histórico por época. Os gráficos:

```bash
python utils/plotCharts.py individual "NeuralNetwork"
python utils/plotCharts.py comparativo "NeuralNetwork" "MLP"
```

### Algoritmos genéticos

```bash
cd genetic_algorithms
javac -d out *.java
java -cp out AgMain
```

Os parâmetros — tamanho da população, elitismo, gerações, problema, semente — estão no corpo de `AgMain.main`.

## 18. Limitações conhecidas e preservadas

Algumas escolhas destes projetos não são ótimas, e permanecem assim de propósito: fazem parte do enunciado acadêmico ou da progressão didática. Estão documentadas no Javadoc das classes correspondentes e listadas aqui para que não sejam lidas como descuido.

**Redes neurais**

- **Balanceamento por replicação.** Todas as classes são levadas ao tamanho da maior copiando amostras. `Base.randomize()` deveria perturbar as cópias, mas devolve a amostra intacta — o resultado é um conjunto de treino majoritariamente formado por duplicatas exatas, com classes raras representadas por uma única amostra repetida centenas de vezes.
- **Codificação dos rótulos.** O alvo de treino é montado a partir do *número do arquivo* de classe, que é um posto ordenado; o arquivo de teste usa a codificação direta do rótulo. Quando algum valor de rótulo está ausente da base — o que ocorre no abalone —, as duas codificações apontam para neurônios diferentes. Isso invalida a comparação direta entre métricas de treino e de teste.
- **Métrica de erro.** O que é registrado como "erro" é a soma dos desvios absolutos por amostra, não o erro quadrático médio. As colunas usam o termo neutro "erro" justamente para não repetir a designação incorreta de MSE.
- **Ausentes por escopo:** momento, regularização, mini-lote, softmax com entropia cruzada, matriz de confusão, F1 macro, persistência de pesos, validação cruzada.

**Algoritmos genéticos**

- **A roleta viciada usa `1/avaliação`**, o que só é válido para funções estritamente positivas. Com a Langermann, cuja região de interesse é negativa, a seleção passa a ignorar os melhores indivíduos. Torneio ou escalonamento linear resolveriam.
- **Não há seleção de pais:** os casais são formados por posição na lista. Também não há taxa de crossover nem de mutação — todo indivíduo cruza e todo indivíduo gera um mutante.
- **O crossover de N-Rainhas é de um ponto**, o que não preserva a propriedade de permutação. PMX, OX ou Cycle Crossover preservariam.
- **Com `alpha` fixo em 0,5**, os dois primeiros filhos do crossover aritmético são numericamente idênticos.
- **Os domínios divergem da literatura:** Dixon-Price usa `[−n, n]` onde o padrão é `[−10, 10]`; Perm β usa `[−10, 10]` onde o padrão é `[−n, n]`.
- **Ausentes por escopo:** medida de diversidade por geração, contador de avaliações da função objetivo, bateria automatizada de execuções com média e desvio, curva de convergência exportada.
**Bases de dados**

- Abalone, Tic-Tac-Toe, Horse Colic e Acute Inflammations — UCI Machine Learning Repository.
