# Tópicos em Engenharia: Inteligência Artificial - 3º Trabalho
Desenvolvimento de uma Rede Neural Convolucional para resolver o problema de reconhecimento de dígitos da base de dados MNIST

## Alunos

Matrícula  | Nome
-----------|-------------------------
16/0019311 | Victor André Gris Costa
16/0032458 | José Luiz Gomes Nogueira

## Ambiente

* Este trabalho foi desenvolvido e testado com Python 3.7.3 no Ubuntu 18.04 LTS

Pacote        | Versão
--------------|---------
Keras         | 2.2.4
keras-metrics | 1.1.0
tensorflow    | 1.13.1
h5py          | 2.9.0
python-mnist  | 0.6
numpy         | 1.16.4
matplotlib    | 3.1.0

## Modo de uso

1. Instale todas as dependências listadas acima
2. Mude para o diretório raiz do projeto
3. Para rodar o programa execute com `python src/main.py [opções]`
    * Dependendo de seu ambiente precisará trocar `python` por `python3` ou até `python3.7` para garantir a versão correta
4. A opção `-h` ou `--help` lhe guiará melhor pelas minúcias do uso do programa
5. Caso deseje treinar e fazer a geração de dados completa, execute com as opções: `-t -g -f -a -w -s`
6. Caso deseje só carregar os modelos treinados e gerar todos os dados, execute com as opções `-g -f -a -w -s`
7. Caso deseje salvar os dados na pasta `img`, adicione a opção `-o`

## Estrutura

O programa está subdividido em 3 arquivos. O arquivo main.py serve para a lógica de execução em alto nível.
O arquivo funcoes.py armazena múltiplas funções de alto nível para deixar a main.py limpa e menos bagunçada.
Ambos os arquivos fazem uso da classe `MnistModel` criada no arquivo cnn.py. Essa classe tem como função
envolver o modelo que estamos treinando, armazenar dados e fornecer uma interface simplificada para a manipulação
do modelo. Essa interface permite a criação de um modelo por certos parâmetros, salvamento e carregamento de modelos
e acesso aos dados, como histórico de treinamento, ativação das camadas convolucionais e filtros treinados.
