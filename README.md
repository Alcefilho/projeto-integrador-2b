# Sistema de Triagem de Sintomas 🏥

Este projeto é um sistema de triagem de sintomas utilizando inteligência artificial, desenvolvido como parte do Projeto Integrador II-B da PUC Goiás.

## Descrição

O sistema simula um ambiente de triagem clínica, onde o usuário pode informar sintomas, sinais vitais e fatores de risco. Utilizando uma árvore de decisão treinada com dados sintéticos realistas, o sistema sugere possíveis diagnósticos (como resfriado, gripe, COVID-19, casos graves ou saudável) e fornece recomendações iniciais.

**Atenção:** Este sistema é apenas para fins educacionais e não substitui a avaliação médica profissional.

## Funcionalidades

- Formulário interativo para entrada de sintomas e sinais vitais
- Predição automática do diagnóstico provável
- Visualização das probabilidades de cada diagnóstico
- Recomendações personalizadas conforme o resultado
- Análise dos dados sintéticos utilizados no treinamento
- Visualização gráfica da árvore de decisão e das principais variáveis clínicas

## Tecnologias Utilizadas

- [Streamlit](https://streamlit.io/) (interface web)
- [scikit-learn](https://scikit-learn.org/) (machine learning)
- [pandas](https://pandas.pydata.org/) e [numpy](https://numpy.org/) (manipulação de dados)
- [plotly](https://plotly.com/python/) (gráficos interativos)
- [faker](https://faker.readthedocs.io/) (geração de dados sintéticos)
- [graphviz](https://graphviz.gitlab.io/) (visualização de árvores)

## Como executar

1. Instale as dependências:
   ```sh
   uv pip install -r requirements.txt
   ```
   Ou, se estiver usando o `pyproject.toml`:
   ```sh
   uv pip install -e .
   ```

2. Execute o aplicativo:
   ```sh
   uv streamlit run main.py
   ```

3. Acesse o sistema pelo navegador, normalmente em [http://localhost:8501](http://localhost:8501).

## Estrutura do Projeto

- `main.py`: Código principal do sistema e interface Streamlit
- `pyproject.toml`: Dependências do projeto
- `README.md`: Este arquivo

## Aviso

Este sistema não realiza diagnóstico médico real. Em caso de sintomas graves, procure um serviço de saúde imediatamente.

---

Projeto Integrador II-B - PUC