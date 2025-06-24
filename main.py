# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz
from sklearn.tree import export_graphviz
import pydotplus
from faker import Faker
import random
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Configurando a página
st.set_page_config(
    page_title="Sistema de Triagem de Sintomas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e descrição
st.title("🏥 Sistema de Triagem de Sintomas")
st.markdown("""
Este sistema utiliza inteligência artificial para realizar uma triagem inicial de sintomas,
auxiliando na identificação de possíveis condições de saúde com base em uma árvore de decisão.

**IMPORTANTE**: Este sistema é apenas educacional e não substitui a avaliação médica profissional.
""")

# Função para gerar dados de pacientes clinicamente mais realistas
@st.cache_data
def gerar_dados_pacientes(n=2000, seed=42):
    # Configurando o Faker para gerar dados realistas
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    dados = []
    
    # Definindo faixas etárias e suas probabilidades
    faixas_etarias = [
        (0, 12),    # Crianças
        (13, 19),   # Adolescentes
        (20, 39),   # Jovens adultos
        (40, 59),   # Adultos
        (60, 85)    # Idosos
    ]
    
    # Probabilidades de cada faixa etária na população
    prob_faixas = [0.15, 0.10, 0.35, 0.25, 0.15]
    
    # Definindo prevalência de doenças por faixa etária
    # Formato: [crianças, adolescentes, jovens adultos, adultos, idosos]
    prevalencia_hipertensao = [0.01, 0.02, 0.08, 0.30, 0.65]
    prevalencia_diabetes = [0.005, 0.01, 0.04, 0.15, 0.25]
    prevalencia_doenca_cardiaca = [0.005, 0.01, 0.02, 0.10, 0.30]
    prevalencia_doenca_respiratoria = [0.08, 0.06, 0.05, 0.08, 0.15]
    prevalencia_imunossupressao = [0.01, 0.01, 0.02, 0.03, 0.05]
    
    # Definindo prevalência de condições
    prevalencia_resfriado = 0.30  # 30% dos casos são resfriados
    prevalencia_gripe = 0.20      # 20% dos casos são gripe
    prevalencia_covid = 0.15      # 15% dos casos são COVID-19
    prevalencia_grave = 0.05      # 5% dos casos são graves
    prevalencia_saudavel = 1 - (prevalencia_resfriado + prevalencia_gripe + prevalencia_covid + prevalencia_grave)
    
    for _ in range(n):
        # Dados demográficos
        faixa_escolhida = random.choices(range(len(faixas_etarias)), weights=prob_faixas)[0]
        idade_min, idade_max = faixas_etarias[faixa_escolhida]
        idade = random.randint(idade_min, idade_max)
        
        sexo = random.choice(['M', 'F'])
        
        # Comorbidades baseadas na faixa etária
        hipertensao = random.random() < prevalencia_hipertensao[faixa_escolhida]
        diabetes = random.random() < prevalencia_diabetes[faixa_escolhida]
        doenca_cardiaca = random.random() < prevalencia_doenca_cardiaca[faixa_escolhida]
        doenca_respiratoria = random.random() < prevalencia_doenca_respiratoria[faixa_escolhida]
        imunossupressao = random.random() < prevalencia_imunossupressao[faixa_escolhida]
        
        # Decidindo o diagnóstico primeiro (para gerar sintomas mais coerentes)
        diagnostico_probs = [
            prevalencia_saudavel,
            prevalencia_resfriado,
            prevalencia_gripe,
            prevalencia_covid,
            prevalencia_grave
        ]
        diagnostico = random.choices(range(5), weights=diagnostico_probs)[0]
        
        # Ajustando probabilidades de sintomas com base no diagnóstico
        # Saudável
        if diagnostico == 0:
            prob_febre = 0.01
            prob_tosse = 0.05
            prob_dor_garganta = 0.03
            prob_coriza = 0.05
            prob_falta_ar = 0.01
            prob_dor_corpo = 0.05
            prob_dor_cabeca = 0.08
            prob_perda_olfato = 0.001
            prob_perda_paladar = 0.001
            prob_diarreia = 0.03
            prob_nausea = 0.02
            prob_vomito = 0.01
            prob_fadiga = 0.05
            prob_contato_covid = 0.05
            duracao_sintomas_range = (0, 1)
            
        # Resfriado
        elif diagnostico == 1:
            prob_febre = 0.3  # Febre leve ou ausente
            prob_tosse = 0.7
            prob_dor_garganta = 0.8
            prob_coriza = 0.9
            prob_falta_ar = 0.05
            prob_dor_corpo = 0.3
            prob_dor_cabeca = 0.6
            prob_perda_olfato = 0.2  # Pode ocorrer por congestão nasal
            prob_perda_paladar = 0.1
            prob_diarreia = 0.05
            prob_nausea = 0.05
            prob_vomito = 0.02
            prob_fadiga = 0.4
            prob_contato_covid = 0.1
            duracao_sintomas_range = (3, 10)
            
        # Gripe
        elif diagnostico == 2:
            prob_febre = 0.9  # Febre alta é comum
            prob_tosse = 0.8
            prob_dor_garganta = 0.7
            prob_coriza = 0.7
            prob_falta_ar = 0.2
            prob_dor_corpo = 0.9
            prob_dor_cabeca = 0.8
            prob_perda_olfato = 0.1
            prob_perda_paladar = 0.1
            prob_diarreia = 0.1
            prob_nausea = 0.2
            prob_vomito = 0.1
            prob_fadiga = 0.9
            prob_contato_covid = 0.1
            duracao_sintomas_range = (5, 14)
            
        # COVID-19
        elif diagnostico == 3:
            prob_febre = 0.8
            prob_tosse = 0.8
            prob_dor_garganta = 0.4
            prob_coriza = 0.4
            prob_falta_ar = 0.3
            prob_dor_corpo = 0.6
            prob_dor_cabeca = 0.6
            prob_perda_olfato = 0.7  # Sintoma característico
            prob_perda_paladar = 0.7  # Sintoma característico
            prob_diarreia = 0.3
            prob_nausea = 0.2
            prob_vomito = 0.1
            prob_fadiga = 0.8
            prob_contato_covid = 0.6
            duracao_sintomas_range = (7, 21)
            
        # Caso grave (pode ser COVID grave ou outra condição respiratória grave)
        else:
            prob_febre = 0.9  # Febre alta
            prob_tosse = 0.9
            prob_dor_garganta = 0.5
            prob_coriza = 0.3
            prob_falta_ar = 0.9  # Sintoma característico de casos graves
            prob_dor_corpo = 0.8
            prob_dor_cabeca = 0.7
            prob_perda_olfato = 0.5
            prob_perda_paladar = 0.5
            prob_diarreia = 0.4
            prob_nausea = 0.5
            prob_vomito = 0.4
            prob_fadiga = 0.95
            prob_contato_covid = 0.4
            duracao_sintomas_range = (3, 7)  # Casos graves tendem a buscar atendimento mais rápido
        
        # Gerando sintomas com base nas probabilidades
        febre = random.random() < prob_febre
        temperatura = None
        if febre:
            # Temperatura varia conforme o diagnóstico
            if diagnostico == 0:  # Saudável
                temperatura = round(random.uniform(37.0, 37.7), 1)
            elif diagnostico == 1:  # Resfriado
                temperatura = round(random.uniform(37.5, 38.2), 1)
            elif diagnostico == 2:  # Gripe
                temperatura = round(random.uniform(38.0, 39.5), 1)
            elif diagnostico == 3:  # COVID-19
                temperatura = round(random.uniform(37.8, 39.0), 1)
            else:  # Caso grave
                temperatura = round(random.uniform(38.5, 40.2), 1)
        
        tosse = random.random() < prob_tosse
        tosse_seca = None
        if tosse:
            # COVID e gripe tendem a ter tosse seca
            if diagnostico in [2, 3]:
                tosse_seca = random.random() < 0.8
            else:
                tosse_seca = random.random() < 0.4
        
        dor_garganta = random.random() < prob_dor_garganta
        coriza = random.random() < prob_coriza
        falta_ar = random.random() < prob_falta_ar
        
        # Aumentar probabilidade de falta de ar em idosos e pessoas com comorbidades
        if idade > 60 or doenca_respiratoria or doenca_cardiaca:
            falta_ar = falta_ar or (random.random() < 0.3)
        
        dor_corpo = random.random() < prob_dor_corpo
        dor_cabeca = random.random() < prob_dor_cabeca
        perda_olfato = random.random() < prob_perda_olfato
        perda_paladar = random.random() < prob_perda_paladar
        
        # Sintomas gastrointestinais
        diarreia = random.random() < prob_diarreia
        nausea = random.random() < prob_nausea
        vomito = random.random() < prob_vomito
        
        # Fadiga (comum em várias condições)
        fadiga = random.random() < prob_fadiga
        
        # Fatores epidemiológicos
        contato_covid = random.random() < prob_contato_covid
        viagem_recente = random.random() < 0.1
        
        # Duração dos sintomas (em dias)
        min_duracao, max_duracao = duracao_sintomas_range
        duracao_sintomas = random.randint(min_duracao, max_duracao) if any([febre, tosse, dor_garganta, falta_ar, dor_corpo, dor_cabeca]) else 0
        
        # Saturação de oxigênio (normal: 95-100%)
        saturacao_o2 = None
        if falta_ar:
            if diagnostico == 4:  # Caso grave
                saturacao_o2 = random.randint(80, 92)
            elif diagnostico == 3:  # COVID-19
                saturacao_o2 = random.randint(88, 95)
            else:
                saturacao_o2 = random.randint(92, 98)
        else:
            saturacao_o2 = random.randint(95, 100)
        
        # Frequência respiratória (normal: 12-20 rpm)
        freq_respiratoria = None
        if falta_ar:
            if diagnostico == 4:  # Caso grave
                freq_respiratoria = random.randint(24, 40)
            else:
                freq_respiratoria = random.randint(20, 28)
        else:
            freq_respiratoria = random.randint(12, 20)
        
        # Dados do paciente
        paciente = {
            'idade': idade,
            'sexo': sexo,
            'hipertensao': int(hipertensao),
            'diabetes': int(diabetes),
            'doenca_cardiaca': int(doenca_cardiaca),
            'doenca_respiratoria': int(doenca_respiratoria),
            'imunossupressao': int(imunossupressao),
            'febre': int(febre),
            'temperatura': temperatura,
            'tosse': int(tosse),
            'tosse_seca': int(tosse_seca) if tosse_seca is not None else None,
            'dor_garganta': int(dor_garganta),
            'coriza': int(coriza),
            'falta_ar': int(falta_ar),
            'dor_corpo': int(dor_corpo),
            'dor_cabeca': int(dor_cabeca),
            'perda_olfato': int(perda_olfato),
            'perda_paladar': int(perda_paladar),
            'diarreia': int(diarreia),
            'nausea': int(nausea),
            'vomito': int(vomito),
            'fadiga': int(fadiga),
            'contato_covid': int(contato_covid),
            'viagem_recente': int(viagem_recente),
            'duracao_sintomas': duracao_sintomas,
            'saturacao_o2': saturacao_o2,
            'freq_respiratoria': freq_respiratoria,
            'diagnostico': diagnostico
        }
        
        dados.append(paciente)
    
    return pd.DataFrame(dados)

# Função para atribuir diagnósticos com base em regras clínicas
def atribuir_diagnosticos(df):
    # Já temos diagnósticos atribuídos na geração de dados
    # Esta função pode ser usada para sobrescrever diagnósticos com base em regras específicas
    
    # Casos graves que não foram identificados na geração inicial
    mask_grave_nao_identificado = (
        (df['diagnostico'] != 4) &  # Não foi classificado como grave inicialmente
        (
            ((df['saturacao_o2'].notna()) & (df['saturacao_o2'] < 90)) |  # Saturação baixa
            ((df['freq_respiratoria'].notna()) & (df['freq_respiratoria'] > 30)) |  # Taquipneia grave
            ((df['temperatura'].notna()) & (df['temperatura'] > 39.5)) |  # Febre muito alta
            (
                (df['falta_ar'] == 1) & 
                (df['idade'] > 70) & 
                ((df['hipertensao'] == 1) | (df['diabetes'] == 1) | (df['doenca_cardiaca'] == 1))
            )  # Idosos com comorbidades e falta de ar
        )
    )
    df.loc[mask_grave_nao_identificado, 'diagnostico'] = 4
    
    # COVID-19 não identificado inicialmente
    mask_covid_nao_identificado = (
        (df['diagnostico'] != 3) &  # Não foi classificado como COVID inicialmente
        (df['diagnostico'] != 4) &  # Não é um caso grave
        (
            ((df['perda_olfato'] == 1) & (df['perda_paladar'] == 1)) |  # Perda de olfato e paladar juntos é muito sugestivo
            ((df['febre'] == 1) & (df['tosse'] == 1) & (df['contato_covid'] == 1) & (df['fadiga'] == 1))  # Sintomas típicos + contato
        )
    )
    df.loc[mask_covid_nao_identificado, 'diagnostico'] = 3
    
    return df

# Função para treinar o modelo
@st.cache_resource
def treinar_modelo():
    # Gerando dados de pacientes com diagnósticos já atribuídos
    df = gerar_dados_pacientes()
    
    # Aplicando regras adicionais para refinar diagnósticos
    df = atribuir_diagnosticos(df)
    
    # Removendo colunas que não serão usadas na árvore de decisão
    colunas_para_modelo = [
        'idade', 'febre', 'tosse', 'tosse_seca', 'dor_garganta', 'coriza', 
        'falta_ar', 'dor_corpo', 'dor_cabeca', 'perda_olfato', 'perda_paladar', 
        'diarreia', 'fadiga', 'contato_covid', 'duracao_sintomas',
        'hipertensao', 'diabetes', 'doenca_respiratoria', 'doenca_cardiaca',
        'saturacao_o2', 'freq_respiratoria'
    ]
    
    # Removendo valores nulos para treinamento
    df_modelo = df.copy()
    for col in colunas_para_modelo:
        if df_modelo[col].isna().any():
            # Para colunas numéricas, preencher com a média
            if df_modelo[col].dtype in ['int64', 'float64']:
                df_modelo[col] = df_modelo[col].fillna(df_modelo[col].mean())
            else:
                # Para colunas categóricas, preencher com o mais frequente
                df_modelo[col] = df_modelo[col].fillna(df_modelo[col].mode()[0])
    
    # Separando features e target
    X = df_modelo[colunas_para_modelo]
    y = df_modelo['diagnostico']
    
    # Dividindo em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Criando e treinando o modelo de árvore de decisão
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Avaliando o modelo
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Retornando o modelo, métricas e dados
    return clf, accuracy, report, X.columns, df

# Função para visualizar a árvore de decisão
def visualizar_arvore(clf, feature_names, class_names):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    graph = graphviz.Source(dot_data)
    return graph

# Função para gerar gráficos de distribuição de diagnósticos
def gerar_graficos_distribuicao(df):
    # Mapeando diagnósticos para nomes mais amigáveis
    diagnostico_map = {
        0: 'Saudável',
        1: 'Resfriado',
        2: 'Gripe',
        3: 'COVID-19',
        4: 'Urgente'
    }
    
    df['diagnostico_nome'] = df['diagnostico'].map(diagnostico_map)
    
    # Gráfico de distribuição de diagnósticos
    fig_diag = px.pie(
        df, 
        names='diagnostico_nome', 
        title='Distribuição de Diagnósticos',
        color='diagnostico_nome',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_diag.update_traces(textposition='inside', textinfo='percent+label')
    
    # Gráfico de distribuição de idade por diagnóstico
    fig_idade = px.box(
        df, 
        x='diagnostico_nome', 
        y='idade', 
        color='diagnostico_nome',
        title='Distribuição de Idade por Diagnóstico',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Gráfico de sintomas mais comuns por diagnóstico
    sintomas = ['febre', 'tosse', 'dor_garganta', 'coriza', 'falta_ar', 
               'dor_corpo', 'dor_cabeca', 'perda_olfato', 'perda_paladar', 'diarreia', 'fadiga']
    
    sintomas_por_diagnostico = {}
    for diag in range(5):
        df_diag = df[df['diagnostico'] == diag]
        sintomas_por_diagnostico[diagnostico_map[diag]] = [df_diag[sintoma].mean() for sintoma in sintomas]
    
    df_sintomas = pd.DataFrame(sintomas_por_diagnostico, index=sintomas)
    
    fig_sintomas = px.imshow(
        df_sintomas,
        labels=dict(x="Diagnóstico", y="Sintoma", color="Frequência"),
        x=df_sintomas.columns,
        y=[s.replace('_', ' ').title() for s in df_sintomas.index],
        color_continuous_scale='YlOrRd',
        title='Frequência de Sintomas por Diagnóstico'
    )
    
    # Gráfico de saturação de oxigênio por diagnóstico
    fig_saturacao = px.box(
        df[df['saturacao_o2'].notna()], 
        x='diagnostico_nome', 
        y='saturacao_o2', 
        color='diagnostico_nome',
        title='Saturação de Oxigênio por Diagnóstico',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_saturacao.add_shape(
        type="line",
        x0=-0.5,
        x1=4.5,
        y0=95,
        y1=95,
        line=dict(color="red", width=2, dash="dash"),
    )
    fig_saturacao.add_annotation(
        x=4,
        y=95,
        text="Limite crítico (95%)",
        showarrow=False,
        yshift=10
    )
    
    return fig_diag, fig_idade, fig_sintomas, fig_saturacao

# Função para fazer previsão com base nos dados do usuário
def fazer_previsao(clf, feature_names, dados_usuario):
    # Criando um DataFrame com as respostas
    df_paciente = pd.DataFrame([dados_usuario])
    
    # Garantindo que todas as colunas necessárias estejam presentes e na ordem correta
    for col in feature_names:
        if col not in df_paciente.columns:
            df_paciente[col] = 0  # Valor padrão para colunas ausentes
    
    df_paciente = df_paciente[feature_names]
    
    # Fazendo a previsão
    resultado = clf.predict(df_paciente)[0]
    probabilidades = clf.predict_proba(df_paciente)[0]
    
    return resultado, probabilidades

# Função para exibir recomendações com base no diagnóstico
def exibir_recomendacoes(resultado, dados_usuario):
    st.subheader("Recomendações:")
    
    # Verificando fatores de risco
    idade_avancada = dados_usuario.get('idade', 0) >= 60
    tem_comorbidades = any([
        dados_usuario.get('hipertensao', 0) == 1,
        dados_usuario.get('diabetes', 0) == 1,
        dados_usuario.get('doenca_cardiaca', 0) == 1,
        dados_usuario.get('doenca_respiratoria', 0) == 1,
        dados_usuario.get('imunossupressao', 0) == 1
    ])
    
    saturacao_baixa = dados_usuario.get('saturacao_o2', 100) < 95 if dados_usuario.get('saturacao_o2') is not None else False
    
    if resultado == 0:
        st.success("✅ Você parece estar saudável.")
        st.write("- Continue monitorando seus sintomas.")
        st.write("- Mantenha hábitos saudáveis e medidas preventivas.")
        
    elif resultado == 1:
        st.info("ℹ️ Seus sintomas sugerem um resfriado comum.")
        st.write("- Descanse e mantenha-se hidratado.")
        st.write("- Considere medicamentos de venda livre para alívio dos sintomas, como analgésicos e descongestionantes.")
        st.write("- Gargarejo com água morna e sal pode aliviar a dor de garganta.")
        
        if idade_avancada or tem_comorbidades:
            st.warning("⚠️ Por ter fatores de risco, monitore seus sintomas com atenção.")
            st.write("- Procure um médico se os sintomas piorarem ou persistirem por mais de 7 dias.")
        else:
            st.write("- Procure um médico se os sintomas piorarem ou persistirem por mais de 10 dias.")
            
    elif resultado == 2:
        st.warning("⚠️ Seus sintomas sugerem um quadro gripal.")
        st.write("- Descanse, mantenha-se hidratado e considere medicamentos para alívio dos sintomas.")
        st.write("- Monitore sua temperatura regularmente.")
        st.write("- Isole-se para evitar transmissão a outras pessoas.")
        
        if idade_avancada or tem_comorbidades:
            st.warning("⚠️ Por ter fatores de risco, considere consultar um médico nas próximas 24 horas.")
            st.write("- Você pode se beneficiar de medicamentos antivirais se iniciados precocemente.")
        else:
            st.write("- Procure um médico se os sintomas piorarem, especialmente se desenvolver falta de ar ou febre persistente.")
            
    elif resultado == 3:
        st.warning("⚠️ Seus sintomas são compatíveis com COVID-19.")
        st.write("- Isole-se imediatamente para evitar transmissão.")
        st.write("- Procure realizar um teste para confirmar o diagnóstico.")
        st.write("- Monitore seus sintomas, temperatura e, se possível, saturação de oxigênio.")
        
        if saturacao_baixa:
            st.error("🚨 Sua saturação de oxigênio está abaixo do normal. Procure atendimento médico imediatamente.")
        
        if idade_avancada or tem_comorbidades:
            st.warning("⚠️ Por ter fatores de risco para COVID-19 grave, consulte um médico mesmo com sintomas leves.")
        else:
            st.write("- Procure atendimento médico se desenvolver falta de ar, dor persistente no peito, confusão mental ou lábios/face azulados.")
            
    elif resultado == 4:
        st.error("🚨 ATENÇÃO: Seus sintomas indicam necessidade de avaliação médica URGENTE.")
        st.write("- Procure um serviço de emergência ou ligue para os serviços de saúde imediatamente.")
        st.write("- Não espere que os sintomas piorem.")
        
        if saturacao_baixa:
            st.error("🚨 Sua saturação de oxigênio está baixa, o que pode indicar comprometimento respiratório significativo.")
    
    st.warning("**AVISO IMPORTANTE**: Este é apenas um sistema de triagem inicial e não substitui a avaliação médica profissional. Em caso de dúvidas ou agravamento dos sintomas, procure atendimento médico.")

# Função principal
def main():
    # Carregando o modelo e os dados
    clf, accuracy, report, feature_names, df = treinar_modelo()
    class_names = ['Saudável', 'Resfriado', 'Gripe', 'COVID-19', 'Urgente']
    
    # Criando abas
    tab1, tab2, tab3 = st.tabs(["📋 Triagem de Sintomas", "📊 Análise de Dados", "🌳 Visualização da Árvore"])
    
    # Aba de Triagem de Sintomas
    with tab1:
        st.header("Formulário de Triagem")
        st.write("Por favor, responda às seguintes perguntas sobre seus sintomas:")
        
        # Formulário para coleta de dados
        with st.form("formulario_triagem"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Dados Pessoais")
                idade = st.slider("Idade", 0, 100, 30)
                sexo = st.radio("Sexo", ["M", "F"])
                
                st.subheader("Comorbidades")
                hipertensao = st.checkbox("Hipertensão")
                diabetes = st.checkbox("Diabetes")
                doenca_cardiaca = st.checkbox("Doença Cardíaca")
                doenca_respiratoria = st.checkbox("Doença Respiratória")
                imunossupressao = st.checkbox("Imunossupressão")
            
            with col2:
                st.subheader("Sintomas Principais")
                febre = st.checkbox("Febre")
                if febre:
                    temperatura = st.number_input("Temperatura (°C)", 35.0, 42.0, 38.0, 0.1)
                else:
                    temperatura = None
                
                tosse = st.checkbox("Tosse")
                if tosse:
                    tosse_seca = st.radio("Tipo de tosse", ["Seca", "Produtiva (com catarro)"]) == "Seca"
                else:
                    tosse_seca = None
                
                dor_garganta = st.checkbox("Dor de Garganta")
                coriza = st.checkbox("Coriza/Congestão Nasal")
                falta_ar = st.checkbox("Falta de Ar/Dificuldade para Respirar")
                
                if falta_ar:
                    saturacao_o2 = st.slider("Saturação de Oxigênio (%)", 70, 100, 95)
                    freq_respiratoria = st.slider("Frequência Respiratória (resp/min)", 10, 40, 16)
                else:
                    saturacao_o2 = None
                    freq_respiratoria = None
            
            with col3:
                st.subheader("Sintomas Adicionais")
                dor_corpo = st.checkbox("Dor no Corpo/Muscular")
                dor_cabeca = st.checkbox("Dor de Cabeça")
                fadiga = st.checkbox("Fadiga/Cansaço")
                perda_olfato = st.checkbox("Perda de Olfato")
                perda_paladar = st.checkbox("Perda de Paladar")
                diarreia = st.checkbox("Diarreia")
                nausea = st.checkbox("Náusea")
                vomito = st.checkbox("Vômito")
                
                st.subheader("Informações Adicionais")
                contato_covid = st.checkbox("Contato com caso confirmado de COVID-19")
                duracao_sintomas = st.slider("Duração dos sintomas (dias)", 0, 30, 0)
            
            submitted = st.form_submit_button("Realizar Triagem")
        
        # Processando o formulário após envio
        if submitted:
            # Preparando os dados do usuário
            dados_usuario = {
                'idade': idade,
                'sexo': sexo,
                'hipertensao': int(hipertensao),
                'diabetes': int(diabetes),
                'doenca_cardiaca': int(doenca_cardiaca),
                'doenca_respiratoria': int(doenca_respiratoria),
                'imunossupressao': int(imunossupressao),
                'febre': int(febre),
                'temperatura': temperatura,
                'tosse': int(tosse),
                'tosse_seca': int(tosse_seca) if tosse_seca is not None else 0,
                'dor_garganta': int(dor_garganta),
                'coriza': int(coriza),
                'falta_ar': int(falta_ar),
                'dor_corpo': int(dor_corpo),
                'dor_cabeca': int(dor_cabeca),
                'perda_olfato': int(perda_olfato),
                'perda_paladar': int(perda_paladar),
                'diarreia': int(diarreia),
                'nausea': int(nausea),
                'vomito': int(vomito),
                'fadiga': int(fadiga),
                'contato_covid': int(contato_covid),
                'duracao_sintomas': duracao_sintomas,
                'saturacao_o2': saturacao_o2,
                'freq_respiratoria': freq_respiratoria
            }
            
            # Fazendo a previsão
            resultado, probabilidades = fazer_previsao(clf, feature_names, dados_usuario)
            
            # Exibindo o resultado
            st.header("Resultado da Triagem")
            
            # Exibindo o diagnóstico sugerido
            diagnostico_texto = class_names[resultado]
            
            if resultado == 0:
                st.success(f"Diagnóstico sugerido: {diagnostico_texto}")
            elif resultado in [1, 2]:
                st.info(f"Diagnóstico sugerido: {diagnostico_texto}")
            elif resultado == 3:
                st.warning(f"Diagnóstico sugerido: {diagnostico_texto}")
            else:
                st.error(f"Diagnóstico sugerido: {diagnostico_texto}")
            
            # Exibindo as probabilidades
            st.subheader("Probabilidades:")
            
            # Criando um gráfico de barras para as probabilidades
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[class_names[i] for i in range(len(class_names))],
                y=[probabilidades[i] for i in range(len(probabilidades))],
                marker_color=['green', 'blue', 'orange', 'red', 'darkred'],
                text=[f"{probabilidades[i]:.2f}" for i in range(len(probabilidades))],
                textposition='auto'
            ))
            fig.update_layout(
                title='Probabilidade de cada diagnóstico',
                xaxis_title='Diagnóstico',
                yaxis_title='Probabilidade',
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig)
            
            # Exibindo sinais vitais e fatores de risco
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sinais Vitais")
                if temperatura:
                    if temperatura < 37.5:
                        st.write(f"✅ Temperatura: {temperatura}°C (Normal)")
                    elif temperatura < 38.0:
                        st.write(f"⚠️ Temperatura: {temperatura}°C (Febrícula)")
                    elif temperatura < 39.0:
                        st.write(f"🔶 Temperatura: {temperatura}°C (Febre)")
                    else:
                        st.write(f"🔴 Temperatura: {temperatura}°C (Febre alta)")
                
                if saturacao_o2:
                    if saturacao_o2 >= 95:
                        st.write(f"✅ Saturação O₂: {saturacao_o2}% (Normal)")
                    elif saturacao_o2 >= 90:
                        st.write(f"⚠️ Saturação O₂: {saturacao_o2}% (Baixa)")
                    else:
                        st.write(f"🔴 Saturação O₂: {saturacao_o2}% (Crítica)")
                
                if freq_respiratoria:
                    if 12 <= freq_respiratoria <= 20:
                        st.write(f"✅ Freq. Respiratória: {freq_respiratoria} resp/min (Normal)")
                    elif freq_respiratoria < 12:
                        st.write(f"⚠️ Freq. Respiratória: {freq_respiratoria} resp/min (Bradipneia)")
                    elif freq_respiratoria <= 24:
                        st.write(f"⚠️ Freq. Respiratória: {freq_respiratoria} resp/min (Taquipneia leve)")
                    else:
                        st.write(f"🔴 Freq. Respiratória: {freq_respiratoria} resp/min (Taquipneia grave)")
            
            with col2:
                st.subheader("Fatores de Risco")
                fatores_risco = []
                
                if idade >= 60:
                    fatores_risco.append("Idade avançada")
                
                if hipertensao:
                    fatores_risco.append("Hipertensão")
                
                if diabetes:
                    fatores_risco.append("Diabetes")
                
                if doenca_cardiaca:
                    fatores_risco.append("Doença cardíaca")
                
                if doenca_respiratoria:
                    fatores_risco.append("Doença respiratória")
                
                if imunossupressao:
                    fatores_risco.append("Imunossupressão")
                
                if fatores_risco:
                    for fator in fatores_risco:
                        st.write(f"⚠️ {fator}")
                else:
                    st.write("✅ Sem fatores de risco identificados")
            
            # Exibindo recomendações
            exibir_recomendacoes(resultado, dados_usuario)
    
    # Aba de Análise de Dados
    with tab2:
        st.header("Análise dos Dados de Treinamento")
        
        # Métricas do modelo
        st.subheader("Métricas do Modelo")
        col1, col2, col3 = st.columns(3)
        col1.metric("Acurácia", f"{accuracy:.2f}")
        col2.metric("Amostras de Treinamento", f"{len(df)}")
        col3.metric("Profundidade da Árvore", "5")
        
        # Gráficos de distribuição
        st.subheader("Distribuição dos Dados")
        fig_diag, fig_idade, fig_sintomas, fig_saturacao = gerar_graficos_distribuicao(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_diag, use_container_width=True)
        with col2:
            st.plotly_chart(fig_idade, use_container_width=True)
        
        st.plotly_chart(fig_sintomas, use_container_width=True)
        st.plotly_chart(fig_saturacao, use_container_width=True)
        
        # Análise de comorbidades
        st.subheader("Prevalência de Comorbidades por Diagnóstico")
        comorbidades = ['hipertensao', 'diabetes', 'doenca_cardiaca', 'doenca_respiratoria', 'imunossupressao']
        
        comorbidades_por_diagnostico = {}
        for diag in range(5):
            df_diag = df[df['diagnostico'] == diag]
            comorbidades_por_diagnostico[class_names[diag]] = [df_diag[comorb].mean() for comorb in comorbidades]
        
        df_comorbidades = pd.DataFrame(comorbidades_por_diagnostico, index=[c.replace('_', ' ').title() for c in comorbidades])
        
        fig_comorbidades = px.imshow(
            df_comorbidades,
            labels=dict(x="Diagnóstico", y="Comorbidade", color="Prevalência"),
            x=df_comorbidades.columns,
            y=df_comorbidades.index,
            color_continuous_scale='Blues',
            title='Prevalência de Comorbidades por Diagnóstico'
        )
        st.plotly_chart(fig_comorbidades, use_container_width=True)
        
        # Relatório de classificação
        st.subheader("Relatório de Classificação")
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.highlight_max(axis=0))
        
        # Matriz de confusão
        st.subheader("Matriz de Confusão")
        X_test = df[feature_names].sample(frac=0.3, random_state=42)
        y_test = df.loc[X_test.index, 'diagnostico']
        y_pred = clf.predict(X_test)
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        fig_conf = px.imshow(
            conf_matrix,
            labels=dict(x="Previsto", y="Real", color="Contagem"),
            x=class_names,
            y=class_names,
            color_continuous_scale='Viridis',
            title='Matriz de Confusão'
        )
        
        # Adicionando anotações com os valores
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[i])):
                fig_conf.add_annotation(
                    x=j, y=i,
                    text=str(conf_matrix[i, j]),
                    showarrow=False,
                    font=dict(color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")
                )
        
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Aba de Visualização da Árvore
    with tab3:
        st.header("Visualização da Árvore de Decisão")
        
        # Visualizando a árvore
        graph = visualizar_arvore(clf, feature_names, class_names)
        st.graphviz_chart(graph)
        
        # Explicação da árvore
        st.subheader("Como interpretar a árvore de decisão:")
        st.write("""
        - Cada nó representa uma decisão baseada em um sintoma ou característica
        - Os valores em cada nó mostram a condição para seguir um caminho (ex: idade <= 65)
        - As cores representam a classe predominante em cada nó
        - Os valores em cada folha mostram a distribuição das classes
        - A árvore segue um caminho do topo (raiz) até as folhas (diagnósticos)
        """)
        
        # Importância das características
        st.subheader("Importância das Características")
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig = px.bar(
            x=[importances[i] for i in indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            title='Importância das Características na Árvore de Decisão',
            labels={'x': 'Importância', 'y': 'Característica'},
            color=[importances[i] for i in indices],
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig)
        
        # Explicação clínica das características mais importantes
        st.subheader("Explicação Clínica das Características Principais")
        
        explicacoes = {
            'saturacao_o2': """
                **Saturação de Oxigênio**: Mede a porcentagem de hemoglobina ligada ao oxigênio no sangue. 
                Valores abaixo de 95% podem indicar problemas respiratórios. Abaixo de 90% é considerado hipoxemia 
                e requer atenção médica imediata. É um indicador crítico em doenças respiratórias como COVID-19.
            """,
            'falta_ar': """
                **Falta de Ar**: Sensação de dificuldade para respirar ou respiração incompleta. 
                É um sintoma cardinal de problemas respiratórios e cardíacos. Na COVID-19, pode indicar 
                comprometimento pulmonar significativo, especialmente quando de início súbito ou progressivo.
            """,
            'perda_olfato': """
                **Perda de Olfato (Anosmia)**: Perda total ou parcial da capacidade de sentir odores. 
                Tornou-se um sintoma distintivo da COVID-19, com alta especificidade. Pode ocorrer mesmo 
                sem congestão nasal e frequentemente precede outros sintomas.
            """,
            'perda_paladar': """
                **Perda de Paladar (Ageusia)**: Perda da capacidade de sentir sabores. 
                Frequentemente ocorre junto com a perda de olfato e é altamente sugestiva de COVID-19. 
                Raramente ocorre em resfriados comuns e gripes.
            """,
            'febre': """
                **Febre**: Elevação da temperatura corporal acima de 37,8°C. 
                É uma resposta do sistema imunológico a infecções. O padrão da febre pode ajudar no diagnóstico: 
                febre alta e súbita é mais comum na gripe, enquanto na COVID-19 pode ser persistente e de intensidade variável.
            """,
            'freq_respiratoria': """
                **Frequência Respiratória**: Número de ciclos respiratórios por minuto. 
                Normal: 12-20 rpm em adultos. Taquipneia (>20 rpm) pode indicar infecção respiratória, 
                ansiedade ou compensação metabólica. Valores >30 rpm geralmente indicam doença grave.
            """,
            'idade': """
                **Idade**: Fator de risco importante para complicações em doenças respiratórias. 
                Pessoas >60 anos têm maior risco de desenvolver formas graves de COVID-19 e gripe, 
                devido à imunossenescência e maior prevalência de comorbidades.
            """,
            'tosse_seca': """
                **Tosse Seca**: Tosse não produtiva, sem catarro. 
                Característica de infecções virais como COVID-19 e gripe. Diferencia-se da tosse produtiva 
                mais comum em bronquite e algumas pneumonias bacterianas.
            """
        }
        
        # Mostrar explicações para as 5 características mais importantes
        for i in indices[:5]:
            feature = feature_names[i]
            if feature in explicacoes:
                st.markdown(f"### {feature.replace('_', ' ').title()}")
                st.markdown(explicacoes[feature])
                st.markdown("---")

if __name__ == "__main__":
    main()