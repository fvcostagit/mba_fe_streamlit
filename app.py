
# import streamlit as st
# import pandas as pd
# # import matlib as ml
# import seaborn as sns

# st.markdown('FEATURE ENGINEERING!')

# st.text('Comando Text')

# st.markdown(
#             """
#             Testando o Markdown com várias linhas
#             Segunda linha
#             Terceira Linha
#             # Com Hashtag
#             ## Com duas Hashtag
#             ### Com três Hashtag
#             - Topico 1            
#             """
#             )

# st.image("https://static-wp-tor15-prd.torcedores.com/wp-content/uploads/2016/09/8.jpg")

# st.video("https://www.youtube.com/watch?v=7WAWY-DktT0")

# df = pd.read_csv('StudentsPerformance.csv', sep=";")

# st.dataframe(df)

# st.pyplot(sns.pairplot(df))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.markdown(
        """
        # Analise de performance de Estudantes
        """             
        )

def mostra_linhas(df):
    qtd = st.sidebar.slider("Selecione quantas linhas você quer", min_value=1, max_value=len(df),step=1)
    st.write(df.head(qtd).style.format(subset=['math score'], formatter='{:.2f}'))
    st.pyplot(sns.pairplot(df))
    # st.pyplot(sns.heatmap(df['math score','reading score','writing score'].corr(), annot=True, fmt='.2f'))
    # x = df.corr()
    # x = df[['math score','reading score','writing score']].corr()
    # st.table(x)
    
df = pd.read_csv('StudentsPerformance.csv', sep=';')

checkbox = st.sidebar.checkbox("Show table")

if checkbox:

    st.sidebar.markdown('## Filtro de dados')

    gen = list(df['gender'].unique())
    gen.append("All")

    genders = st.sidebar.selectbox("Selecione o gênero", options=gen)

    if genders != "All":

        df_gen = df.query('gender == @genders')
        mostra_linhas(df_gen)
        x = df[['math score','reading score','writing score']].corr()
        st.table(x)
        # st.pyplot(plt.boxplot(x = 'math score'))

    else:

        mostra_linhas(df)
        x = df[['math score','reading score','writing score']].corr()
        st.table(x)
        # st.pyplot(plt.boxplot(data = df))
        st.plotly_chart(plt.boxplot(data = df))

# st.dataframe(df)
# plt.title('Pairplot')
# st.pyplot(sns.pairplot(df))