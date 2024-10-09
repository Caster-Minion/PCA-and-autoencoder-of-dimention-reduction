import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import plotly.express as px
import matplotlib.pyplot as plt

import keras
from keras import layers



#selecting CSV

hedged_asset =  pd.read_csv("ENI.MI.csv")



#-----Data cleaning


def data_mod():#function to modify original data (the first 325 rows where multiplied by 100)

    list = [2, 3, 4, 5, 6, 9, 10, 14, 15]
    for i in list:
       df = pd.read_csv('Carbon_price_daily_complete_{}.csv'.format(i))
       #df1 = df.copy(deep=True)

       for j in range(324):
          df.loc[j, 'Price'] = (df.loc[j, 'Price']) / 100

       #print(df)
       df.to_csv('Carbon_price_daily_complete_modified_{}.csv'.format(i), index=False)



def price_date_selection():

    list = [2,3,4,5,6,9,10,14,15]
    for i in list:
       df = pd.read_csv("Carbon_price_daily_complete_modified_{}.csv".format(i))
       df = df[['Price','Date']]
       df = df.tail(-69)  # rimuove le prime 69 righe che in alcuni casi sono vuote
       df.to_csv('Carbon_price_daily_complete_modified_{}.csv'.format(i), index=False)



def prices_merge(): #seleziona solo le date presenti sia nei CPA che nello storico della stock

   df2 = pd.read_csv('Carbon_price_daily_complete_modified_2.csv')
   df3 = pd.read_csv('Carbon_price_daily_complete_modified_3.csv')
   df4 = pd.read_csv('Carbon_price_daily_complete_modified_4.csv')
   df5 = pd.read_csv('Carbon_price_daily_complete_modified_5.csv')
   df6 = pd.read_csv('Carbon_price_daily_complete_modified_6.csv')
   df7 = pd.read_csv('Carbon_price_daily_complete_modified_9.csv')
   df8 = pd.read_csv('Carbon_price_daily_complete_modified_10.csv')
   df9 = pd.read_csv('Carbon_price_daily_complete_modified_14.csv')
   df10 = pd.read_csv('Carbon_price_daily_complete_modified_15.csv')

   df_date_mach = hedged_asset
   df_date_mach = df_date_mach[['Date']]



   merged_df = pd.concat([df2, df3, df4, df5, df6, df7, df8, df9, df10], axis=1)
   merged_df.columns = ['var1','DATE1', 'var2', 'DATE2', 'var3','DATE3', 'var4', 'DATE4', 'var5', 'DATE5', 'var6', 'DATE6', 'var7', 'DATE8', 'var8', 'DATE9', 'var9', 'Date' ]
   #merged_df = merged_df.tail(-69) rimuove le prime 69 righe che in alcuni casi sono vuote


   merged_df = merged_df.drop(merged_df.columns[[1,3,5,7,9,11,13,15]], axis=1)
   merged_df = pd.merge(merged_df, df_date_mach, on='Date', how='inner')
   #print(merged_df)
   merged_df.to_csv('merged_prices', index=False)



def stock_data_join(): #unisce il CSV dei prezzi con data ad quello della stock da hedgiare

    df1 = pd.read_csv('merged_prices')
    df2 = hedged_asset

    ENI_merged = pd.merge(df1, df2)

    ENI_merged.to_csv('stockCPA_merged', index=False)


#-----Dymention reduction

def PCA_1():  #PCA 1 principal component

    merged_prices = pd.read_csv('merged_prices')
    features = ['var1', 'var2', 'var3', 'var4', 'var5','var6', 'var7', 'var8','var9']

    # Separating out the features
    x = merged_prices.loc[:, features].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=1)
    components = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = components , columns = ['principal component 1'])

    total_var = pca.explained_variance_ratio_.sum() * 100
    #print("% varainza spiegata",total_var)
    loadings = pca.components_.T * np.sqrt( pca.explained_variance_)
    #print("",loadings)
    #print(principalDf)

    principalDf.to_csv('principalDf', index=False)


def PCA_2(): #PCA 2 principal components

    merged_prices = pd.read_csv('merged_prices')
    features = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9']

    # Separating out the features
    x = merged_prices.loc[:, features].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # PCA computation 2
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=components, columns=['principal component 1', 'principal component 2'])

    total_var = pca.explained_variance_ratio_.sum() * 100
    print(total_var)

    # plot PCA nota: loadings sono i coefficenti di correlazione la componente principale e v-esima ed ognuna delle variabili.
    # components invece sono i punteggi delle componenti

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # pca.components_.T restituisce gli autovettori ortonormali delle componenti utilizzate

    fig = px.scatter(components, x=0, y=1)

    for i, feature in enumerate(features):
        fig.add_annotation(
            ax=0, ay=0,
            axref="x", ayref="y",
            x=loadings[i, 0],
            y=loadings[i, 1],
            showarrow=True,
            arrowsize=2,
            arrowhead=2,
            xanchor="right",
            yanchor="top"
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            yshift=5,
        )
    fig.show()
    print("",loadings)

  
    fig = px.scatter_matrix(
        principalComponents,
        dimensions=range(2),
        title=f'Total Explained Variance: {total_var:.2f}%',
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()
    
    ax1 = principalDf.plot.scatter(x='principal component 1',
                          y='principal component 2',
                          c='DarkBlue')
    # Display plot
    plt.show()
 


def autoencoder():

    merged_prices = pd.read_csv('merged_prices')
    merged_prices = merged_prices.drop(merged_prices.columns[[9]], axis=1)
    merged_prices.to_csv('merged_prices_no_date', index=False)

    data = np.loadtxt(open("merged_prices_no_date", "rb"), delimiter=",", skiprows=1)


    encoding_dim = 1
    # This is our input
    input = keras.Input(shape=(9,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='sigmoid')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(9, activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input, decoded)

    encoder = keras.Model(input, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    max = data.max()
    min = data.min()


    data_std = (data - min)/(max - min)


    autoencoder.fit(data_std, data_std,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(data_std, data_std))

    # Use the encoder to get the encoded data
    encoded_data = encoder.predict(data_std)

    # Convert the encoded data to a pandas DataFrame for easier viewing
    encoded_df = pd.DataFrame(encoded_data, columns=['Encoded Feature'])

    # Print the encoded data
    #print("Encoded data:")
    #print(encoded_df)
    encoded_df.to_csv('encoded_data', index=False)


#------Testing


def linear_reg():

    #test per  PCA

    df = hedged_asset
    y = df["Close"].values  # as numpy array

    df = pd.read_csv("principalDf")
    x = df["principal component 1"].values  # as numpy array
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)

    print("using PCA")

    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")

    print(f"R^2: {model.score(x, y)}")


    #test per autoencoder

    df = pd.read_csv("encoded_data")
    x = df["Encoded Feature"].values  # as numpy array
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)

    print("using autoencoder")

    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")

    print(f"R^2: {model.score(x, y)}")



