import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from wsgiref.util import request_uri
from streamlit_lottie import st_lottie
from unittest import result
from PIL import Image
from math import sqrt
from datetime import datetime, date

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def explore(df):
    # convert dtype object to datetime
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i] = pd.to_datetime(df[i], format='%d/%m/%Y')
            df.sort_values([i], inplace=True, ignore_index=True)

            # Menampilkan Dataset
            st.subheader('Dataset')
            st.write(df)

            with st.expander("Visuaisasi Dataset"):
                # Pilih Kolom yang ingin di Prediksi
                
                option = st.sidebar.selectbox(
                'Choose columns',
                (df.columns.values))
                # submit = st.sidebar.button("View Data", option)
                # if submit:
                
                rolling_window = 52
                f, ax = plt.subplots(figsize=(15, 5))

                result = adfuller(df[option])
                significance_level = 0.05
                adf_stat = result[0]
                p_val = result[1]
                crit_val_1 = result[4]['1%']
                crit_val_5 = result[4]['5%']
                crit_val_10 = result[4]['10%']
                if(p_val < significance_level) & ((adf_stat < crit_val_1)):
                    linecolor = 'forestgreen'
                elif(p_val < significance_level) & (adf_stat < crit_val_5):
                    linecolor = 'orange'
                elif(p_val < significance_level) & (adf_stat < crit_val_10):
                    linecolor = 'red'
                elif(p_val > significance_level):
                    linecolor = 'purple'
                else:
                    linecolor = 'magenta'
                sns.lineplot(x=df[i], y=df[option], color=linecolor)
                # check rolling mean and rolling std
                sns.lineplot(x=df[i], y=df[option].rolling(rolling_window).mean(), color='yellow', label='rolling mean')
                sns.lineplot(x=df[i], y=df[option].rolling(rolling_window).std(), color='black', label='rolling std')
                ax.set_title(f'{option}\nADF Statistic {adf_stat:0.3f}, p_value {p_val:0.3f}\nCritical Values 1% {crit_val_1:0.3f}, 5% {crit_val_5:0.3f}, 10% {crit_val_10:0.3f}', fontsize=14)
                ax.set_ylabel(option, fontsize=14)
                ax.set_xlabel(i, fontsize=14)
                # ax.set_xlim([date(2011, 10, 26), date(2020, 10, 26)])
            
                st.subheader('Augmented Dickeyy-fuller (ADF) :')
                st.write("""
                1. Jika p_value dibawah 0.05, dan adf dibawah critical value 1%, linechart berwarna hijau
                2. Jika p_value dibawah 0.05, dan adf dibawah critical value 5%, linechart berwarna jingga
                3. Jika p_value dibawah 0.05, dan adf dibawah critical value 10%, linechart berwarna merah
                """)
                st.write(f)

                # MENGECEK MISSING VALUE
                g, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,5))

                sns.heatmap(df.T.isnull(), cmap='Blues')
                ax.set_title('Missing Values', fontsize=16)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(14)

                st.subheader('Missing Values')
                st.write(g)


            # FORCASTING
            form = st.sidebar.form(key='my-form')
            period_input = form.number_input('Enter day to predict', max_value=1825)
            submit = form.form_submit_button('Submit')
            if submit:
                with st.spinner("Loading...."):

                    # Mengubah dtype datetime menjadi index
                    df.set_index(i, inplace=True)
                    ts = df[option]

                    with st.expander("Modelling"):
                        # Training Data
                        st.subheader("Training Data")
                        train_size = 0.7
                        split_idx = round(len(ts)* train_size)
                        split_idx

                        # Split
                        train = ts.iloc[:split_idx]
                        test = ts.iloc[split_idx:]

                        do,ax= plt.subplots(figsize=(12,4))
                        ax.plot(train, label='Train')
                        ax.plot(test, label='Test')
                        ax.legend(bbox_to_anchor=[1,1])
                        st.write(do)

                        # Modelling
                        auto_model = pm.auto_arima(train, start_p=0, start_q=0)
                        st.write('Best p,d,q order: {} x {}'.format(auto_model.order, auto_model.seasonal_order))

                        pred_model = auto_model.predict(n_periods=len(test), typ='levels')
                        score_mae = mean_absolute_error(test, pred_model)
                        score_rmse = sqrt(mean_squared_error(pred_model, test))
                        sarimax_prediksi = SARIMAX(ts, order=auto_model.order, seasonal_order=auto_model.seasonal_order, trend='t')
                        results_SARIMA_t = sarimax_prediksi.fit(disp=-1)
                        predictions_SARIMA_diff_t = pd.Series(results_SARIMA_t.fittedvalues,copy=True)
                        st.subheader("Modelling")
                        di = plt.figure(figsize=(10,5))
                        plt.plot(ts)             
                        plt.plot(predictions_SARIMA_diff_t, color='red')               #fitting dengan data
                        st.pyplot(di)

                        st.write(auto_model.summary())

                    # HASIL PREDIKSI
                    st.subheader("Hasil Prediksi")
                    c1,c2 = st.columns(2)
                    with c1:
                        prediksi = results_SARIMA_t.predict(start=len(ts),end=len(ts)+period_input)
                        # print(prediksi)
                        prediksi_df=pd.DataFrame(prediksi)
                        st.write(prediksi_df)
                    with c2:
                        fg = plt.figure(figsize=(10,5))
                        plt.plot(ts, label='Data sebelumnya')
                        plt.plot(prediksi, label='Data prediksi')
                        plt.legend(loc='best') 
                        st.pyplot(fg)

                st.success('SELESAI')
                st.markdown(
                    """
                    <div class="text-center p-5">
                        <h3>--- THANK YOU ---</h3>
                    </div>
                    """, unsafe_allow_html=True)



def get_df(file):
    # get extension and read file
    extension = file.name.split('.')[1]
    if extension.upper() == 'CSV':
        df = pd.read_csv(file)
    elif extension.upper() == 'XLSX':
        df = pd.read_excel(file, engine='openpyxl')
    elif extension.upper() == 'PICKLE':
        df = pd.read_pickle(file)
    return df

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    # with open('style.css') as f:
    #     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    icon = Image.open('static/img/logo.png')
    st.set_page_config(page_title="APES", page_icon=icon, layout="wide")
    st.markdown("""<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">""",unsafe_allow_html=True)
    local_css("style/style.css")

    with st.container():
        leftCol, centerCol, rightCol = st.columns(3)
        # Logo
        with leftCol:
            image = Image.open('static/img/logo-web.png')
            st.write("")
            st.image(image, width=120)
        with centerCol:
            st.empty()
        
        # Memilih page
        with rightCol:     
            menu = ['Home', 'Forecasting']
            navbar = st.selectbox("Halaman",menu)

        # Page Home
        if navbar == "Home":
            with st.container():
                left_col, right_col = st.columns(2)
                with left_col:
                    st.header('##')
                    st.markdown(
                    """
                    <div class="container">
                        <div class="row pt-5">
                            <div class="col-lg-12 d-flex flex-column justify-content-center pt-4 pt-lg-0 order-2 order-lg-1" data-aos="fade-up" data-aos-delay="200">
                                <h1><strong>Artificial Predictive Energy Supply</strong></h1>
                                <h4>Prediksikan dataset time series menggunakan model AUTO ARIMA</h4>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with right_col:
                    lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json")
                    st_lottie(lottie_coding, height=500, key="coding") 
                    
            with st.container():
                st.markdown("<h1 style='text-align: center;'><strong>APES</strong></h1>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="container">
                        <div class="text-center">
                            <h5>Artificial Predictive Energy Supply (APES) adalah solusi berbasis AI digital yang dapat digunakan oleh perusahaan pembangkit listrik untuk mengantisipasi hasil produksi secara tepat berdasarkan unit waktu.Produk ini memiliki alat visualisasi data yang memudahkan pengolahan data bagi konsumen. Selanjutnya, produk mengungkapkan akurasi proses prediksi, dan data prediksi yang ditampilkan oleh produk dapat dengan mudah diunduh oleh pengguna untuk administrasi lebih lanjut.</h5>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.header('##')
            with st.container():
                st.markdown("<h1 style='text-align: center;'><strong>Manfaat</strong></h1>", unsafe_allow_html=True)
                st.header('##')
                colom1, colom2, colom3 = st.columns(3)
                with colom1:
                    lottie_pengawasan = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_49rdyysj.json")
                    st_lottie(lottie_pengawasan, height=220)
                    st.write('Membantu dalam proses pengawasan produksi energi listrik')
                with colom2:
                    lottie_rencana = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ca8zbrdk.json")
                    st_lottie(lottie_rencana, height=220)
                    st.write('Membantu dalam perencanaan perawatan peralatan produksi')
                with colom3:
                    lottie_conside = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_jx5xafds.json")
                    st_lottie(lottie_conside, height=220)
                    st.write('Menjadi bahan pertimbangan dalam proses suatu perluasan dan pengembangan produksi.')


        # Halaman Forecasting
        if navbar == "Forecasting":
            with st.container():
                with st.sidebar:
                    lottie_sidebar = load_lottieurl("https://assets7.lottiefiles.com/private_files/lf30_iifp5sng.json")
                    st_lottie(lottie_sidebar, key="sidebar")
                file = st.sidebar.file_uploader('Upload file :', type=['csv', 'xlsx'])
                if not file:
                    st.sidebar.write("Upload a .csv or .xlsx file to get started")
                    return
                df = get_df(file)
                explore(df)


    



     # Memilih kolom yang akan di forecast
    

    st.markdown("""
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)

main()

  