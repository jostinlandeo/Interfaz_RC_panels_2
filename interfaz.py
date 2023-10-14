import streamlit as st
import pandas as pd
from pickle import load
import pickle
import numpy as np
import math
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler

#import joblib
#from config.definitions import ROOT_DIR

#Indication to run interfaz in a localhost
#1 open the terminal cmd and change root direcory to file directory
#2 write this: streamlit run interfaz.py --server.port=9876


PROJECT_ROOT_DIR = "."
OutModel_PATH = os.path.join(PROJECT_ROOT_DIR, "model_output")

#Recovering regression model
model_file = os.path.join(OutModel_PATH, "final_model_RC_membrane_reg.pkl") #archivo
with open(model_file, 'rb') as f:
    loaded_model_reg = pickle.load(f)

#Recovering classifcation model
model_file = os.path.join(OutModel_PATH, "final_model_RC_membrane_class.pkl")#archivo
with open(model_file, 'rb') as f:
    loaded_model_cla = pickle.load(f)


st.title('Shear strength and Failure mode of Reinforced Concrete Membranes Predicted by ML Methods')
st.subheader('Dimensional Parameters')
st.sidebar.header('User Input Parameters')

PROJECT_ROOT_DIR_Fig = "."
OutModel_PATH2 = os.path.join(PROJECT_ROOT_DIR_Fig, "figures_interfaz")

image = Image.open(os.path.join(OutModel_PATH2,'panel.png'))
st.image(image)

st.subheader('Nomenclature')
st.write('Lx: Length of the square panels ')
st.write('t: Thickness of the panels  ')
st.write('rho_sx: Longitudinal reinforcement ratio ')
st.write('rho_sy: Transverse reinforcement ratio ')
st.write('fc: Compressive strength of the concrete ')
st.write('Fyx: Yield strength of the longitudinal steel ')
st.write('Fyy: Yield strength of the transverse steel  ')
st.write('Vx: Ratio of applied shear load or shear stress (in-plane shear load or stress')
st.write('Fx: Ratio of applied normal load or normal stress in the x axis ')
st.write('Fy: Ratio of applied normal load or normal stress in the y axis ')


def user_input_features():
    Lx = st.sidebar.slider('Lx (mm)', min_value=600, max_value=3000, step=10)
    t = st.sidebar.slider('t (mm)', min_value=50, max_value=290, step=10)
    rho_sx = st.sidebar.slider('rho_sx (%)', min_value=0.00, max_value=6.50, step=0.01)
    rho_sy = st.sidebar.slider('rho_sy (%)', min_value=0.00, max_value=5.50, step=0.01) 
    fc = st.sidebar.slider('fc (MPa)', min_value=10.0, max_value=110.0, step=0.1)
    fyx = st.sidebar.slider('Fyx (MPa)', min_value=0, max_value=1200, step=1)
    fyy = st.sidebar.slider('Fyy (MPa)', min_value=0, max_value=1200, step=1) 
    Vx = st.sidebar.slider('Vx (Load_ratio)', min_value=0.00, max_value=1.00, step=0.01) 
    Fx = st.sidebar.slider('Fx (Load_ratio)', min_value=-3.00, max_value=7.00, step=0.01) 
    Fy = st.sidebar.slider('Fy (Load_ratio)', min_value=-1.00, max_value=1.00, step=0.01) 

    data = {'Lx (mm)': Lx,
            't (mm)': t,
            'rho_sx (%)': rho_sx,
            'rho_sy (%)': rho_sy,
            'fc (MPa)': fc,
            'Fyx (MPa)': fyx,
            'Fyy (MPa)': fyy,  
            'Vx (Load_ratio)': Vx, 
            'Fx (Load_ratio)': Fx,
            'Fy (Load_ratio)': Fy}                       
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

Lx1=df['Lx (mm)'].values.item()
t1=df['t (mm)'].values.item()
rho_sx1=df['rho_sx (%)'].values.item()
rho_sy1=df['rho_sy (%)'].values.item()
fc1=df['fc (MPa)'].values.item()
fyx1=df['Fyx (MPa)'].values.item()
fyy1=df['Fyy (MPa)'].values.item()
Vx1=df['Vx (Load_ratio)'].values.item()
Fx1=df['Fx (Load_ratio)'].values.item()
Fy1=df['Fy (Load_ratio)'].values.item()

#var_names = ["psy*Fyy ","F'c","psx*Fyx","Vxy","Fx","Fy","Lx*t"] #data_mod["Lx*t"]= data_mod["Lx"]*data_mod["tag_thickness"]/(10**6)
#var_names = ["psy*Fyy ","F'c","psx*Fyx","Vxy","Fx","Fy","Lx/t"]

rho_sy_fyy = (rho_sy1/100)*fyy1
fc2 = fc1
rho_sx_fyx = (rho_sx1/100)*fyx1
Vx2 = Vx1
Fx2 = Fx1
Fy2 = Fy1
Lx_t = Lx1*t1/(10**6)
Lx__t = Lx1/t1


user_input={'Lx (mm)': "{:.0f}".format(Lx1),
            't (mm)': "{:.0f}".format(t1),
            'rho_sx (%)': "{:.2f}".format(rho_sx1),
            'rho_sy (%)': "{:.2f}".format(rho_sy1),
            'fc (MPa)': "{:.1f}".format(fc1),
            'Fyx (MPa)': "{:.0f}".format(fyx1),
            'Fyy (MPa)': "{:.0f}".format(fyy1),
            'Vx (Load_ratio)': "{:.2f}".format(Vx1),
            'Fx (Load_ratio)': "{:.2f}".format(Fx1),
            'Fy (Load_ratio)': "{:.2f}".format(Fy1)}

user_input_df=pd.DataFrame(user_input, index=[0])
st.subheader('User Input Parameters')
#st.dataframe(user_input_df, 900, 1500)
st.table(user_input_df)
#st.write(user_input_df)
#
#Parameters for regression
calculated_param={'rho_sy_Fyy': "{:.2f}".format(rho_sy_fyy),
                  'fc (MPa)': "{:.2f}".format(fc2),
                  'rho_sx_Fyx': "{:.2f}".format(rho_sx_fyx),
                  'Vx': "{:.2f}".format(Vx2),
                  'Fx': "{:.2f}".format(Fx2),
                  'Fy': "{:.2f}".format(Fy2),
                  'Lx_t (m2)': "{:.2f}".format(Lx_t)}
calculated_param_df=pd.DataFrame(calculated_param, index=[0])
st.subheader('Model Input Parameters for Shear Strength')
st.table(calculated_param_df)
#
#Parameters for clasification
calculated_param_cla={'rho_sy_Fyy': "{:.2f}".format(rho_sy_fyy),
                  'fc (MPa)': "{:.2f}".format(fc2),
                  'rho_sx_Fyx': "{:.2f}".format(rho_sx_fyx),
                  'Vx': "{:.2f}".format(Vx2),
                  'Fx': "{:.2f}".format(Fx2),
                  'Fy': "{:.2f}".format(Fy2),
                  'Lx/t': "{:.2f}".format(Lx__t)}
calculated_param_df_cla=pd.DataFrame(calculated_param_cla, index=[0])
st.subheader('Model Input Parameters for Failure Mode')
st.table(calculated_param_df_cla)
#

var_names_reg = ['rho_sy_Fyy', 'fc', 'rho_sx_Fyx', "Vx", 'Fx', 'Fy', 'Lx*t']
var_names_cla = ['rho_sy_Fyy', 'fc', 'rho_sx_Fyx', "Vx", 'Fx', 'Fy', 'Lx/t']

#Definning input to model predictions
reg=np.array([[rho_sy_fyy,fc2,rho_sx_fyx,Vx2,Fx2,Fy2,Lx_t]])
cla=np.array([[rho_sy_fyy,fc2,rho_sx_fyx,Vx2,Fx2,Fy2,Lx__t]])

# Escalando los inputs (forma correcta para los inputs)
s_reg = np.load('std_scale_reg_panel.npy')#archivos
m_reg = np.load('mean_scale_reg_panel.npy')#archivos

s_cla = np.load('std_scale_cla_panel.npy')#archivos
m_cla = np.load('mean_scale_cla_panel.npy')#archivos

reg_sca=pd.DataFrame((reg-m_reg)/s_reg,index=[0])
cla_sca=pd.DataFrame((cla-m_cla)/s_cla,index=[0])

##Regression
Load_pred_reg=loaded_model_reg.predict(reg_sca).item()
V_test=np.exp(Load_pred_reg)*reg[0,1]

##Classification  
Load_pred_cla=loaded_model_cla.predict(cla_sca).item()
resultado=Load_pred_cla
res=str()
if resultado==0:
   res="Concrete failure"
if resultado==1:
   res="Other type of failure"
if resultado==2:
   res="Reinforcement failure"


st.subheader('XGBoost Model Predictions')
w_cr_results={'Shear Strength (MPa)':"{:.2f}".format(V_test),
               'Failure Mode':format(res)}
w_cr_results_df=pd.DataFrame(w_cr_results, index=[0])
st.table(w_cr_results_df)

st.subheader('Copyright')
st.write('Code by: Bedriñana, L. , Sucasaca, J. and Landeo, J.')
st.write('Universidad de Ingeniería y Tecnología - Department of Civil Engeneering')
st.write('How to cited this: ')