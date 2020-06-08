
"""
Trabajo Bioseñales , punto 4a) Ejemplo de no desfase despues del filtrado

@author: Paulina Salazar, Laura Berrio
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import pywt



#inicialmente trabajando con una sola señal de soniddo para realizar el ejemplo

filename ='./audio_and_txt_files/130_2b3_Al_mc_AKGC417L.wav'
se, sr = librosa.load(filename) #signal and sampling rate
print(type(se))
print(type(sr))
print(se.shape)
print(sr)

# 2 función para filtrar los ruidos y dejar la señal entre 100 Hz y 1000 Hz
from linearFIR import filter_design
import scipy.signal as signal

def filtrado(se,sr):
    fs = sr;
    #diseño del filtro pasa bajas de 1000Hz y pasa altas de 100 Hz, estos valores
    #debido al sonido de interes de los pulmones 
    order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 1000, revfilt = 0); #filtro pasa bajas
    order, highpass = filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1);# filtro pasa altas
    y_hp = signal.filtfilt(highpass, 1, se);
    y_bp = signal.filtfilt(lowpass, 1, y_hp);
    y_bp = np.asfortranarray(y_bp) # asi y_pb es la señal filtrada con la que se va a trabajar
    print(y_bp.shape) 
    #return y_bp
    #librosa.display.waveplot(y_bp, sr=sr);
    return y_bp

#filtrado(se,sr)




# 3 función de filtro wavelet

#umbral duro
#lamda universal
#funciones que ingresan los datos necesarios para filtrado wavelet, con el objetivo de retirar la señal del corazón

def wthresh(coeff,thr):
    y   = list();
    s = wnoisest(coeff); #se llama la función wnoisest
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));
    return y;
    
def thselect(signal):
    Num_samples = 0;
    for i in range(0,len(signal)):
        Num_samples = Num_samples + signal[i].shape[0];
    
    thr = np.sqrt(2*(np.log(Num_samples)))
    return thr

def wnoisest(coeff):
    stdc = np.zeros((len(coeff),1));
    for i in range(1,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;
    return stdc;

#funcion punto 3 aplicación filtro wavelet
#aplicación filtro wavelet sobre la señal, se llama funciones anteriores   
LL = int(np.floor(np.log2(se.shape[0])));
coeff = pywt.wavedec( se, 'db6', level=LL );


def f_wavelet(se,coeff):
    
    
    thr = thselect(coeff); #llama thselect
    coeff_t = wthresh(coeff,thr); #llama la función wthresh
    
    x_rec = pywt.waverec( coeff_t, 'db6');
    
    x_rec = x_rec[0:se.shape[0]];
    
    #librosa.display.waveplot(se[:], sr=sr,label='Original');
    #debe restarse a la señal original para no eliminar los sonidos pulmonares sino solo
    #ruido cardiaco
    x_filt = np.squeeze(se - x_rec);
    #librosa.display.waveplot(x_filt[:], sr=sr,label='Original- Umbralizada');
    
    #librosa.display.waveplot(x_rec[:], sr=sr,label='Umbralizada por Wavelet');
    
    #plt.legend()
    return x_filt;
    
 
#f_wavelet(se,coeff) 

# 4 
#función que permite el preprocesamiento de la señal usando los filtros previos
sig = filtrado(se,sr)  
LL = int(np.floor(np.log2(sig.shape[0])));
coeff1 = pywt.wavedec( sig, 'db6', level=LL );
def preprocesar(se, coeff1):
    
    
    #llamado función de filtrado pasa bajas y pasa altas
    senal_filtro1= filtrado(se,sr)
    #llamado función de filtro wavelet
    se_procesada= f_wavelet(senal_filtro1,coeff1);
    print(se_procesada)
    #librosa.display.waveplot(se[:], sr=sr,label='Original'); # se grafica la señal original
    #plt.figure()
    #plt.title('Señal filtrada Vs Señal Procesada');
    #librosa.display.waveplot(senal_filtro1[:], sr=sr,label='Filtrada'); #señal filtrada por pasabajas y pasaltas
    #librosa.display.waveplot(se_procesada[:], sr=sr,label='Procesada'); # señal procesada finalmente
    #plt.legend()
    return se_procesada

#preprocesar(se,coeff1)    
    
    
 
#punto 4 a) mostrar y verificar que efectivamente despues del procesamiento no hay desfases ni se perdieron 
#los inicios o finales de los ciclos    
# función unicamente explicativa para mostrar ejemplos de una señal y que no hay desfases del preprocesamiento
def desfase(senal_aensayo):
    
    ciclos = np.loadtxt('./audio_and_txt_files/130_2b3_Al_mc_AKGC417L.txt')
    print(ciclos)
    ciclo_1= ciclos[0] #  primer ciclo respiratorio
    print(' primer ciclo respiratorio:',ciclo_1[0],ciclo_1[1])
    
    
    plt.title("Primer ciclo respiratorio")
    
    m1 = np.int (0.689*sr)
    m2 = np.int(1.732*sr)
    print('Primer ciclo en muestras:',+ m1,m2)
    
    librosa.display.waveplot(se[m1:m2], sr=sr,label='Original');
    librosa.display.waveplot(senal_aensayo[m1:m2], sr=sr,label='Filtrada Procesada');
    plt.legend()
    
    #ciclo_ultimo
    #
    m3=np.int(16.946*sr)
    m4 = np.int (19.156*sr)
    print('Último ciclo en muestras:',+ m3,m4)
    
    ciclo_u = ciclos[8]
    print(' último ciclo respiratorio:',ciclo_u[0],ciclo_u[1])
    plt.show()
    plt.figure()
    plt.title("Ultimo ciclo respiratorio")
    librosa.display.waveplot(se[m3:m4], sr=sr,label='Original');
    librosa.display.waveplot(senal_aensayo[m3:m4], sr=sr,label='Filtrada Procesada');
    plt.legend()
    plt.show()

       
sig = filtrado(se,sr)  
LL = int(np.floor(np.log2(sig.shape[0])));
coeff1 = pywt.wavedec( sig, 'db6', level=LL );
  
senal_aensayo= preprocesar(se,coeff1) 
desfase(senal_aensayo)        
        