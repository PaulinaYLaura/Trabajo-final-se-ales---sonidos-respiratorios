
"""
Trabajo final Bioseñales y sistemas

@author: Laura Berrio y Maria Paulina Salazar 
"""
# 2 función para filtrar los ruidos y dejar la señal entre 100 Hz y 1000 Hz

from linearFIR import filter_design
import scipy.signal as signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt  
import pandas as pd
  
import glob
import seaborn as sns #Graficacion

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


#3 función de filtro wavelet

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


 # 4 
 #función que permite el preprocesamiento de la señal usando los filtros previos

def preprocesar(se, coeff1):
    
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

#Procesamiento y extracción de características de la señal
#6. Crear una función que reciba un ciclo respiratorio y extraiga los índices explicados
#en el documento : https://munin.uit.no/handle/10037/11260        
         
#indices a extraer rango, varianza,promedios moviles,promedio espectral   
         
def varianza(ciclo):
    varianza = np.var(ciclo)
    return varianza

def rango(ciclo):
    rango = np.abs(np.max(ciclo) - np.min(ciclo))
    return rango
def promedios_moviles(ciclo):
    muestras = 800
    corrido = 100 #corrimiento
    recorrido = np.arange(0, len(ciclo) - muestras, corrido)
    promedio_moviles = []
    for i in recorrido:
        promedio_moviles.append(np.mean([ciclo[i:i+muestras]]))
    promedio_moviles.append(np.mean([ciclo[recorrido[len(recorrido) - 1]: ]]))
    suma_promedios= np.mean(promedio_moviles)
    return suma_promedios

def indice_espectro(ciclo):
    f, Pxx = signal.periodogram(ciclo)
    espectro = np.mean(Pxx)
    return espectro
    
    
    
         
#funcion que reune las funciones para el procesamiento de cada indice
def extraer_indices(ciclo):
    in_varianza = varianza(ciclo)
    in_rango = rango(ciclo)
    in_promedios_moviles = promedios_moviles(ciclo)
    in_espectro = indice_espectro(ciclo)
    return in_varianza, in_rango, in_promedios_moviles, in_espectro


#Crear una rutina que aplique sobre todos los archivos de la base de datos las rutinas
#de preprocesamiento y extracción de características y guarde la información en un
#dataframe donde se pueda discriminar información relacionada con ciclos normales,
#ciclos con crepitaciones y ciclos con sibilancias


ruta_archivos = './audio_and_txt_files/'

#listando los archivos en el directorio
archivos_wav = glob.glob(ruta_archivos + '/*.wav')
archivos_textos_tiempo = glob.glob(ruta_archivos + '/*.txt')

def subir_archivos(texto,wav):
    se, sr = librosa.load(wav)
    datos = np.loadtxt(texto)
    return datos,se,sr
    
#listas para dataframe general
lista_audio_procesado=[]
lista_estados = []
lista_ciclos_procesados = []
lista_varianza = []
lista_rango = []
lista_promedio_movil_f = []
lista_espectro = []



#listas para el dataframe sanos

lista_varianza_sanos= []
lista_rango_sanos= []
lista_promedio_movil_f_sanos= []
lista_espectro_sanos= [] 

#listas para el dataframe crepitancia

lista_varianza_crepitancia= []
lista_rango_crepitancia= []
lista_promedio_movil_f_crepitancia= []
lista_espectro_crepitancia= [] 

#lista para el dataframe sibilancia

lista_varianza_sibilancia= []
lista_rango_sibilancia= []
lista_promedio_movil_f_sibilancia= []
lista_espectro_sibilancia= [] 


#lista para el dataframe ambas patologías

lista_varianza_ambas= []
lista_rango_ambas= []
lista_promedio_movil_f_ambas= []
lista_espectro_ambas= []

#for i in range(0,len(archivos)):
for i in range(0,50):
    tiempo, se, sr = subir_archivos(archivos_textos_tiempo[i],archivos_wav[i]) #signal and sampling rate
    sig = filtrado(se,sr)  
    LL = int(np.floor(np.log2(sig.shape[0])));
    coeff1 = pywt.wavedec( sig, 'db6', level=LL );
    audio_procesado = preprocesar(se,coeff1) 
    lista_audio_procesado.append(audio_procesado)
    tiempo_ciclo = tiempo.shape[0]
    e=0
     #rutina que recibe la ruta de un archivo de audio(Los audios ya filtrados) y la ruta del archivo de
      #anotaciones y extraiga del archivo de audio los ciclos respiratorios con su respectiva
       #anotación de estertores y sibilancias 
    for e in range(0,tiempo_ciclo):
        m1 = np.int((tiempo[e,0])*sr)
        m2 = np.int((tiempo[e,1])*sr)
        extraccion = audio_procesado[m1:m2]
        lista_ciclos_procesados.append(extraccion)
        toma_ciclo = lista_ciclos_procesados[i]
        #llamado a la función de indices para procesar cada ciclo
        in_varianza, in_rango, in_promedios_moviles, in_espectro = extraer_indices(toma_ciclo)
        lista_varianza.append(in_varianza)
        lista_rango.append(in_rango)
        lista_promedio_movil_f.append(in_promedios_moviles)
        lista_espectro.append(in_espectro)
    j=0
    for j in range(0,tiempo_ciclo):
        if tiempo[j,2] == 0 and tiempo[j,3]== 0 :
            lista_estados.append('0_sano')
            in_varianza, in_rango, in_promedios_moviles, in_espectro = extraer_indices(toma_ciclo)
            lista_varianza_sanos.append(in_varianza)
            lista_rango_sanos.append(in_rango)
            lista_promedio_movil_f_sanos.append(in_promedios_moviles)
            lista_espectro_sanos.append(in_espectro)
            
        elif tiempo[j,2] == 1 and tiempo[j,3]==0 :
            lista_estados.append('1_crepitancia')
            in_varianza, in_rango, in_promedios_moviles, in_espectro = extraer_indices(toma_ciclo)
            lista_varianza_crepitancia.append(in_varianza)
            lista_rango_crepitancia.append(in_rango)
            lista_promedio_movil_f_crepitancia.append(in_promedios_moviles)
            lista_espectro_crepitancia.append(in_espectro)
            
            
        elif tiempo[j,2] == 0 and tiempo[j,3]== 1 :
            lista_estados.append('2_sibilancia')
            in_varianza, in_rango, in_promedios_moviles, in_espectro = extraer_indices(toma_ciclo)
            lista_varianza_sibilancia.append(in_varianza)
            lista_rango_sibilancia.append(in_rango)
            lista_promedio_movil_f_sibilancia.append(in_promedios_moviles)
            lista_espectro_sibilancia.append(in_espectro)
            
            
        elif tiempo[j,2] == 1 and tiempo[j,3]== 1:
            lista_estados.append('3_ambas_patologías')
            in_varianza, in_rango, in_promedios_moviles, in_espectro = extraer_indices(toma_ciclo)
            lista_varianza_ambas.append(in_varianza)
            lista_rango_ambas.append(in_rango)
            lista_promedio_movil_f_ambas.append(in_promedios_moviles)
            lista_espectro_ambas.append(in_espectro)
        

        
df = pd.DataFrame({'Ciclos':lista_ciclos_procesados,'Rango':lista_rango, 'Varianza':lista_varianza, 'Promedio movil':lista_promedio_movil_f,'Promedio espectro':lista_espectro, 'Estado':lista_estados})       
    
    
#para lograr hacer un analisis estadistico de cada estado de paciente separaremos sus indices en data frames

df_sanos = pd.DataFrame({'Rango':lista_rango_sanos, 'Varianza':lista_varianza_sanos, 'Promedio movil':lista_promedio_movil_f_sanos,'Promedio espectro':lista_espectro_sanos})       
df_crepi = pd.DataFrame({'Rango':lista_rango_crepitancia, 'Varianza':lista_varianza_crepitancia, 'Promedio movil':lista_promedio_movil_f_crepitancia,'Promedio espectro':lista_espectro_crepitancia})       
df_sibi= pd.DataFrame({'Rango':lista_rango_sibilancia, 'Varianza':lista_varianza_sibilancia, 'Promedio movil':lista_promedio_movil_f_sibilancia,'Promedio espectro':lista_espectro_sibilancia})       
df_ambas= pd.DataFrame({'Rango':lista_rango_ambas, 'Varianza':lista_varianza_ambas, 'Promedio movil':lista_promedio_movil_f_ambas,'Promedio espectro':lista_espectro_ambas})       
    

#realizando estadistica a cada condicion 

df_sanos.describe() #saca los valores promedio, la desviación estándar, los cuartiles y los valores máximos y mínimos
df_crepi.describe()
df_sibi.describe()
df_ambas.describe()

#realizando estadistica para el data frame general ,comparaando los estados en relacion con cada indice
#Graficos de dispersión

plt.scatter(df['Estado'],df['Rango'])
plt.ylabel('Rango')
plt.xlabel('Estado')
plt.show()


plt.figure()
plt.scatter(df['Estado'],df['Varianza'])
plt.ylabel('Varianza')
plt.xlabel('Estado')
plt.show()

plt.figure()
plt.scatter(df['Estado'],df['Promedio movil'])
plt.ylabel('Promedio movil')
plt.xlabel('Estado')
plt.show()

plt.figure()
plt.scatter(df['Estado'],df['Promedio espectro'])
plt.ylabel('Promedio espectro')
plt.xlabel('Estado')
plt.show()

#realizando estadistica para el data frame general ,comparaando los estados en relacion con cada indice
#Graficos de dispersión

plt.figure()
sns.boxplot(x='Estado',y='Rango',data=df)#se agrupa en torno a la variable
plt.title('Grafico Box plot-Rango')
plt.plot()
plt.show()


plt.figure()
sns.boxplot(x='Estado',y='Varianza',data=df)#se agrupa en torno a la variable
plt.title('Grafico Box plot-Varianza')
plt.plot()
plt.show()


plt.figure()
sns.boxplot(x='Estado',y='Promedio movil',data=df)#se agrupa en torno a la variable
plt.title('Grafico Box plot-Promedio movil')
plt.plot()
plt.show()

plt.figure()
sns.boxplot(x='Estado',y='Promedio espectro',data=df)#se agrupa en torno a la variable
plt.title('Grafico Box plot-Promedio espectro')
plt.plot()
plt.show()

#Histogramas de cada indice para los pacientes sanos
plt.figure()
count,bin_edges = np.histogram(df_sanos['Rango'])
df_sanos['Rango'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma sanos Rango')
plt.xlabel('sanos')
plt.ylabel('Rango')
plt.grid()
plt.show()


plt.figure()
count,bin_edges = np.histogram(df_sanos['Varianza'])
df_sanos['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma sanos Varianza')
plt.xlabel('sanos')
plt.ylabel('Varianza')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(df_sanos['Promedio movil'])
df_sanos['Promedio movil'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma sanos Promedio moviles')
plt.xlabel('sanos')
plt.ylabel('Rango')
plt.grid()
plt.show()


plt.figure()
count,bin_edges = np.histogram(df_sanos['Promedio espectro'])
df_sanos['Promedio espectro'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma sanos Promedio espectro')
plt.xlabel('sanos')
plt.ylabel('Promedio espectro')
plt.grid()
plt.show()

#Histogramas de cada indice para los pacientes crepitancias
plt.figure()
count,bin_edges = np.histogram(df_crepi['Rango'])
df_sanos['Rango'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Crepitancias Rango')
plt.xlabel('Crepitancias')
plt.ylabel('Rango')
plt.grid()
plt.show()


plt.figure()
count,bin_edges = np.histogram(df_crepi['Varianza'])
df_sanos['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Crepitancias Varianza')
plt.xlabel('Crepitancias')
plt.ylabel('Varianza')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(df_crepi['Promedio movil'])
df_sanos['Promedio movil'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Crepitancias Promedio movil')
plt.xlabel('Crepitancias')
plt.ylabel('Rango')
plt.grid()
plt.show()


plt.figure()
count,bin_edges = np.histogram(df_crepi['Promedio espectro'])
df_sanos['Promedio espectro'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Crepitancias Promedio espectro')
plt.xlabel('Crepitancias')
plt.ylabel('Promedio espectro')
plt.grid()
plt.show()

#Histogramas de cada indice para los pacientes sibilancias
plt.figure()
count,bin_edges = np.histogram(df_sibi['Rango'])
df_sanos['Rango'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Sibilancias Rango')
plt.xlabel('Sibilancias')
plt.ylabel('Rango')
plt.grid()
plt.show()


plt.figure()
count,bin_edges = np.histogram(df_sibi['Varianza'])
df_sanos['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Sibilancias Varianza')
plt.xlabel('Sibilancias')
plt.ylabel('Varianza')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(df_sibi['Promedio movil'])
df_sanos['Promedio movil'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Sibilancias Promedio movil')
plt.xlabel('Sibilancias')
plt.ylabel('Promedio movil')
plt.grid()
plt.show()


plt.figure()
count,bin_edges = np.histogram(df_sibi['Promedio espectro'])
df_sanos['Promedio espectro'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Sibilancias Promedio espectro')
plt.xlabel('sibilancias')
plt.ylabel('Promedio espectro')
plt.grid()
plt.show()
#Histogramas de cada indice para los pacientes con ambas patologias

plt.figure()
count,bin_edges = np.histogram(df_ambas['Rango'])
df_sanos['Rango'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Ambas Patologias Rango')
plt.xlabel('Ambas')
plt.ylabel('Rango')
plt.grid()
plt.show()


plt.figure()
count,bin_edges = np.histogram(df_ambas['Varianza'])
df_sanos['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Ambas Patologias Varianza')
plt.xlabel('ambas')
plt.ylabel('Varianza')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(df_ambas['Promedio movil'])
df_sanos['Promedio movil'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Ambas Patologias Promedio movil')
plt.xlabel('ambas')
plt.ylabel('Rango')
plt.grid()
plt.show()


plt.figure()
count,bin_edges = np.histogram(df_ambas['Promedio espectro'])
df_sanos['Promedio espectro'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma Ambas Patologias Promedio espectro')
plt.xlabel('ambas')
plt.ylabel('Promedio espectro')
plt.grid()
plt.show()

#correlacion de indices para el dataframe
plt.title('Grafico de correlacion entre los indices de las diferentes condiciones')
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()



# como no se cumple una distribución normal entre los indices de los pacientes sanos, con crepitancias, sibilancias
#o ambas patologias se hace un analisis no parametrico, la prueba U de Mann-Whitney

import scipy.stats as stats

#rango entre sanos Vs Crepitancias
print('Prueba U MaNn whitney sanos Vs crepitancias Rango',stats.mannwhitneyu(df_sanos['Rango'],df_crepi['Rango']))
print('Prueba U MaNn whitney sanos Vs crepitancias Varianza',stats.mannwhitneyu(df_sanos['Varianza'],df_crepi['Varianza']))
print('Prueba U MaNn whitney sanos Vs crepitancias Promedio movil',stats.mannwhitneyu(df_sanos['Promedio movil'],df_crepi['Promedio movil']))
print('Prueba U MaNn whitney sanos Vs crepitancias Promedio espectro',stats.mannwhitneyu(df_sanos['Promedio espectro'],df_crepi['Promedio espectro']))



#rango entre sanos Vs Sibilancias
print('Prueba U MaNn whitney sanos Vs Sibilancias Rango',stats.mannwhitneyu(df_sanos['Rango'],df_sibi['Rango']))
print('Prueba U MaNn whitney sanos Vs Sibilancias Varianza',stats.mannwhitneyu(df_sanos['Varianza'],df_sibi['Varianza']))
print('Prueba U MaNn whitney sanos Vs Sibilancias Promedio movil',stats.mannwhitneyu(df_sanos['Promedio movil'],df_sibi['Promedio movil']))
print('Prueba U MaNn whitney sanos Vs Sibilancias Promedio espectro',stats.mannwhitneyu(df_sanos['Promedio espectro'],df_sibi['Promedio espectro']))


#rango entre Sanos Vs ambas patologias
print('Prueba U MaNn whitney Sanos Vs Ambas Patologías Rango',stats.mannwhitneyu(df_sanos['Rango'],df_ambas['Rango']))
print('Prueba U MaNn whitney Sanos Vs Ambas Patologías Varianza',stats.mannwhitneyu(df_sanos['Varianza'],df_ambas['Varianza']))
print('Prueba U MaNn whitney Sanos Vs Ambas Patologías Promedio movil',stats.mannwhitneyu(df_sanos['Promedio movil'],df_ambas['Promedio movil']))
print('Prueba U MaNn whitney Sanos Vs Ambas Patologías Promedio espectro',stats.mannwhitneyu(df_sanos['Promedio espectro'],df_ambas['Promedio espectro']))









